//! Bad Apple P2P mesh helper.
//!
//! This binary is spawned by the Swift daemon when P2P is enabled. It starts
//! a `ConnectionManager`, accepts peer connections, and encrypts all mesh
//! traffic with AES-256-GCM derived from the SLICKS secret.
//!
//! Subcommands:
//!   peers         - list connected peers
//!   sync          - broadcast a sync pulse and return peer count
//!   sync-doc      - broadcast a document kind (personas, prompt, settings, models)
//!   receive-mesh  - listen for incoming mesh sync packets and persist them
//!   models        - list known models advertised by peers
//!   pull          - pull a model from a peer
//!   send          - send a model to a peer
//!   receive       - wait for an incoming model transfer
//!
//! Environment:
//!   BADAPPLE_P2P_SECRET      - 32+ byte pre-shared key (defaults to SLICKS secret)
//!   BADAPPLE_P2P_TCP_PORT    - TCP listen port (default 9876)
//!   BADAPPLE_P2P_WS_PORT     - WebSocket listen port (default 9877)
//!   BADAPPLE_P2P_PEERS       - comma-separated list of peers to connect to
//!   BADAPPLE_P2P_MAX_PEERS   - maximum concurrent peers (default 8)

use anyhow::{bail, Context, Result};
use bad_apple::mesh_sync::{
    build_engram, handle_incoming, read_local_doc, MeshDocKind, MeshPacket, MeshStore,
};
use bad_apple::protocol::{CompactEngramPacket, ConnectionManager, LockFreeRing, SwarmMetrics};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;

fn main() {
    if let Err(e) = run() {
        eprintln!("badapple-p2p: {:#}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        anyhow::bail!("usage: badapple-p2p <peers|sync|sync-doc <kind>|receive-mesh [timeout_ms]|models|pull <peer_id> <model_id>|send <peer_id> <model_id>|receive [peer_id model_id]|ask <peer_id> <prompt...>>\n  doc kinds: personas, prompt, settings, models, checkpoint, ledger_checkpoint");
    }

    match args[0].as_str() {
        "peers" => list_peers(),
        "sync" => do_sync(),
        "sync-doc" => {
            if args.len() < 2 {
                anyhow::bail!("usage: badapple-p2p sync-doc <personas|prompt|settings|models|checkpoint|ledger_checkpoint>");
            }
            sync_doc(&args[1])
        }
        "receive-mesh" => {
            let timeout_ms = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);
            receive_mesh(timeout_ms)
        }
        "models" => list_models(),
        "pull" => {
            if args.len() < 3 {
                anyhow::bail!("usage: badapple-p2p pull <peer_id> <model_id>");
            }
            pull_model(&args[1], &args[2])
        }
        "send" => {
            if args.len() < 3 {
                anyhow::bail!("usage: badapple-p2p send <peer_id> <model_id>");
            }
            send_model(&args[1], &args[2])
        }
        "receive" => {
            let peer_id = args.get(1).map(String::as_str);
            let model_id = args.get(2).map(String::as_str);
            receive_model(peer_id, model_id)
        }
        "ask" => {
            if args.len() < 3 {
                anyhow::bail!("usage: badapple-p2p ask <peer_id> <prompt...>");
            }
            let prompt = args[2..].join(" ");
            ask_peer(&args[1], &prompt)
        }
        _ => anyhow::bail!("unknown p2p subcommand: {}", args[0]),
    }
}

fn with_runtime<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    std::thread::spawn(f)
        .join()
        .map_err(|_| anyhow::anyhow!("p2p runtime thread panicked"))?
}

fn build_manager() -> Result<Arc<ConnectionManager>> {
    let secret = resolve_secret()?;
    let incoming = Arc::new(LockFreeRing::<CompactEngramPacket>::new(1024));
    let metrics = Arc::new(Mutex::new(SwarmMetrics::default()));
    let max_peers = std::env::var("BADAPPLE_P2P_MAX_PEERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    Ok(Arc::new(ConnectionManager::new(
        secret,
        incoming,
        metrics,
        max_peers,
        Duration::from_secs(1),
        Duration::from_secs(60),
    )))
}

fn resolve_secret() -> Result<Vec<u8>> {
    let secret = if let Ok(secret) = std::env::var("BADAPPLE_P2P_SECRET") {
        if secret.is_empty() {
            bail!("BADAPPLE_P2P_SECRET is empty")
        }
        secret.into_bytes()
    } else if let Ok(path) = std::env::var("BADAPPLE_SLICKS_KEY_PATH") {
        let key = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read SLICKS key from {}", path))?;
        let key = key.trim();
        if key.is_empty() {
            bail!("SLICKS key file at {} is empty", path)
        }
        key.as_bytes().to_vec()
    } else {
        // No configured secret. Fail closed: never fall back to a hard-coded key.
        bail!("P2P requires BADAPPLE_P2P_SECRET or BADAPPLE_SLICKS_KEY_PATH")
    };
    if secret.len() < 32 {
        bail!("P2P secret must be at least 32 bytes")
    }
    Ok(secret)
}

fn list_peers() -> Result<()> {
    with_runtime(|| {
        let rt = Runtime::new().context("tokio runtime")?;
        rt.block_on(async {
            let cm = build_manager().context("failed to build connection manager")?;
            let tcp_port = std::env::var("BADAPPLE_P2P_TCP_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(9876);
            let ws_port = std::env::var("BADAPPLE_P2P_WS_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(9877);
            let _ = cm.start_server(tcp_port, ws_port).await;
            cm.start_outbound_sweeper();
            connect_to_peers(&cm);

            tokio::time::sleep(Duration::from_millis(500)).await;

            let peers = cm.active_peers();
            let result = serde_json::json!({
                "peers": peers.iter().map(|p| {
                    serde_json::json!({
                        "id": p.id,
                        "transport": p.transport_label(),
                        "addr": p.addr,
                    })
                }).collect::<Vec<_>>(),
                "count": peers.len(),
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        })
    })
}

fn do_sync() -> Result<()> {
    with_runtime(|| {
        let rt = Runtime::new().context("tokio runtime")?;
        rt.block_on(async {
            let cm = build_manager().context("failed to build connection manager")?;
            let tcp_port = std::env::var("BADAPPLE_P2P_TCP_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(9876);
            let ws_port = std::env::var("BADAPPLE_P2P_WS_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(9877);
            let _ = cm.start_server(tcp_port, ws_port).await;
            cm.start_outbound_sweeper();
            connect_to_peers(&cm);

            // Send a compact engram pulse to all peers.
            let packet = CompactEngramPacket {
                id: rand::random(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                experiential_text: "sync".to_string(),
                emotional_state_snapshot: "".to_string(),
                origin_instance: hostname(),
                brain_state: vec![0.0; bad_apple::protocol::ENGRAM_DIM],
                embedding: vec![0.0; bad_apple::protocol::EMBEDDING_DIM],
                priority: u8::MAX,
                payload: None,
            };
            cm.push_outgoing(packet);

            tokio::time::sleep(Duration::from_millis(800)).await;

            let count = cm.peer_count();
            println!("{{\"synced\": true, \"peer_count\": {}}}", count);
            Ok(())
        })
    })
}

fn sync_doc(kind: &str) -> Result<()> {
    let kind: MeshDocKind = kind.parse()?;
    with_runtime(move || {
        let rt = Runtime::new().context("tokio runtime")?;
        rt.block_on(async {
            let cm = build_manager().context("failed to build connection manager")?;
            let tcp_port = p2p_tcp_port();
            let ws_port = p2p_ws_port();
            let _ = cm.start_server(tcp_port, ws_port).await;
            cm.start_outbound_sweeper();
            connect_to_peers(&cm);

            // Wait a moment for outbound peers to connect.
            tokio::time::sleep(Duration::from_millis(400)).await;

            let doc = read_local_doc(kind).context("failed to read local document")?;
            let packet = MeshPacket::Push {
                doc_id: doc.doc_id(),
                doc,
            };
            let engram = build_engram(&packet, &hostname())?;
            cm.push_outgoing(engram);

            // Give peers time to receive and ack.
            tokio::time::sleep(Duration::from_millis(1200)).await;

            let count = cm.peer_count();
            println!(
                "{{\"status\": \"ok\", \"kind\": \"{}\", \"peers\": {}}}",
                kind, count
            );
            Ok(())
        })
    })
}

fn receive_mesh(timeout_ms: u64) -> Result<()> {
    with_runtime(move || {
        let rt = Runtime::new().context("tokio runtime")?;
        rt.block_on(async {
            let cm = build_manager().context("failed to build connection manager")?;
            let tcp_port = p2p_tcp_port();
            let ws_port = p2p_ws_port();
            let _ = cm.start_server(tcp_port, ws_port).await;
            cm.start_outbound_sweeper();
            connect_to_peers(&cm);

            let store = MeshStore::new(MeshStore::default_root())?;
            let incoming = cm.incoming.clone();
            let mut received = 0usize;
            let mut acks = 0usize;
            let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);

            while let Ok(Some(packet)) =
                tokio::time::timeout_at(deadline, incoming.pop_async()).await
            {
                if packet.payload.is_some() {
                    received += 1;
                    if let Some(reply) = handle_incoming(&packet, &store) {
                        let reply_engram = build_engram(&reply, &hostname())?;
                        cm.push_outgoing(reply_engram);
                        if matches!(reply, MeshPacket::Ack { .. }) {
                            acks += 1;
                        }
                    }
                }
            }

            println!(
                "{{\"status\": \"ok\", \"received\": {}, \"acks_sent\": {}}}",
                received, acks
            );
            Ok(())
        })
    })
}

fn p2p_tcp_port() -> u16 {
    std::env::var("BADAPPLE_P2P_TCP_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9876)
}

fn p2p_ws_port() -> u16 {
    std::env::var("BADAPPLE_P2P_WS_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9877)
}

fn model_dir() -> Result<std::path::PathBuf> {
    let raw = std::env::var("BADAPPLE_P2P_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/var/lib/bad_apple/p2p_models"));
    validate_p2p_dir(raw)
}

fn pull_output_dir() -> Result<std::path::PathBuf> {
    let raw = std::env::var("BADAPPLE_P2P_OUTPUT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/var/lib/bad_apple/p2p_models"));
    validate_p2p_dir(raw)
}

fn validate_p2p_dir(path: std::path::PathBuf) -> Result<std::path::PathBuf> {
    let abs = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()?.join(path)
    };
    if abs
        .components()
        .any(|c| c == std::path::Component::ParentDir)
    {
        bail!("P2P output directory must not contain '..' components");
    }
    let allowed = {
        let home = std::env::var("HOME").unwrap_or_default();
        [
            std::path::PathBuf::from("/var/lib/bad_apple"),
            std::path::PathBuf::from(home).join(".bad_apple"),
        ]
    };
    if !allowed.iter().any(|root| abs.starts_with(root)) {
        bail!("P2P output directory must be under /var/lib/bad_apple or ~/.bad_apple");
    }
    Ok(abs)
}

fn transfer_service() -> Result<bad_apple::p2p_model::P2PModelTransfer> {
    let secret = resolve_secret()?;
    Ok(bad_apple::p2p_model::P2PModelTransfer::new(
        secret,
        model_dir()?,
    ))
}

fn list_models() -> Result<()> {
    with_runtime(|| {
        let rt = Runtime::new().context("tokio runtime")?;
        rt.block_on(async {
            let svc = transfer_service()?;
            svc.scan_and_advertise(hostname()).await?;
            let manifests = svc.advertised_manifests().await;
            let result = serde_json::json!({
                "models": manifests.iter().map(|m| {
                    serde_json::json!({
                        "model_id": m.model_id,
                        "file_name": m.file_name,
                        "total_bytes": m.total_bytes,
                        "sha256": m.sha256,
                        "origin_peer": m.origin_peer,
                        "chunk_size": m.chunk_size,
                    })
                }).collect::<Vec<_>>(),
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        })
    })
}

fn pull_model(peer_id: &str, model_id: &str) -> Result<()> {
    let peer_id = peer_id.to_string();
    let model_id = model_id.to_string();
    with_runtime(move || {
        let rt = Runtime::new().context("tokio runtime")?;
        rt.block_on(async {
            let svc = transfer_service()?;
            let output_dir = pull_output_dir()?;
            let peer_addr = if peer_id.contains(':') {
                peer_id
            } else {
                format!("{peer_id}:9878")
            };
            match svc.pull(&peer_addr, &model_id, &[], &output_dir).await {
                Ok(path) => {
                    println!("{{\"status\": \"ok\", \"path\": \"{}\"}}", path.display());
                }
                Err(e) => {
                    eprintln!("{{\"status\": \"error\", \"message\": \"{e:#}\"}}");
                    bail!("{e:#}");
                }
            }
            Ok(())
        })
    })
}

fn send_model(peer_id: &str, model_id: &str) -> Result<()> {
    // `send` now actively pushes a model to a listening peer.
    let peer_id = peer_id.to_string();
    let model_id = model_id.to_string();
    with_runtime(move || {
        let rt = Runtime::new().context("tokio runtime")?;
        rt.block_on(async {
            let svc = transfer_service()?;
            let peer_addr = if peer_id.contains(':') {
                peer_id
            } else {
                format!("{peer_id}:9878")
            };
            match svc.push(&peer_addr, &model_id).await {
                Ok(()) => {
                    println!("{{\"status\": \"ok\", \"peer\": \"{peer_addr}\", \"model\": \"{model_id}\"}}");
                    Ok(())
                }
                Err(e) => {
                    eprintln!("{{\"status\": \"error\", \"message\": \"{e:#}\"}}");
                    bail!("{e:#}");
                }
            }
        })
    })
}

/// Delegated inference: send a prompt to a trusted peer's engine and print
/// its answer. The peer must serve with `BADAPPLE_P2P_INFER=1`; the action is
/// attested on the serving machine's ledger under this machine's peer label.
fn ask_peer(peer_id: &str, prompt: &str) -> Result<()> {
    let peer_id = peer_id.to_string();
    let prompt = prompt.to_string();
    with_runtime(move || {
        let rt = Runtime::new().context("tokio runtime")?;
        rt.block_on(async {
            let svc = transfer_service()?;
            let peer_addr = if peer_id.contains(':') {
                peer_id
            } else {
                format!("{peer_id}:9878")
            };
            match svc.infer(&peer_addr, &prompt, 512, &hostname()).await {
                Ok(resp) => {
                    println!("{}", resp.text);
                    eprintln!(
                        "\n[served by {peer_addr}{} in {} ms — attested on the serving peer's ledger]",
                        resp.tier.map(|t| format!(" · tier {t}")).unwrap_or_default(),
                        resp.elapsed_ms
                    );
                    Ok(())
                }
                Err(e) => {
                    eprintln!("{{\"status\": \"error\", \"message\": \"{e:#}\"}}");
                    bail!("{e:#}");
                }
            }
        })
    })
}

fn receive_model(_peer_id: Option<&str>, _model_id: Option<&str>) -> Result<()> {
    with_runtime(|| {
        let rt = Runtime::new().context("tokio runtime")?;
        rt.block_on(async {
            let svc = transfer_service()?;
            svc.scan_and_advertise(hostname()).await?;
            let port = std::env::var("BADAPPLE_P2P_TRANSFER_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(9878);
            let (addr, handle) = svc.serve(port).await?;
            let dir = model_dir()?;
            println!(
                "{{\"status\": \"serving\", \"addr\": \"{addr}\", \"model_dir\": \"{}\"}}",
                dir.display()
            );
            handle.await?
        })
    })
}

fn connect_to_peers(cm: &Arc<ConnectionManager>) {
    if let Ok(peers) = std::env::var("BADAPPLE_P2P_PEERS") {
        let list: Vec<String> = peers
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let cm = cm.clone();
        tokio::spawn(async move {
            cm.connect_to_peers(list).await;
        });
    }
}

fn hostname() -> String {
    std::env::var("BADAPPLE_ORIGIN_INSTANCE").unwrap_or_else(|_| {
        std::process::Command::new("hostname")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    })
}
