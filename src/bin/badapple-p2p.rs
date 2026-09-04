//! Bad Apple P2P mesh helper.
//!
//! This binary is spawned by the Swift daemon when P2P is enabled. It starts
//! a `ConnectionManager`, accepts peer connections, and encrypts all mesh
//! traffic with AES-256-GCM derived from the SLICKS secret.
//!
//! Subcommands:
//!   peers    - list connected peers
//!   sync     - broadcast a sync pulse and return peer count
//!   models   - list known models advertised by peers
//!   pull     - pull a model from a peer (placeholder)
//!   send     - send a model to a peer (placeholder)
//!   receive  - wait for an incoming model transfer (placeholder)
//!
//! Environment:
//!   BADAPPLE_P2P_SECRET      - 32+ byte pre-shared key (defaults to SLICKS secret)
//!   BADAPPLE_P2P_TCP_PORT    - TCP listen port (default 9876)
//!   BADAPPLE_P2P_WS_PORT     - WebSocket listen port (default 9877)
//!   BADAPPLE_P2P_PEERS       - comma-separated list of peers to connect to
//!   BADAPPLE_P2P_MAX_PEERS   - maximum concurrent peers (default 8)

use anyhow::{Context, Result};
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
        anyhow::bail!("usage: badapple-p2p <peers|sync|models|pull <peer_id> <model_id>|send <peer_id> <model_id>|receive [peer_id model_id]>");
    }

    match args[0].as_str() {
        "peers" => list_peers(),
        "sync" => do_sync(),
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
        _ => anyhow::bail!("unknown p2p subcommand: {}", args[0]),
    }
}

fn with_runtime<F, T>(f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    std::thread::spawn(f)
        .join()
        .expect("p2p runtime thread panicked")
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
    if let Ok(secret) = std::env::var("BADAPPLE_P2P_SECRET") {
        return Ok(secret.into_bytes());
    }
    if let Ok(path) = std::env::var("BADAPPLE_SLICKS_KEY_PATH") {
        let key = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read SLICKS key from {}", path))?;
        return Ok(key.trim().into());
    }
    // Fallback to a hard-coded dev secret. This is unsafe and should only be
    // used for local testing; production installs distribute keys out of band.
    Ok(b"bad-apple-p2p-dev-secret-do-not-use-in-prod".to_vec())
}

fn list_peers() -> Result<()> {
    with_runtime(|| {
        let rt = Runtime::new().expect("tokio runtime");
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
        let rt = Runtime::new().expect("tokio runtime");
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
            };
            cm.push_outgoing(packet);

            tokio::time::sleep(Duration::from_millis(800)).await;

            let count = cm.peer_count();
            println!("{{\"synced\": true, \"peer_count\": {}}}", count);
            Ok(())
        })
    })
}

fn model_dir() -> std::path::PathBuf {
    std::env::var("BADAPPLE_P2P_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/var/lib/bad_apple/p2p_models"))
}

fn transfer_service() -> Result<bad_apple::p2p_model::P2PModelTransfer> {
    let secret = resolve_secret()?;
    Ok(bad_apple::p2p_model::P2PModelTransfer::new(
        secret,
        model_dir(),
    ))
}

fn list_models() -> Result<()> {
    with_runtime(|| {
        let rt = Runtime::new().expect("tokio runtime");
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
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let svc = transfer_service()?;
            let output_dir = std::env::var("BADAPPLE_P2P_OUTPUT_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| model_dir());
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
                    std::process::exit(1);
                }
            }
            Ok(())
        })
    })
}

fn send_model(peer_id: &str, model_id: &str) -> Result<()> {
    // `send` starts a server and advertises that the named peer can pull.
    // This keeps the protocol symmetric: a model always moves from a listening
    // sender to an active puller.
    let peer_id = peer_id.to_string();
    let model_id = model_id.to_string();
    with_runtime(move || {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let svc = transfer_service()?;
            svc.scan_and_advertise(hostname()).await?;
            let manifests = svc.advertised_manifests().await;
            if !manifests.iter().any(|m| m.model_id == model_id) {
                anyhow::bail!("model {model_id} not found in {}", model_dir().display());
            }
            let port = std::env::var("BADAPPLE_P2P_TRANSFER_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(9878);
            let (addr, handle) = svc.serve(port).await?;
            println!("{{\"status\": \"serving\", \"peer\": \"{peer_id}\", \"model\": \"{model_id}\", \"addr\": \"{addr}\", \"message\": \"peer should run: badapple-p2p pull {addr} {model_id}\"}}");
            handle.await?
        })
    })
}

fn receive_model(_peer_id: Option<&str>, _model_id: Option<&str>) -> Result<()> {
    with_runtime(|| {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let svc = transfer_service()?;
            svc.scan_and_advertise(hostname()).await?;
            let port = std::env::var("BADAPPLE_P2P_TRANSFER_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(9878);
            let (addr, handle) = svc.serve(port).await?;
            println!(
                "{{\"status\": \"serving\", \"addr\": \"{addr}\", \"model_dir\": \"{}\"}}",
                model_dir().display()
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
