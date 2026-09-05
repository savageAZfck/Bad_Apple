use anyhow::{anyhow, bail, Context, Result};
use bad_apple::automation_cage::{
    parse_actions as parse_cage_actions, Action as CageAction, AutomationCage,
};
use bad_apple::bad_apple_ipc::{
    client_proof, load_slicks_secret, now_unix_ms, random_nonce, read_frame, server_proof,
    socket_path, validate_request, verify_client_proof, verify_server_proof, ClientFrame,
    ReplayCache, ServerFrame, MAX_FRAME_BYTES, SLICKS_VERSION, SLICKS_VERSION_2,
};
use bad_apple::tensor_brain::{text_to_grounded_embedding, CandleBrain};
use bad_apple::wasm_cage::WasmCage;
use base64::{engine::general_purpose, Engine as _};
use rand::Rng;
use regex::Regex;
use std::fs;
use std::io::{BufReader, Write};
use std::os::unix::fs::{chown, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MLX_SOCKET_PATH: &str = "/var/run/badapple/substrate_mlx.sock";
const LOW_COMPLEXITY_THRESHOLD: f64 = 0.62;
const REPLAY_CACHE_MAX: usize = 4096;

/// One classification request sent to the brain worker thread.
struct ClassifyRequest {
    text: String,
    reply: Sender<f64>,
}

/// Native semantic router. The 576-D Candle transformer produces a brain state
/// and the conscience head projects it to a 2-class logit (low / high).
#[derive(Clone)]
struct SemanticRouter {
    request_tx: Sender<ClassifyRequest>,
}

impl SemanticRouter {
    fn new() -> Result<Self> {
        let (request_tx, request_rx) = channel::<ClassifyRequest>();

        std::thread::spawn(move || {
            let mut brain =
                match CandleBrain::new("gatekeeper", 2, &bad_apple::tensor_brain::layer_dims()) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("[gatekeeper] failed to initialize CandleBrain: {e}");
                        std::process::exit(1);
                    }
                };

            let env_path = std::env::var("BADAPPLE_GATEKEEPER_WEIGHTS")
                .map(std::path::PathBuf::from)
                .ok();
            let exe_path = std::env::current_exe().ok().and_then(|exe| {
                exe.parent()
                    .and_then(|p| p.parent())
                    .and_then(|p| p.parent())
                    .map(|root| root.join("data").join("gatekeeper.safetensors"))
            });
            let default_path =
                std::path::PathBuf::from("/var/lib/bad_apple/data/gatekeeper.safetensors");
            let weights_path = env_path.or(exe_path).unwrap_or(default_path);
            if let Err(e) = brain.load_weights(&weights_path) {
                eprintln!("[gatekeeper] no trained weights at {weights_path:?}: {e} (using untrained brain)");
            } else {
                eprintln!("[gatekeeper] loaded trained weights from {weights_path:?}");
            }
            // Keep System 2 active so the full Transformer participates in classification.
            brain.set_system2_active(true);

            let axes = [0.0, 0.0, 0.0, 0.0];

            while let Ok(req) = request_rx.recv() {
                let score = classifier_score(&brain, &req.text, &axes);
                let _ = req.reply.send(score);
            }
        });

        Ok(Self { request_tx })
    }

    fn classify(&self, text: &str) -> Result<f64> {
        let (tx, rx) = channel();
        self.request_tx
            .send(ClassifyRequest {
                text: text.to_string(),
                reply: tx,
            })
            .context("brain worker thread is dead")?;
        rx.recv().context("brain worker did not respond")
    }
}

fn softmax_1(logits: &[f64]) -> f64 {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 0.0 {
        exps[1] / sum
    } else {
        0.5
    }
}

fn classifier_score(brain: &CandleBrain, text: &str, axes: &[f64; 4]) -> f64 {
    let emb = text_to_grounded_embedding(text, axes);
    match brain.classify(&emb) {
        Ok(logits) => softmax_1(&logits),
        Err(e) => {
            eprintln!("[gatekeeper] classify failed: {e}");
            0.5
        }
    }
}

enum FastAction {
    Time,
    OpenWorkspace(String),
    OpenApp(String),
    CreateDirectory(String),
    CreateFile(String),
    ListDirectory(String),
    Delete(String),
    CopyFile { from: String, to: String },
    MoveFile { from: String, to: String },
    RunWasm(String),
    NewChat,
}

#[derive(Clone)]
struct FastActionResolver {
    open_workspace: Regex,
    open_app: Regex,
    create_file: Regex,
    create_dir: Regex,
    list_dir: Regex,
    delete: Regex,
    copy_file: Regex,
    move_file: Regex,
    run_wasm: Regex,
    time: Regex,
}

impl FastActionResolver {
    fn new() -> Result<Self> {
        Ok(Self {
            open_workspace: Regex::new(r"(?i)\bopen\b(?:\s+\w+){0,3}\s+(?:workspace|folder|repo|directory)\s+(?:my\s+)?([a-z0-9_\-\.]+)").context("open_workspace regex")?,
            open_app: Regex::new(r"(?i)\b(open|launch)\b(?:\s+\w+){0,2}\s+(?:the\s+)?([a-z0-9_\-\.]+(?:\.app)?)").context("open_app regex")?,
            create_file: Regex::new(r"(?i)\bcreate\b(?:\s+\w+){0,3}\s+(?:file)\s+(?:at\s+)?([~/a-z0-9_\.\-\s/]+)").context("create_file regex")?,
            create_dir: Regex::new(r"(?i)\bcreate\b(?:\s+\w+){0,3}\s+(?:directory|folder)\s+(?:at\s+)?([~/a-z0-9_\.\-\s/]+)").context("create_dir regex")?,
            list_dir: Regex::new(r"(?i)\b(list|show)\b(?:\s+\w+){0,3}\s+(?:files|contents|in)?\s+(?:of\s+)?(.+?)(?:\s+(?:and|then|or)\b|$)").context("list_dir regex")?,
            delete: Regex::new(r"(?i)\b(delete|remove|trash)\b(?:\s+\w+){0,3}\s+([~/a-z0-9_\.\-\s/]+)").context("delete regex")?,
            copy_file: Regex::new(r"(?i)\bcopy\b(?:\s+\w+){0,3}\s+([~/a-z0-9_\.\-\s/]+)\s+(?:to\s+)?([~/a-z0-9_\.\-\s/]+)").context("copy_file regex")?,
            move_file: Regex::new(r"(?i)\b(move)\b(?:\s+\w+){0,3}\s+([~/a-z0-9_\.\-\s/]+)\s+(?:to\s+)?([~/a-z0-9_\.\-\s/]+)").context("move_file regex")?,
            run_wasm: Regex::new(r"(?i)\b(?:run|execute)\b(?:\s+\w+){0,3}\s+(?:wasm\s+)?script\s+([~/a-z0-9_\.\-\s/]+\.wasm)").context("run_wasm regex")?,
            time: Regex::new(r"(?i)^(what'?s?\s+(?:the\s+)?time|what\s+time\s+is\s+it|current\s+time|time\s+is\s+it|clock|what\s+hour\s+is\s+it|qu[eé]\s+hora\s+es)\b").context("time regex")?,
        })
    }

    fn resolve(&self, prompt: &str) -> Option<FastAction> {
        let lower = prompt.to_lowercase();

        if self.time.is_match(prompt) {
            return Some(FastAction::Time);
        }

        if let Some(cap) = self.open_workspace.captures(prompt) {
            let name = cap.get(1)?.as_str();
            let home = dirs::home_dir()?;
            let candidate = if name == "bad_apple" || name == "badapple" {
                home.join("bad_apple")
            } else if name == "firefly" || name == "firefly_inferno" {
                home.join("firefly_inferno")
            } else {
                home.join(name)
            };
            return Some(FastAction::OpenWorkspace(
                candidate.to_string_lossy().into_owned(),
            ));
        }

        if let Some(cap) = self.open_app.captures(prompt) {
            let app = cap.get(2)?.as_str().trim_end_matches(".app").to_string();
            return Some(FastAction::OpenApp(app));
        }

        if let Some(cap) = self.create_file.captures(prompt) {
            return Some(FastAction::CreateFile(expand_path(cap.get(1)?.as_str())));
        }

        if let Some(cap) = self.create_dir.captures(prompt) {
            return Some(FastAction::CreateDirectory(expand_path(
                cap.get(1)?.as_str(),
            )));
        }

        if let Some(cap) = self.list_dir.captures(prompt) {
            return Some(FastAction::ListDirectory(expand_path(cap.get(2)?.as_str())));
        }

        if let Some(cap) = self.copy_file.captures(prompt) {
            return Some(FastAction::CopyFile {
                from: expand_path(cap.get(1)?.as_str()),
                to: expand_path(cap.get(2)?.as_str()),
            });
        }

        if let Some(cap) = self.move_file.captures(prompt) {
            return Some(FastAction::MoveFile {
                from: expand_path(cap.get(1)?.as_str()),
                to: expand_path(cap.get(2)?.as_str()),
            });
        }

        if let Some(cap) = self.run_wasm.captures(prompt) {
            return Some(FastAction::RunWasm(expand_path(cap.get(1)?.as_str())));
        }

        if let Some(cap) = self.delete.captures(prompt) {
            return Some(FastAction::Delete(expand_path(cap.get(2)?.as_str())));
        }

        if lower.contains("new chat") || lower.contains("clear chat") {
            return Some(FastAction::NewChat);
        }

        None
    }
}

fn expand_path(p: &str) -> String {
    if p.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.to_string_lossy().into_owned() + &p[1..];
        }
    }
    p.to_string()
}

/// Check that a path is inside one of the automation cage roots.
fn is_allowed_path(path: &str, cage: &AutomationCage) -> bool {
    let expanded = expand_path(path);
    let target = PathBuf::from(&expanded);
    // Reject broken/dangling symlinks: symlink_metadata succeeds on a symlink
    // even if its target is missing, while canonicalize would fail.
    if let Ok(meta) = fs::symlink_metadata(&target) {
        if meta.file_type().is_symlink() {
            // Follow the symlink and re-check the resolved target.
            let Ok(canon) = target.canonicalize() else {
                return false;
            };
            return cage.roots().iter().any(|r| canon.starts_with(r));
        }
    }
    let Ok(canon) = target.canonicalize() else {
        // For non-existent files, canonicalize the parent.
        let Some(parent) = target.parent() else {
            return false;
        };
        let Ok(parent_canon) = parent.canonicalize() else {
            return false;
        };
        return cage.roots().iter().any(|r| parent_canon.starts_with(r));
    };
    cage.roots().iter().any(|r| canon.starts_with(r))
}

fn list_directory_safe(path: &str, cage: &AutomationCage) -> Result<String> {
    let expanded = expand_path(path);
    if !is_allowed_path(path, cage) {
        bail!("path escapes allowlisted automation roots");
    }
    let entries: Vec<_> = fs::read_dir(&expanded)?
        .filter_map(std::result::Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .take(20)
        .collect();
    if entries.is_empty() {
        Ok("Nothing there, babe.".to_string())
    } else {
        Ok(format!("In {expanded}: {}", entries.join(", ")))
    }
}

fn execute_fast(action: FastAction, cage: &AutomationCage) -> Result<String> {
    let reply = match action {
        FastAction::Time => {
            let out = Command::new("date").arg("+%I:%M %p").output()?;
            let time = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let tails = [
                format!("It's {time}, babe."),
                format!("{time}, homie."),
                format!("{time}, hun."),
            ];
            let idx = rand::thread_rng().gen_range(0..tails.len());
            tails[idx].clone()
        }
        FastAction::OpenWorkspace(path) | FastAction::OpenApp(path) => {
            Command::new("open").arg(&path).spawn()?;
            format!("Opening {path}...")
        }
        FastAction::CreateFile(path) => {
            let action = CageAction::CreateFile {
                path: PathBuf::from(expand_path(&path)),
            };
            let report = cage.execute(&action)?;
            format!(
                "Created file in {} ({})",
                report.paths[0].display(),
                report.result
            )
        }
        FastAction::CreateDirectory(path) => {
            let action = CageAction::CreateDirectory {
                path: PathBuf::from(expand_path(&path)),
            };
            let report = cage.execute(&action)?;
            format!(
                "Created directory in {} ({})",
                report.paths[0].display(),
                report.result
            )
        }
        FastAction::ListDirectory(path) => list_directory_safe(&path, cage)?,
        FastAction::Delete(path) => {
            let action = CageAction::MoveToTrash {
                path: PathBuf::from(expand_path(&path)),
            };
            let report = cage.execute(&action)?;
            format!(
                "Moved {} to the Trash ({}).",
                report.paths[0].display(),
                report.result
            )
        }
        FastAction::CopyFile { from, to } => {
            let action = CageAction::CopyFile {
                source: PathBuf::from(expand_path(&from)),
                destination: PathBuf::from(expand_path(&to)),
            };
            let report = cage.execute(&action)?;
            format!(
                "Copied {} to {} ({}).",
                report.paths[0].display(),
                report.paths[1].display(),
                report.result
            )
        }
        FastAction::MoveFile { from, to } => {
            let action = CageAction::MoveFile {
                source: PathBuf::from(expand_path(&from)),
                destination: PathBuf::from(expand_path(&to)),
            };
            let report = cage.execute(&action)?;
            format!(
                "Moved {} to {} ({}).",
                report.paths[0].display(),
                report.paths[1].display(),
                report.result
            )
        }
        FastAction::RunWasm(path) => {
            let expanded = expand_path(&path);
            if !is_allowed_path(&expanded, cage) {
                bail!("wasm path escapes allowlisted automation roots");
            }
            let wasm_bytes =
                fs::read(&expanded).with_context(|| format!("cannot read WASM {expanded}"))?;
            let mut cage =
                WasmCage::new().map_err(|e| anyhow!("WasmCage init failed: {}", e.reason))?;
            cage.compile(&wasm_bytes)
                .map_err(|e| anyhow!("WASM compile failed: {}", e.reason))?;
            let output = cage
                .run_with_input(b"")
                .map_err(|e| anyhow!("WASM run failed: {}", e.reason))?;
            format!("WASM ran: {output}")
        }
        FastAction::NewChat => "new chat".to_string(),
    };
    Ok(reply)
}

/// Scan the 8B response for ```badapple-action blocks, validate them in the
/// AutomationCage, and execute them. Returns the original text if none found,
/// or a report string if actions were executed.
fn execute_cage_blocks(text: &str, cage: &AutomationCage) -> String {
    match parse_cage_actions(text) {
        Ok(actions) if !actions.is_empty() => {
            let mut reports = Vec::new();
            for action in actions {
                match cage.execute(&action) {
                    Ok(report) => reports.push(format!(
                        "{} {} -> {} ({} ms)",
                        report.operation,
                        report
                            .paths
                            .iter()
                            .map(|p| p.display().to_string())
                            .collect::<Vec<_>>()
                            .join(" "),
                        report.result,
                        report.elapsed_ms
                    )),
                    Err(e) => reports.push(format!("automation error: {e:#}")),
                }
            }
            reports.join("\n")
        }
        _ => text.to_string(),
    }
}

/// Scan the 8B response for ```badapple-wasm blocks. The block may be a path to
/// a .wasm file or a base64 blob. Execute in the WasmCage and return outputs.
fn execute_wasm_blocks(text: &str, cage: &AutomationCage) -> String {
    let Ok(re) = Regex::new(r"(?s)```badapple-wasm\s*(.*?)\s*```") else {
        // The pattern is a hardcoded constant; if it ever fails to compile,
        // return the text unchanged rather than panicking.
        return text.to_string();
    };
    let mut outputs = Vec::new();
    let mut replaced = text.to_string();

    for cap in re.captures_iter(text) {
        let Some(body_match) = cap.get(1) else {
            continue;
        };
        let body = body_match.as_str().trim();
        let result: Result<String> = (|| {
            let bytes = if body.starts_with('/')
                || body.starts_with('~')
                || std::path::Path::new(body)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("wasm"))
            {
                let path = expand_path(body);
                if !is_allowed_path(&path, cage) {
                    bail!("wasm path escapes allowlisted automation roots");
                }
                fs::read(&path).with_context(|| format!("cannot read WASM {path}"))?
            } else {
                general_purpose::STANDARD
                    .decode(body)
                    .context("invalid base64 in badapple-wasm block")?
            };
            let mut cage = WasmCage::new().map_err(|e| anyhow!("WasmCage init: {}", e.reason))?;
            cage.compile(&bytes)
                .map_err(|e| anyhow!("WASM compile: {}", e.reason))?;
            let out = cage
                .run_with_input(b"")
                .map_err(|e| anyhow!("WASM run: {}", e.reason))?;
            Ok(out)
        })();

        match result {
            Ok(out) => outputs.push(format!("[wasm output] {out}")),
            Err(e) => outputs.push(format!("[wasm error] {e:#}")),
        }
        // Remove the block from the display text and append the execution report.
        if let Some(full_match) = cap.get(0) {
            replaced = replaced.replace(full_match.as_str(), "");
        }
    }

    if outputs.is_empty() {
        return text.to_string();
    }

    let trimmed = replaced.trim();
    if trimmed.is_empty() {
        outputs.join("\n")
    } else {
        format!("{}\n\n{}", trimmed, outputs.join("\n"))
    }
}

/// Combine cage and wasm post-processing.
fn post_process_response(text: &str, cage: &AutomationCage) -> String {
    let text = execute_cage_blocks(text, cage);
    execute_wasm_blocks(&text, cage)
}

/// Forward an execute request to the Python MLX server and stream back responses.
fn forward_to_mlx(
    prompt: &str,
    max_new_tokens: usize,
    writer: &mut UnixStream,
    cage: &AutomationCage,
) -> Result<()> {
    let secret = load_slicks_secret().context("unable to load SLICKS secret for MLX backend")?;
    let mlx_path = PathBuf::from(
        std::env::var_os("BADAPPLE_MLX_SOCKET_PATH").unwrap_or_else(|| MLX_SOCKET_PATH.into()),
    );

    let mut mlx_stream = UnixStream::connect(&mlx_path)
        .with_context(|| format!("cannot connect to MLX backend at {mlx_path:?}"))?;
    mlx_stream.set_read_timeout(Some(Duration::from_mins(2)))?;
    mlx_stream.set_write_timeout(Some(Duration::from_secs(30)))?;

    let timestamp_ms = now_unix_ms()?;
    let client_nonce = random_nonce();

    let hello = ClientFrame::Hello {
        version: SLICKS_VERSION,
        timestamp_ms,
        client_nonce: client_nonce.clone(),
        client_pubkey: None,
    };
    write_frame(&mut mlx_stream, &hello)?;

    let mut reader = BufReader::new(mlx_stream.try_clone()?);
    let challenge: ServerFrame = read_frame(&mut reader)?;
    let (server_nonce, proof) = match challenge {
        ServerFrame::Challenge {
            version,
            server_nonce,
            proof,
            server_pubkey: _,
        } if version == SLICKS_VERSION => (server_nonce, proof),
        ServerFrame::Error { message } => bail!("MLX backend rejected handshake: {message}"),
        _ => bail!("MLX backend returned an invalid SLICKS challenge"),
    };

    if !verify_server_proof(&secret, timestamp_ms, &client_nonce, &server_nonce, &proof) {
        bail!("MLX backend failed SLICKS server authentication");
    }

    let proof = client_proof(
        &secret,
        timestamp_ms,
        &client_nonce,
        &server_nonce,
        prompt,
        max_new_tokens,
    );
    let execute = ClientFrame::Execute {
        version: SLICKS_VERSION,
        timestamp_ms,
        client_nonce,
        server_nonce,
        prompt: prompt.to_string(),
        max_new_tokens,
        proof,
        client_pubkey: None,
    };
    write_frame(&mut mlx_stream, &execute)?;

    // Stream accepted / token / done / error back to the client.
    let mut accepted = false;
    loop {
        let frame: ServerFrame = read_frame(&mut reader)?;
        match frame {
            ServerFrame::Accepted => {
                accepted = true;
                write_frame(writer, &ServerFrame::Accepted)?;
            }
            ServerFrame::Token { text } if accepted => {
                write_frame(writer, &ServerFrame::Token { text })?;
            }
            ServerFrame::Done { text, metrics } if accepted => {
                let text = post_process_response(&text, cage);
                write_frame(writer, &ServerFrame::Done { text, metrics })?;
                return Ok(());
            }
            ServerFrame::Response { result } if accepted => {
                write_frame(writer, &ServerFrame::Response { result })?;
                return Ok(());
            }
            ServerFrame::Error { message } => {
                write_frame(writer, &ServerFrame::Error { message })?;
                return Ok(());
            }
            _ => bail!("MLX backend returned an out-of-order IPC frame"),
        }
    }
}

/// Transparent v2 proxy: the gatekeeper does not hold a Secure Enclave key, so it
/// forwards the end-to-end handshake between the client and the MLX daemon. It still
/// applies automation-cage / wasm post-processing to the final response.
fn forward_v2_to_mlx(
    client_writer: &mut UnixStream,
    client_reader: &mut BufReader<UnixStream>,
    hello: ClientFrame,
    cage: &AutomationCage,
) -> Result<()> {
    let mlx_path = std::env::var_os("BADAPPLE_MLX_SOCKET_PATH")
        .map_or_else(|| PathBuf::from(MLX_SOCKET_PATH), PathBuf::from);
    let mut mlx_stream =
        UnixStream::connect(&mlx_path).context("unable to connect to MLX daemon for v2 proxy")?;
    mlx_stream.set_read_timeout(Some(Duration::from_mins(15)))?;
    mlx_stream.set_write_timeout(Some(Duration::from_secs(30)))?;

    let mut mlx_reader = BufReader::new(mlx_stream.try_clone()?);

    // Client Hello -> MLX
    write_frame(&mut mlx_stream, &hello)?;

    // MLX Challenge -> Client
    let challenge: ServerFrame = read_frame(&mut mlx_reader)?;
    match challenge {
        ServerFrame::Challenge { .. } | ServerFrame::Error { .. } => {
            write_frame(client_writer, &challenge)?;
        }
        _ => bail!("MLX daemon returned an invalid v2 challenge"),
    }

    // Client Execute -> MLX
    let execute: ClientFrame = read_frame(client_reader)?;
    // Validate the v2 Execute frame before forwarding — don't blindly proxy
    // unvalidated requests to the MLX backend.
    if let ClientFrame::Execute {
        version,
        prompt,
        max_new_tokens,
        ..
    } = &execute
    {
        if *version != SLICKS_VERSION_2 {
            write_frame(
                client_writer,
                &ServerFrame::Error {
                    message: "invalid v2 execute frame".to_string(),
                },
            )?;
            bail!("client sent a non-v2 execute frame on v2 path");
        }
        if let Err(e) = validate_request(prompt, *max_new_tokens) {
            write_frame(
                client_writer,
                &ServerFrame::Error {
                    message: e.to_string(),
                },
            )?;
            return Ok(());
        }
    } else {
        write_frame(
            client_writer,
            &ServerFrame::Error {
                message: "invalid v2 execute frame".to_string(),
            },
        )?;
        bail!("client sent a non-execute frame on v2 path");
    }
    write_frame(&mut mlx_stream, &execute)?;

    // Stream MLX frames back to the client, post-processing Done text.
    let mut accepted = false;
    loop {
        let frame: ServerFrame = read_frame(&mut mlx_reader)?;
        match frame {
            ServerFrame::Accepted => {
                accepted = true;
                write_frame(client_writer, &ServerFrame::Accepted)?;
            }
            ServerFrame::Token { text } if accepted => {
                write_frame(client_writer, &ServerFrame::Token { text })?;
            }
            ServerFrame::Done { text, metrics } if accepted => {
                let text = post_process_response(&text, cage);
                write_frame(client_writer, &ServerFrame::Done { text, metrics })?;
                return Ok(());
            }
            ServerFrame::Response { result } if accepted => {
                write_frame(client_writer, &ServerFrame::Response { result })?;
                return Ok(());
            }
            ServerFrame::Error { message } => {
                write_frame(client_writer, &ServerFrame::Error { message })?;
                return Ok(());
            }
            _ => bail!("MLX daemon returned an out-of-order v2 frame"),
        }
    }
}

fn write_frame<W: Write, T: serde::Serialize>(writer: &mut W, value: &T) -> Result<()> {
    let mut frame = serde_json::to_vec(value)?;
    if frame.len() > MAX_FRAME_BYTES {
        bail!("IPC frame exceeds size limit");
    }
    frame.push(b'\n');
    writer.write_all(&frame)?;
    writer.flush()?;
    Ok(())
}

/// Recognize the class of I/O error produced when a client (very often a
/// bare TCP/Unix-socket health-check probe, e.g. badapple_supervisor.py's
/// `_socket_ready`) connects and disconnects without completing the SLICKS
/// handshake. On macOS in particular, a client that closes its socket in a
/// tight race right after `connect()` can surface to the server's `read()`
/// as `EINVAL` rather than a clean end-of-stream, in addition to the more
/// familiar broken-pipe/connection-reset/unexpected-EOF kinds seen on other
/// platforms or in other early-disconnect timings. None of these indicate a
/// real protocol or server problem, so they are logged quietly rather than
/// as alarming "client error" lines.
fn is_benign_disconnect(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    if msg.contains("connection closed") || msg.contains("out-of-order IPC frame") {
        return true;
    }
    // anyhow::Error::downcast_ref searches the entire cause chain, not just
    // the outermost error, so this finds an io::Error wrapped at any depth.
    let Some(io_err) = err.downcast_ref::<std::io::Error>() else {
        return false;
    };
    match io_err.kind() {
        std::io::ErrorKind::UnexpectedEof
        | std::io::ErrorKind::ConnectionReset
        | std::io::ErrorKind::ConnectionAborted
        | std::io::ErrorKind::BrokenPipe => true,
        std::io::ErrorKind::InvalidInput => io_err.raw_os_error() == Some(22), // EINVAL
        _ => false,
    }
}

fn handle_client(
    mut stream: UnixStream,
    router: &SemanticRouter,
    resolver: &FastActionResolver,
    cage: &AutomationCage,
    replay_cache: &ReplayCache,
) -> Result<()> {
    let secret = load_slicks_secret()?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_mins(2)))?;

    let mut reader = BufReader::new(stream.try_clone()?);

    // 1) Hello
    let hello: ClientFrame = read_frame(&mut reader)?;
    let (client_ts, client_nonce, client_version) = match &hello {
        ClientFrame::Hello {
            version,
            timestamp_ms,
            client_nonce,
            client_pubkey: _,
        } if (*version == SLICKS_VERSION || *version == SLICKS_VERSION_2)
            && bad_apple::bad_apple_ipc::nonce_is_valid(client_nonce) =>
        {
            (*timestamp_ms, client_nonce.clone(), *version)
        }
        _ => {
            write_frame(
                &mut stream,
                &ServerFrame::Error {
                    message: "invalid hello".to_string(),
                },
            )?;
            bail!("invalid hello frame");
        }
    };

    if !bad_apple::bad_apple_ipc::timestamp_is_fresh(client_ts) {
        write_frame(
            &mut stream,
            &ServerFrame::Error {
                message: "stale timestamp".to_string(),
            },
        )?;
        bail!("stale hello timestamp");
    }

    // v2 is end-to-end proxied; the gatekeeper does not hold a Secure Enclave key.
    if client_version == SLICKS_VERSION_2 {
        return forward_v2_to_mlx(&mut stream, &mut reader, hello, cage);
    }

    // 2) Challenge
    let _timestamp_ms = now_unix_ms()?;
    let server_nonce = random_nonce();
    let proof = server_proof(&secret, client_ts, &client_nonce, &server_nonce);
    write_frame(
        &mut stream,
        &ServerFrame::Challenge {
            version: SLICKS_VERSION,
            server_nonce: server_nonce.clone(),
            proof,
            server_pubkey: None,
        },
    )?;

    // 3) Execute
    let execute: ClientFrame = read_frame(&mut reader)?;
    let (prompt, max_new_tokens) = match execute {
        ClientFrame::Execute {
            version,
            timestamp_ms: exec_ts,
            client_nonce: exec_client_nonce,
            server_nonce: exec_server_nonce,
            prompt,
            max_new_tokens,
            proof,
            client_pubkey: _,
        } if version == SLICKS_VERSION
            && exec_client_nonce == client_nonce
            && exec_server_nonce == server_nonce =>
        {
            if !bad_apple::bad_apple_ipc::timestamp_is_fresh(exec_ts) {
                bail!("stale execute timestamp");
            }
            // Replay protection: reject duplicate (client_nonce, server_nonce) pairs
            if !replay_cache.check_and_insert(&client_nonce, &server_nonce) {
                write_frame(
                    &mut stream,
                    &ServerFrame::Error {
                        message: "replay detected".to_string(),
                    },
                )?;
                bail!("replay detected: nonce pair already used");
            }
            if !verify_client_proof(
                &secret,
                exec_ts,
                &client_nonce,
                &server_nonce,
                &prompt,
                max_new_tokens,
                &proof,
            ) {
                bail!("invalid client proof");
            }
            if let Err(e) = validate_request(&prompt, max_new_tokens) {
                write_frame(
                    &mut stream,
                    &ServerFrame::Error {
                        message: e.to_string(),
                    },
                )?;
                return Ok(());
            }
            (prompt, max_new_tokens)
        }
        _ => {
            write_frame(
                &mut stream,
                &ServerFrame::Error {
                    message: "invalid execute frame".to_string(),
                },
            )?;
            bail!("invalid execute frame");
        }
    };

    // 4) Classify and route. The 576-D trained brain produces a semantic
    // complexity score. Low = structural command, high = abstract reasoning.
    // BADAPPLE_COGNITIVE=0 disables the cognitive layer entirely, routing all
    // queries to the deep MLX core without classification.
    let cognitive_enabled = std::env::var("BADAPPLE_COGNITIVE")
        .map(|v| {
            !matches!(
                v.as_str(),
                "0" | "false" | "no" | "off" | "disable" | "disabled"
            )
        })
        .unwrap_or(true);

    if !cognitive_enabled {
        eprintln!(
            "[gatekeeper] cognitive layer disabled (BADAPPLE_COGNITIVE=0), routing directly to MLX"
        );
        forward_to_mlx(&prompt, max_new_tokens, &mut stream, cage)?;
        return Ok(());
    }

    let start = Instant::now();
    let score = router.classify(&prompt)?;
    let classify_us = start.elapsed().as_micros();
    eprintln!("[gatekeeper] '{prompt}' -> complexity {score:.3} (classify {classify_us} us)");

    if score < LOW_COMPLEXITY_THRESHOLD {
        if let Some(action) = resolver.resolve(&prompt) {
            match execute_fast(action, cage) {
                Ok(reply) if reply == "new chat" => {
                    // The deep core is responsible for clearing state; fall through.
                }
                Ok(reply) => {
                    write_frame(&mut stream, &ServerFrame::Accepted)?;
                    write_frame(
                        &mut stream,
                        &ServerFrame::Done {
                            text: reply,
                            metrics: Some(bad_apple::bad_apple_ipc::Metrics {
                                tier: Some("fast_action".to_string()),
                                ..Default::default()
                            }),
                        },
                    )?;
                    return Ok(());
                }
                Err(e) => {
                    // Fast path failed (e.g. path not in automation roots). Let the
                    // deep MLX core try to handle it with its own tool sandbox.
                    eprintln!("[gatekeeper] fast action failed, falling through to MLX: {e:#}");
                }
            }
        }
    }

    // 5) Deep path: wake the MLX core and stream.
    forward_to_mlx(&prompt, max_new_tokens, &mut stream, cage)?;
    Ok(())
}

/// Detect the console user (the owner of `/dev/console`) so the gatekeeper
/// — which runs as root — can hand the socket directory back to the user-owned
/// MLX daemon after a restart. Falls back to the parent directory's current
/// owner if `/dev/console` cannot be read.
fn console_user() -> Option<(String, u32, u32)> {
    // Primary: `stat -f %Su /dev/console` (macOS-specific).
    if let Ok(out) = Command::new("stat")
        .args(["-f", "%Su", "/dev/console"])
        .output()
    {
        if out.status.success() {
            let user = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !user.is_empty() && user != "root" {
                if let (Ok(uid_out), Ok(gid_out)) = (
                    Command::new("id").args(["-u", &user]).output(),
                    Command::new("id").args(["-g", &user]).output(),
                ) {
                    let uid = String::from_utf8_lossy(&uid_out.stdout)
                        .trim()
                        .parse::<u32>()
                        .ok();
                    let gid = String::from_utf8_lossy(&gid_out.stdout)
                        .trim()
                        .parse::<u32>()
                        .ok();
                    if let (Some(uid), Some(gid)) = (uid, gid) {
                        return Some((user, uid, gid));
                    }
                }
            }
        }
    }
    // Fallback: inherit the existing owner/group of the socket directory's
    // parent if it was already created with the correct ownership by the
    // installer. This keeps restarts from clobbering a working setup.
    None
}

fn ensure_socket_dir() -> Result<()> {
    let path = socket_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();

        // The installer creates /var/run/badapple as console_user:console_group
        // mode 0o770. When the gatekeeper (running as root) recreates the
        // directory after it was removed (e.g. a reboot cleared /var/run), it
        // would otherwise be root:root 0o755, which the user-owned MLX daemon
        // cannot write into. Restore the expected ownership and permissions so
        // the daemon can bind its socket on the next launch.
        let perms = std::fs::Permissions::from_mode(0o770);
        let _ = fs::set_permissions(parent, perms);

        if let Some((_user, uid, gid)) = console_user() {
            // `std::os::unix::fs::chown` sets ownership without needing libc.
            let _ = chown(parent, Some(uid), Some(gid));
        } else {
            // Last-resort fallback: shell out to `chown` using the parent
            // directory's current group so the daemon (a member of that group)
            // retains write access even if we could not resolve the console
            // user. Ownership stays root but the group is preserved.
            if let Ok(meta) = fs::metadata(parent) {
                let parent_gid = meta.gid();
                let _ = chown(parent, None, Some(parent_gid));
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    ensure_socket_dir()?;
    let path = socket_path();
    if path.exists() {
        std::fs::remove_file(&path).ok();
    }

    eprintln!("[gatekeeper] initializing 576-D semantic router");
    let router = SemanticRouter::new()?;
    eprintln!("[gatekeeper] router ready");

    let resolver = FastActionResolver::new()?;

    let cage = AutomationCage::from_env().context("cannot initialize automation cage")?;
    eprintln!("[gatekeeper] automation cage roots: {:?}", cage.roots());

    let replay_cache = Arc::new(ReplayCache::new(REPLAY_CACHE_MAX));

    let listener = UnixListener::bind(&path)
        .with_context(|| format!("cannot bind Bad Apple socket at {path:?}"))?;
    if let Some(parent) = path.parent() {
        let parent_gid = fs::metadata(parent)?.gid();
        chown(&path, None, Some(parent_gid))?;
    }
    // 0o660: owner and group can connect, no world access. The gatekeeper
    // runs as root; the console user is added to the group at install time.
    let perms = std::fs::Permissions::from_mode(0o660);
    std::fs::set_permissions(&path, perms)?;
    eprintln!("[gatekeeper] listening on {path:?}");

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let router = router.clone();
                let resolver = resolver.clone();
                let cage = cage.clone();
                let replay_cache = Arc::clone(&replay_cache);
                std::thread::spawn(move || {
                    if let Err(e) = handle_client(stream, &router, &resolver, &cage, &replay_cache)
                    {
                        if is_benign_disconnect(&e) {
                            // A client (very often a health-check probe that
                            // connects and disconnects without ever sending a
                            // hello frame) went away before or during the
                            // handshake. This is routine, not an error worth
                            // alarming a human over; log it quietly.
                            eprintln!("[gatekeeper] client disconnected before completing handshake: {e:#}");
                        } else {
                            eprintln!("[gatekeeper] client error: {e:#}");
                        }
                    }
                });
            }
            Err(e) => eprintln!("[gatekeeper] accept error: {e}"),
        }
    }

    Ok(())
}
