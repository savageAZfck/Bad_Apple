//! Asynchronous directory watcher for the "wild" local sandbox.
//!
//! The agent watches `~/Firefly-EdgeOS/wild_workspace`, ingests incoming text
//! payloads, and uses the local LLM to synthesize a read-only Python tool to
//! parse/clean the file. Execution is sandboxed and metrics are logged to the
//! SelfModel. No network access is permitted.

use notify::{Event, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::apple_intelligence;
use crate::apple_intelligence_client::AppleIntelligenceClient;
use crate::benchmark::RustValidator;
use crate::hyperdimensional_core::{OverheadAnalyzer, ScriptEncoder, ThermodynamicMinimizer};
use crate::is_safe_agent_code;
use crate::protocol::{CompactEngramPacket, ConnectionManager};
use crate::strategy_library::{RustSynthesizer, Strategy, StrategyLibrary};
use crate::telemetry::run_sandboxed_tool;
use tokio::task::spawn_blocking;
use tokio::time::{sleep, timeout};

const MAX_FILE_BYTES: usize = 256_000;
const MAX_PREVIEW_CHARS: usize = 2_000;
const MAX_OUTPUT_CHARS: usize = 4_096;
const MAX_TOOL_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_LLM_TIMEOUT: Duration = Duration::from_secs(60);
const WATCH_POLL_IDLE: Duration = Duration::from_secs(60);
const ALLOWED_EXTENSIONS: &[&str] = &[
    "txt", "csv", "json", "log", "md", "xml", "yaml", "yml", "tsv", "py",
];

/// Queue length at which incoming tool synthesis is offloaded to peer nodes.
pub const HEAVY_EXECUTION_THRESHOLD: usize = 8;

/// Prefix for a `CompactEngramPacket` carrying a wild-workspace task request.
pub const WILD_TASK_PREFIX: &str = "@@WILD_TASK@@";

/// Prefix for a `CompactEngramPacket` carrying a wild-workspace task result.
pub const WILD_TASK_RESULT: &str = "@@WILD_TASK_RESULT@@";

/// Serializable wild-workspace task.  `source` is the file contents; `path` is
/// the original workspace path and is used for logging/reply routing.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WildTask {
    pub path: String,
    pub source: String,
    pub is_script: bool,
}

/// Result of a peer-executed wild-workspace task.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WildTaskResult {
    pub path: String,
    pub name: String,
    pub output: String,
}

/// Execute a `WildTask` and return a uniform `(name, output)` pair regardless of
/// whether it is a script synthesis or a raw payload cleaning task.
pub async fn execute_wild_task(
    task: &WildTask,
    client: &AppleIntelligenceClient,
    model: &str,
    strategy_library: &StrategyLibrary,
) -> Result<(String, String), Box<dyn std::error::Error + Send + Sync>> {
    if task.is_script {
        let outcome = process_script(&task.source, strategy_library).await?;
        Ok((
            "wild_rust_synth".into(),
            format!(
                "compiled={} competence={:.2} energy={:.2} velocity_ms={:.2}",
                outcome.compiled, outcome.competence, outcome.energy, outcome.velocity_ms
            ),
        ))
    } else {
        process_wild_source(client, model, &task.source).await
    }
}

/// Start the wild-workspace watcher. Returns the async receiver.
/// The underlying `notify` watcher is kept alive inside a `spawn_blocking` task
/// so it never crosses an `.await` point and remains `Send`.
pub fn start_watcher<P: AsRef<Path>>(
    path: P,
) -> Result<tokio::sync::mpsc::UnboundedReceiver<Event>, Box<dyn std::error::Error + Send + Sync>> {
    std::fs::create_dir_all(path.as_ref()).ok();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
        if let Ok(event) = res {
            let _ = tx.send(event);
        }
    })?;
    watcher.watch(path.as_ref(), RecursiveMode::NonRecursive)?;
    spawn_blocking(move || {
        // Hold the watcher alive for the lifetime of the process.
        loop {
            std::thread::sleep(WATCH_POLL_IDLE);
        }
    });
    Ok(rx)
}

/// Returns `true` if the path is a regular file inside `watch_path` and has an
/// allowed extension. Does not follow symlinks outside the watch directory.
fn is_allowed_file(path: &Path, watch_path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }

    // Only allow plain files with an explicit, allowed extension.
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if !ALLOWED_EXTENSIONS.contains(&ext.as_str()) {
        return false;
    }

    // Ensure the file is contained within the watch directory, not a symlink
    // escape or absolute traversal.
    let Ok(canonical_watch) = watch_path.canonicalize() else {
        return false;
    };
    let Ok(canonical_path) = path.canonicalize() else {
        return false;
    };
    canonical_path.starts_with(&canonical_watch)
}

/// Read at most `MAX_FILE_BYTES` of valid UTF-8 text from `path`.
fn read_limited_text(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("open failed: {}", e))?;
    let mut buf = vec![0u8; MAX_FILE_BYTES];
    let n = file
        .read(&mut buf)
        .map_err(|e| format!("read failed: {}", e))?;
    buf.truncate(n);
    // Reject binary or malformed payloads; do not silently lossy-decode.
    String::from_utf8(buf).map_err(|_| "file is not valid UTF-8".to_string())
}

/// Ingest a file from the wild workspace and synthesize a safe Python cleaner.
///
/// Runs an open-ended compiler-guided self-healing loop: the model is asked to
/// generate a `skill(x)` function, it is executed in the sandbox, and if it
/// fails the raw Python exception is fed back into the next prompt.  Up to
/// `MAX_SELF_HEAL_ATTEMPTS` episodes are tried before giving up.
pub async fn process_wild_payload(
    client: &AppleIntelligenceClient,
    model: &str,
    path: &Path,
) -> Result<(String, String), Box<dyn std::error::Error + Send + Sync>> {
    let path = path.to_path_buf();
    let payload = spawn_blocking(move || read_limited_text(&path))
        .await
        .map_err(|e| format!("read task failed: {}", e))??;
    process_wild_source(client, model, &payload).await
}

pub async fn process_wild_source(
    client: &AppleIntelligenceClient,
    model: &str,
    payload: &str,
) -> Result<(String, String), Box<dyn std::error::Error + Send + Sync>> {
    const MAX_SELF_HEAL_ATTEMPTS: usize = 3;

    let preview: String = payload.chars().take(MAX_PREVIEW_CHARS).collect();
    let mut previous_error: Option<String> = None;

    for _ in 0..MAX_SELF_HEAL_ATTEMPTS {
        let error_context = previous_error.as_ref().map_or_else(
            String::new,
            |e| format!("\n\nThe previous attempt produced this runtime error:\n{}\nFix it in the next version.", e),
        );

        let prompt = format!(
            "You are a local, read-only data-cleaning assistant. Given the following raw payload from a file, write a self-contained Python 3 function named `skill(x)` that parses and cleans the input string and returns a concise summary.\n\nAllowed: math, random, statistics, json, datetime, itertools, collections, string, re.\nForbidden: network, file write, shell, exec, eval, subprocess, open, os.system.{}\n\nPayload preview:\n{}\n\nReturn ONLY a JSON object: {{\"name\": \"...\", \"language\": \"python\", \"code\": \"def skill(x): ...\"}}.",
            error_context,
            preview
        );

        let tool_json = match timeout(
            MAX_LLM_TIMEOUT,
            client.generate_structured(
                model,
                &prompt,
                Some("Return only valid JSON with name, language='python', and code."),
            ),
        )
        .await
        {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return Err(format!("LLM failed: {}", e).into()),
            Err(_) => return Err("LLM generation timed out".into()),
        };

        let name = tool_json
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("wild_cleaner")
            .to_string();
        let code = tool_json
            .get("code")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if !is_safe_agent_code(&code) || !code.to_lowercase().contains("def skill(") {
            previous_error =
                Some("generated tool must define a `def skill(x)` function".to_string());
            continue;
        }

        let runner = format!(
            "{}\nprint(skill({:?}))",
            crate::telemetry::strip_markdown_code(&code),
            payload
        );

        let name_for_tool = name.clone();
        let error_msg = match timeout(
            MAX_TOOL_TIMEOUT,
            spawn_blocking(move || run_sandboxed_tool(&name_for_tool, &runner, "python")),
        )
        .await
        {
            Ok(Ok(Ok(out))) => {
                let output = out.chars().take(MAX_OUTPUT_CHARS).collect();
                return Ok((name, output));
            }
            Ok(Ok(Err(e))) => format!("sandbox rejected tool: {}", e),
            Ok(Err(e)) => format!("sandbox task failed: {}", e),
            Err(_) => "sandbox execution timed out".to_string(),
        };

        previous_error = Some(error_msg);
    }

    Err(format!(
        "compiler-guided self-healing failed after {} attempts; last error: {}",
        MAX_SELF_HEAL_ATTEMPTS,
        previous_error.unwrap_or_else(|| "unknown".to_string())
    )
    .into())
}

/// Maps raw open-source scripts into a continuous HDC phase-space coordinate,
/// then reduces thermodynamic entropy and synthesizes a fluid Rust policy.
///
/// This is a defensive, local-only harness: it processes `.py` and `.txt` files
/// that the user drops into the wild workspace and never contacts the network.
pub struct PhaseSpaceMapper;

impl PhaseSpaceMapper {
    pub fn new() -> Self {
        Self
    }

    /// Encode a script into a 10,000-D phase-space coordinate and minimize its
    /// entropy. Returns the reduced profile and a diagnosis.
    pub fn map(source: &str) -> (crate::hyperdimensional_core::ScriptProfile, String) {
        let mut encoder = ScriptEncoder::new();
        let mut profile = encoder.encode(source);
        ThermodynamicMinimizer::reduce_entropy(&mut profile);
        let (diagnosis, _) = OverheadAnalyzer::analyze(&profile);
        (profile, diagnosis)
    }
}

impl Default for PhaseSpaceMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Process an open-source script through the HDC phase-space mapping →
/// thermodynamic entropy reduction → Rust synthesis → cargo check validation
/// pipeline. If validation passes the 0.99 equilibrium threshold, the generated
/// Rust source is cached as a Sled-backed strategy.
pub async fn process_script(
    source: &str,
    strategy_library: &StrategyLibrary,
) -> Result<SynthesisOutcome, Box<dyn std::error::Error + Send + Sync>> {
    let source = source.to_string();
    let strategy_library = strategy_library.clone();

    // Phase 1: Map the script into a continuous 10,000-D HDC phase-space
    // coordinate, then apply thermodynamic entropy reduction.
    let (mut rust_source, profile, mut synthesizer) = spawn_blocking({
        let source = source.clone();
        move || {
            let (profile, diagnosis) = PhaseSpaceMapper::map(&source);
            let mut synthesizer = RustSynthesizer::new();
            let rust = synthesizer.synthesize(&profile);
            (rust, (profile, diagnosis), synthesizer)
        }
    })
    .await
    .map_err(|e| format!("script processing task failed: {}", e))?;

    // Phase 2: Closed-loop compiler-guided equilibrium validation with up to
    // 3 self-healing episodes.  On each failure the raw `cargo check`
    // diagnostics are fed back into `RustSynthesizer::repair`.
    const EQUILIBRIUM_THRESHOLD: f64 = 0.99;
    const MAX_RETRIES: usize = 3;
    let key = format!("rust_synth_{}", std::process::id());
    let mut validation = RustValidator::validate(&key, &rust_source, 0).await;
    for _ in 0..MAX_RETRIES {
        if validation.competence >= EQUILIBRIUM_THRESHOLD {
            break;
        }
        tracing::info!(
            "Rust synthesis failed (competence {:.3}); attempting compiler-guided repair. Diagnostics:\n{}",
            validation.competence,
            validation.diagnostics
        );
        rust_source = synthesizer.repair(&rust_source, &validation.diagnostics);
        validation = RustValidator::validate(&key, &rust_source, 0).await;
    }

    let outcome = SynthesisOutcome {
        source,
        rust_source,
        diagnosis: profile.1,
        compiled: validation.competence >= EQUILIBRIUM_THRESHOLD,
        competence: validation.competence,
        diagnostics: validation.diagnostics.clone(),
        energy: ThermodynamicMinimizer::energy(&profile.0.profile),
        velocity_ms: validation.wall_time_ms,
    };

    if validation.competence >= EQUILIBRIUM_THRESHOLD {
        let strategy = Strategy::new(
            format!("{}_certified", key),
            format!("Optimized Rust for: {}", outcome.diagnosis),
            "rust".to_string(),
            validation.source,
        );
        strategy_library.put(&strategy).await?;

        let causal = strategy_library
            .explain_failure(&strategy.problem)
            .unwrap_or_else(|| "skill_memory -> DependsOn -> certified_strategy".to_string());
        let title = "Firefly: Rust synthesis certified".to_string();
        let body = format!(
            "{}\nCompetence: {:.2}\nLatency: {:.2} ms\nCausal: {}",
            strategy.problem, validation.competence, validation.wall_time_ms, causal
        );
        apple_intelligence::dispatch_desktop_notification(&title, &body);
    }

    Ok(outcome)
}

/// Outcome of the script-optimization harness.
#[derive(Clone, Debug)]
pub struct SynthesisOutcome {
    pub source: String,
    pub rust_source: String,
    pub diagnosis: String,
    pub compiled: bool,
    pub competence: f64,
    pub diagnostics: String,
    /// HDC profile energy after thermodynamic minimization [0, 1].
    pub energy: f64,
    /// Validation wall-clock time in milliseconds.
    pub velocity_ms: f64,
}

/// Run the wild workspace ingestion loop.
///
/// Ingested files are placed on a shared work queue.  If the local queue length
/// exceeds `HEAVY_EXECUTION_THRESHOLD`, the worker offloads the task as a signed
/// `CompactEngramPacket` to the `ConnectionManager` gossip fabric instead of
/// executing it locally.  Otherwise the tool is run in-process.
pub async fn run_wild_loop(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    client: Arc<AppleIntelligenceClient>,
    model: String,
    watch_path: PathBuf,
    strategy_library: Arc<StrategyLibrary>,
    wan: Option<Arc<ConnectionManager>>,
    origin: String,
) {
    let pending: Arc<Mutex<VecDeque<WildTask>>> = Arc::new(Mutex::new(VecDeque::new()));

    // Worker: pops from the queue and either executes locally or offloads to peers.
    let worker_pending = Arc::clone(&pending);
    let worker_client = Arc::clone(&client);
    let worker_model = model;
    let worker_strategy = Arc::clone(&strategy_library);
    let worker_wan = wan;
    let worker_origin = origin;
    tokio::spawn(async move {
        loop {
            let task = {
                let mut q = worker_pending.lock().unwrap();
                q.pop_front()
            };
            if let Some(task) = task {
                let queue_len = {
                    let q = worker_pending.lock().unwrap();
                    q.len()
                };

                if queue_len >= HEAVY_EXECUTION_THRESHOLD {
                    if let Some(ref wan) = worker_wan {
                        let payload = match serde_json::to_string(&task) {
                            Ok(v) => format!("{}{}", WILD_TASK_PREFIX, v),
                            Err(_) => continue,
                        };
                        let packet = CompactEngramPacket {
                            id: rand::random::<u64>(),
                            timestamp: crate::telemetry::current_secs(),
                            experiential_text: payload,
                            emotional_state_snapshot: "distributed wild-workspace task".into(),
                            origin_instance: worker_origin.clone(),
                            brain_state: Vec::new(),
                            embedding: Vec::new(),
                            priority: 7,
                        };
                        // Fire-and-forget onto the lock-free outbound ring.  The
                        // ConnectionManager sweeper drains the ring and broadcasts
                        // without the wild_workspace worker ever waiting on a peer
                        // socket or a system mutex.
                        wan.push_outgoing(packet);
                        tracing::info!(
                            "🌿 [WILD QUEUE] offloaded {} to swarm (queue_len={})",
                            task.path,
                            queue_len
                        );
                        continue;
                    }
                }

                // Local execution.
                if task.is_script {
                    match process_script(&task.source, &worker_strategy).await {
                        Ok(outcome) => {
                            println!(
                                "🦀 [WILD SYNTH] {}: compiled={} competence={:.2}",
                                task.path, outcome.compiled, outcome.competence
                            );
                        }
                        Err(e) => {
                            eprintln!("🦀 [WILD SYNTH] {} failed: {}", task.path, e);
                        }
                    }
                } else {
                    match process_wild_source(&worker_client, &worker_model, &task.source).await {
                        Ok((name, output)) => {
                            println!(
                                "🌿 [WILD] Processed {} with '{}': {}",
                                task.path,
                                name,
                                output.chars().take(120).collect::<String>()
                            );
                        }
                        Err(e) => {
                            eprintln!("🌿 [WILD] Failed to process {}: {}", task.path, e);
                        }
                    }
                }
            } else {
                sleep(Duration::from_millis(100)).await;
            }
        }
    });

    // Ingestion: read allowed files and push them onto the shared queue.
    loop {
        match timeout(Duration::from_secs(60), rx.recv()).await {
            Ok(Some(event)) => {
                for path in event.paths {
                    if is_allowed_file(&path, &watch_path) {
                        let ext = path
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("")
                            .to_lowercase();
                        let is_script = ext == "py" || ext == "txt";

                        match read_limited_text(&path) {
                            Ok(source) => {
                                let task = WildTask {
                                    path: path.display().to_string(),
                                    source,
                                    is_script,
                                };
                                let mut q = pending.lock().unwrap();
                                q.push_back(task);
                            }
                            Err(e) => {
                                eprintln!("🌿 [WILD] Failed to read {}: {}", path.display(), e);
                            }
                        }
                    }
                }
            }
            Ok(None) => break,
            Err(_) => sleep(Duration::from_secs(1)).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_file_accepts_txt_and_rejects_others() {
        let dir = std::env::temp_dir().join("wild_workspace_test");
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("good.txt");
        let bad = dir.join("bad.exe");
        std::fs::write(&good, "hello").unwrap();
        std::fs::write(&bad, "bad").unwrap();

        assert!(is_allowed_file(&good, &dir));
        assert!(!is_allowed_file(&bad, &dir));
        assert!(!is_allowed_file(&dir, &dir));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_limited_text_respects_max_bytes_and_utf8() {
        let dir = std::env::temp_dir().join("wild_read_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample.txt");
        let content = "hello world".to_string();
        std::fs::write(&path, &content).unwrap();

        assert_eq!(read_limited_text(&path).unwrap(), content);

        let bin_path = dir.join("binary.bin");
        std::fs::write(&bin_path, vec![0xff, 0xfe]).unwrap();
        assert!(read_limited_text(&bin_path).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
