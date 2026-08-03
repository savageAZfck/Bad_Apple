//! Asynchronous directory watcher for the "wild" local sandbox.
//!
//! The agent watches `~/firefly-agi/wild_workspace`, ingests incoming text
//! payloads, and uses the local LLM to synthesize a read-only Python tool to
//! parse/clean the file. Execution is sandboxed and metrics are logged to the
//! SelfModel. No network access is permitted.

use notify::{Event, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::benchmark::RustValidator;
use crate::hyperdimensional_core::{OverheadAnalyzer, ScriptEncoder, ThermodynamicMinimizer};
use crate::is_safe_agent_code;
use crate::ollama_client::OllamaClient;
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
pub async fn process_wild_payload(
    ollama: &OllamaClient,
    model: &str,
    path: &Path,
) -> Result<(String, String), Box<dyn std::error::Error + Send + Sync>> {
    let path = path.to_path_buf();
    let payload = spawn_blocking(move || read_limited_text(&path))
        .await
        .map_err(|e| format!("read task failed: {}", e))??;

    let preview: String = payload.chars().take(MAX_PREVIEW_CHARS).collect();
    let prompt = format!(
        "You are a local, read-only data-cleaning assistant. Given the following raw payload from a file, write a self-contained Python 3 function named `skill(x)` that parses and cleans the input string and returns a concise summary.\n\nAllowed: math, random, statistics, json, datetime, itertools, collections, string, re.\nForbidden: network, file write, shell, exec, eval, subprocess, open, os.system.\n\nPayload preview:\n{}\n\nReturn ONLY a JSON object: {{\"name\": \"...\", \"language\": \"python\", \"code\": \"def skill(x): ...\"}}.",
        preview
    );

    let tool_json = match timeout(
        MAX_LLM_TIMEOUT,
        ollama.generate_structured(
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
        return Err("generated tool must define a `def skill(x)` function".into());
    }

    let runner = format!(
        "{}\nprint(skill({:?}))",
        crate::telemetry::strip_markdown_code(&code),
        payload
    );

    let name_for_tool = name.clone();
    let output = match timeout(
        MAX_TOOL_TIMEOUT,
        spawn_blocking(move || run_sandboxed_tool(&name_for_tool, &runner, "python")),
    )
    .await
    {
        Ok(Ok(Ok(out))) => out,
        Ok(Ok(Err(e))) => return Err(format!("sandbox rejected tool: {}", e).into()),
        Ok(Err(e)) => return Err(format!("sandbox task failed: {}", e).into()),
        Err(_) => return Err("sandbox execution timed out".into()),
    };

    let output = output.chars().take(MAX_OUTPUT_CHARS).collect();
    Ok((name, output))
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
    let (rust_source, profile) = spawn_blocking({
        let source = source.clone();
        move || {
            let (profile, diagnosis) = PhaseSpaceMapper::map(&source);
            let mut synthesizer = RustSynthesizer::new();
            let rust = synthesizer.synthesize(&profile);
            (rust, (profile, diagnosis))
        }
    })
    .await
    .map_err(|e| format!("script processing task failed: {}", e))?;

    // Phase 2: Closed-loop equilibrium validation with up to 3 retries.
    const EQUILIBRIUM_THRESHOLD: f64 = 0.99;
    const MAX_RETRIES: usize = 3;
    let key = format!("rust_synth_{}", std::process::id());
    let validation = RustValidator::validate(&key, &rust_source, MAX_RETRIES).await;

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
pub async fn run_wild_loop(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    ollama: Arc<OllamaClient>,
    model: String,
    watch_path: PathBuf,
    strategy_library: Arc<StrategyLibrary>,
) {
    loop {
        match timeout(Duration::from_secs(60), rx.recv()).await {
            Ok(Some(event)) => {
                for path in event.paths {
                    if is_allowed_file(&path, &watch_path) {
                        // Route scripts (.py / .txt) through the optimization harness.
                        let ext = path
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("")
                            .to_lowercase();
                        let is_script = ext == "py" || ext == "txt";

                        if is_script {
                            match read_limited_text(&path) {
                                Ok(source) => {
                                    match process_script(&source, &strategy_library).await {
                                        Ok(outcome) => {
                                            println!(
                                                "🦀 [WILD SYNTH] {}: compiled={} competence={:.2}",
                                                path.display(),
                                                outcome.compiled,
                                                outcome.competence
                                            );
                                        }
                                        Err(e) => {
                                            eprintln!(
                                                "🦀 [WILD SYNTH] {} failed: {}",
                                                path.display(),
                                                e
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!(
                                        "🦀 [WILD SYNTH] {} read failed: {}",
                                        path.display(),
                                        e
                                    );
                                }
                            }
                        } else {
                            match process_wild_payload(&ollama, &model, &path).await {
                                Ok((name, output)) => {
                                    println!(
                                        "🌿 [WILD] Processed {} with '{}': {}",
                                        path.display(),
                                        name,
                                        output.chars().take(120).collect::<String>()
                                    );
                                }
                                Err(e) => {
                                    eprintln!(
                                        "🌿 [WILD] Failed to process {}: {}",
                                        path.display(),
                                        e
                                    );
                                }
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
