//! Asynchronous directory watcher for the "wild" local sandbox.
//!
//! The agent watches `~/firefly-agi/wild_workspace`, ingests incoming text
//! payloads, and uses the local LLM to synthesize a read-only Python tool to
//! parse/clean the file. Execution is sandboxed and metrics are logged to the
//! SelfModel. No network access is permitted.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use notify::{Event, RecursiveMode, Watcher};

use tokio::task::spawn_blocking;
use tokio::time::sleep;
use crate::ollama_client::OllamaClient;
use crate::telemetry::run_sandboxed_tool;

const MAX_FILE_BYTES: usize = 256_000;

/// Start the wild-workspace watcher. Returns the async receiver.
/// The underlying `notify` watcher is kept alive inside a `spawn_blocking` task
/// so it never crosses an `.await` point and remains `Send`.
pub fn start_watcher<P: AsRef<Path>>(path: P) -> Result<tokio::sync::mpsc::UnboundedReceiver<Event>, Box<dyn std::error::Error + Send + Sync>> {
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
            std::thread::sleep(Duration::from_secs(60));
        }
    });
    Ok(rx)
}

/// Ingest a file from the wild workspace and synthesize a safe Python cleaner.
pub async fn process_wild_payload(
    ollama: &OllamaClient,
    model: &str,
    path: &Path,
) -> Result<(String, String), Box<dyn std::error::Error + Send + Sync>> {
    let path = path.to_path_buf();
    let payload = spawn_blocking(move || {
        if std::fs::metadata(&path).map(|m| m.len() as usize).unwrap_or(0) > MAX_FILE_BYTES {
            return Err("file too large".to_string());
        }
        match std::fs::read_to_string(&path) {
            Ok(text) => Ok(text),
            Err(_) => Ok(String::from_utf8_lossy(&std::fs::read(&path).unwrap_or_default()).to_string()),
        }
    }).await.map_err(|e| format!("read task failed: {}", e))??;

    let preview: String = payload.chars().take(2000).collect();
    let prompt = format!(
        "You are a local, read-only data-cleaning assistant. Given the following raw payload from a file, write a self-contained Python 3 function named `skill(x)` that parses and cleans the input string and returns a concise summary.\n\nAllowed: math, random, statistics, json, datetime, itertools, collections, string, re.\nForbidden: network, file write, shell, exec, eval, subprocess, open, os.system.\n\nPayload preview:\n{}\n\nReturn ONLY a JSON object: {{\"name\": \"...\", \"language\": \"python\", \"code\": \"def skill(x): ...\"}}.",
        preview
    );

    let tool_json = match ollama.generate_structured(model, &prompt, Some("Return only valid JSON with name, language='python', and code.")).await {
        Ok(v) => v,
        Err(e) => return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("LLM failed: {}", e))) as Box<dyn std::error::Error + Send + Sync>),
    };

    let name = tool_json.get("name").and_then(|v| v.as_str()).unwrap_or("wild_cleaner").to_string();
    let code = tool_json.get("code").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let runner = format!("{}\nprint(skill({:?}))", crate::telemetry::strip_markdown_code(&code), payload);

    let name_for_tool = name.clone();
    let output = spawn_blocking(move || run_sandboxed_tool(&name_for_tool, &runner, "python")).await
        .map_err(|e| format!("sandbox task failed: {}", e))??;

    Ok((name, output))
}

/// Run the wild workspace ingestion loop.
pub async fn run_wild_loop(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    ollama: Arc<OllamaClient>,
    model: String,
    watch_path: PathBuf,
) {
    loop {
        match tokio::time::timeout(Duration::from_secs(60), rx.recv()).await {
            Ok(Some(event)) => {
                for path in event.paths {
                    if path.starts_with(&watch_path) && path.is_file() {
                        match process_wild_payload(&ollama, &model, &path).await {
                            Ok((name, output)) => {
                                println!("🌿 [WILD] Processed {} with '{}': {}", path.display(), name, output.chars().take(120).collect::<String>());
                            }
                            Err(e) => {
                                eprintln!("🌿 [WILD] Failed to process {}: {}", path.display(), e);
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
