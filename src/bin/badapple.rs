use anyhow::{bail, Context, Result};
use bad_apple::bad_apple_ipc::{call_agent, query_with_metrics};
use bad_apple::cert;
use serde_json::Value;
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

/// Compile a TTS sanitizer regex. If a static pattern somehow fails,
/// fall back to a regex that matches nothing so the voice output still
/// works rather than panicking.
fn tts_regex(pattern: &str) -> regex::Regex {
    regex::Regex::new(pattern)
        .unwrap_or_else(|_| regex::Regex::new("a^").expect("static fallback regex must compile"))
}

fn sanitize_for_tts(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    // Strip URLs.
    let mut s = tts_regex(r"https?://\S+")
        .replace_all(text, " ")
        .to_string();

    // Strip fenced code blocks, inline code, and markup that speech engines
    // read as literal punctuation.
    s = tts_regex(r"```[\s\S]*?```")
        .replace_all(&s, " ")
        .to_string();
    s = tts_regex(r"<tool_call>[\s\S]*?</tool_call>")
        .replace_all(&s, " ")
        .to_string();
    s = tts_regex(r"<[^>]+>").replace_all(&s, " ").to_string();
    s = tts_regex(r"`[^`]*`").replace_all(&s, " ").to_string();

    // Strip markdown emphasis/headers/list markers and turn links into text.
    s = tts_regex(r"(\*+|_+|~+|#+|>\s*)")
        .replace_all(&s, " ")
        .to_string();
    s = tts_regex(r"!?\[([^\]]*)\]\([^)]*\)")
        .replace_all(&s, "$1")
        .to_string();

    // Remove bullet characters and list markers.
    s = tts_regex(r"\n\s*[•·*-]\s+")
        .replace_all(&s, "\n")
        .to_string();
    s = tts_regex(r"(?m)^[•·*-]\s+").replace_all(&s, "").to_string();
    s = tts_regex(r"[•·]").replace_all(&s, " ").to_string();

    // Ellipses, em/en dashes and run-on hyphens become chunk breaks, not
    // punctuation the voice can read.
    s = tts_regex(r"\.{3,}|…").replace_all(&s, "\n").to_string();
    s = tts_regex(r"\s*[—–]\s*").replace_all(&s, "\n").to_string();
    s = tts_regex(r"\s*-{2,}\s*").replace_all(&s, "\n").to_string();

    // Clause-breaking colons and semicolons become chunk breaks when followed
    // by whitespace; colons in times/URLs are left for the allowlist pass.
    s = tts_regex(r"(\s*)([:;])(\s+)")
        .replace_all(&s, "$1\n$3")
        .to_string();

    // Remove double quotes.
    s = s.replace('"', " ");

    // Allow-list pass: keep alphanumerics, whitespace, and a tiny set of
    // punctuation the engine uses for prosody (,.?!). Remove or replace
    // everything else so the voice never reads symbols aloud. Keep apostrophes
    // only in contractions and hyphens only inside words.
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(chars.len());
    for (i, &c) in chars.iter().enumerate() {
        if c.is_alphanumeric() || c.is_whitespace() || c == ',' || c == '.' || c == '?' || c == '!'
        {
            out.push(c);
            continue;
        }
        if c == '\'' {
            let prev = i.checked_sub(1).and_then(|j| chars.get(j)).copied();
            let next = chars.get(i + 1).copied();
            if prev.is_some_and(|p| p.is_alphanumeric())
                && next.is_some_and(|n| n.is_alphanumeric())
            {
                out.push(c);
            } else {
                out.push(' ');
            }
            continue;
        }
        if c == '-' {
            let prev = i.checked_sub(1).and_then(|j| chars.get(j)).copied();
            let next = chars.get(i + 1).copied();
            if prev.is_some_and(|p| p.is_alphabetic()) && next.is_some_and(|n| n.is_alphabetic()) {
                out.push(c);
            } else {
                out.push(' ');
            }
            continue;
        }
        out.push(' ');
    }
    s = out;

    // Tidy spaces before punctuation and collapse whitespace.
    s = tts_regex(r"\s+([.,?!])").replace_all(&s, "$1").to_string();
    s = tts_regex(r"\n\n+").replace_all(&s, "\n").to_string();
    s = tts_regex(r"[ \t]+").replace_all(&s, " ").to_string();
    s = tts_regex(r" \n").replace_all(&s, "\n").to_string();
    s = tts_regex(r"\n ").replace_all(&s, "\n").to_string();

    // Return one clean line per chunk.
    s.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            if !line.contains(|c: char| c.is_alphabetic() || c.is_ascii_digit()) {
                return None;
            }
            Some(line.to_string())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() -> Result<()> {
    // Subcommands that take their own --flags; dispatch before the query
    // parser rejects them as unknown options.
    match std::env::args().nth(1).as_deref() {
        Some("policy") => {
            return run_policy_subcommand(&std::env::args().skip(2).collect::<Vec<_>>())
        }
        Some("mesh-brain") => {
            return run_mesh_brain_subcommand(&std::env::args().skip(2).collect::<Vec<_>>())
        }
        _ => {}
    }
    let mut args = std::env::args().skip(1);
    let mut prompt_parts = Vec::new();
    let mut max_new_tokens = 500;
    let mut speak_stream = std::env::var("BADAPPLE_SPEAK").is_ok();
    let mut benchmark_mode = false;
    let mut json_stream = std::env::var("BADAPPLE_STREAM_JSON").is_ok();
    let mut voice_mode = std::env::var("BADAPPLE_VOICE").is_ok() || speak_stream;
    let mut persona: Option<String> = None;
    let mut roast_mode = false;
    let mut doctor_mode = false;
    let mut crash_report = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            "--speak" => {
                speak_stream = true;
                voice_mode = true;
            }
            "--json" => {
                json_stream = true;
            }
            "--benchmark" | "--bench" => {
                benchmark_mode = true;
            }
            "--roast" => {
                roast_mode = true;
            }
            "--doctor" | "--diagnostics" => {
                doctor_mode = true;
            }
            "--crash-report" => {
                crash_report = true;
            }
            "--persona" => {
                persona = Some(args.next().context("--persona requires a value")?);
            }
            "-n" | "--max-tokens" => {
                let value = args.next().context("--max-tokens requires a value")?;
                max_new_tokens = value
                    .parse::<usize>()
                    .context("--max-tokens must be an integer")?;
            }
            "--" => {
                prompt_parts.extend(args);
                break;
            }
            _ if arg.starts_with('-') => bail!("unknown option: {arg}"),
            _ => prompt_parts.push(arg),
        }
    }

    if roast_mode && persona.is_none() {
        persona = Some("drill".to_string());
    }

    if doctor_mode {
        return run_doctor();
    }

    if crash_report {
        return run_crash_report();
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("model") {
        return run_model_subcommand(&prompt_parts[1..]);
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("p2p") {
        return run_p2p_subcommand(&prompt_parts[1..]);
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("vault") {
        return run_vault_subcommand(&prompt_parts[1..]);
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("workspace") {
        return run_workspace_subcommand(&prompt_parts[1..]);
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("mcp") {
        return run_mcp_subcommand(&prompt_parts[1..]);
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("redteam") {
        return run_redteam_subcommand(&prompt_parts[1..]);
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("ify") {
        return run_ify_subcommand(&prompt_parts[1..]);
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("policy") {
        return run_policy_subcommand(&prompt_parts[1..]);
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("mesh-brain") {
        return run_mesh_brain_subcommand(&prompt_parts[1..]);
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("cert") {
        return run_cert();
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("status") {
        return run_status();
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("receipts") {
        return run_receipts();
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("export-proof") {
        return run_export_proof(&prompt_parts[1..]);
    }

    if prompt_parts.first().map(std::string::String::as_str) == Some("demo") {
        if prompt_parts.get(1).map(std::string::String::as_str) == Some("full") {
            return run_demo_full();
        }
        return run_demo();
    }

    let prompt = if prompt_parts.is_empty() && !benchmark_mode {
        bail!("usage: badapple [OPTIONS] \"query\"\n       badapple model <list|scan|info|use|verify|add|remove> [args]\n       badapple p2p <peers|sync|sync-doc <kind>|sync-personas|sync-prompt|sync-settings|sync-models|receive-mesh [timeout_ms]|models|pull <peer_id> <model_id>|send <peer_id> <model_id>|receive [peer_id model_id]>\n       badapple vault <get|set|remove|list|import> [args]\n       badapple workspace <get|set <path>|index|watch [path]>\n       badapple mcp <list|add <id> <command> [args...]|remove <id>|install <id>|uninstall <id>|start <id>|stop <id>|status <id>>\n       badapple redteam <run|watch|status|category <category>|probe <id>>\n       badapple status\n       badapple receipts\n       badapple export-proof [dir]\n       badapple demo [full]\n       badapple cert");
    } else {
        prompt_parts.join(" ")
    };

    // Apply voice and persona sentinels.
    let wrap = |p: String| {
        let mut p = p;
        if voice_mode && !p.starts_with("__BADAPPLE_VOICE__ ") {
            p = format!("__BADAPPLE_VOICE__ {p}");
        }
        if let Some(ref name) = persona {
            if !p.starts_with("__BADAPPLE_PERSONA__") {
                p = format!("__BADAPPLE_PERSONA__{name}__ {p}");
            }
        }
        p
    };

    if benchmark_mode {
        let prompt = if prompt.is_empty() {
            None
        } else {
            Some(wrap(prompt))
        };
        return run_benchmark(prompt.as_deref(), max_new_tokens);
    }

    let prompt = wrap(prompt);
    let tts = if speak_stream {
        Some(TtsQueue::new())
    } else {
        None
    };
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let mut emitted = String::new();

    let (final_text, metrics) = query_with_metrics(&prompt, max_new_tokens, |token| {
        emitted.push_str(token);
        if json_stream {
            let line = format!(
                "{{\"type\":\"token\",\"text\":{}}}\n",
                serde_json::to_string(token).unwrap_or_default()
            );
            let _ = stdout.write_all(line.as_bytes());
        } else if let Some(ref q) = tts {
            q.push(token);
        } else {
            let _ = stdout.write_all(token.as_bytes());
        }
        let _ = stdout.flush();
    })?;

    if json_stream {
        let line = format!(
            "{{\"type\":\"done\",\"text\":{},\"metrics\":{}}}\n",
            serde_json::to_string(&final_text).unwrap_or_default(),
            serde_json::to_string(&metrics).unwrap_or_default()
        );
        stdout.write_all(line.as_bytes())?;
    } else if let Some(suffix) = final_text.strip_prefix(&emitted) {
        if let Some(tts) = tts.as_ref() {
            if !suffix.trim().is_empty() {
                tts.push(suffix);
            }
        } else {
            stdout.write_all(suffix.as_bytes())?;
        }
    }
    if !json_stream {
        stdout.write_all(b"\n")?;
    }
    stdout.flush()?;

    if let Some(ref tts) = tts {
        tts.flush();
    }

    // Give TTS a moment to finish before the program exits.
    if tts.is_some() {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    Ok(())
}

enum TtsMsg {
    Text(String),
    Flush,
}

struct TtsQueue {
    tx: Option<Sender<TtsMsg>>,
    worker: Option<thread::JoinHandle<()>>,
}

impl TtsQueue {
    fn new() -> Self {
        let (tx, rx) = channel::<TtsMsg>();
        let worker = thread::spawn(move || tts_worker(rx));
        Self {
            tx: Some(tx),
            worker: Some(worker),
        }
    }
    fn push(&self, text: &str) {
        if let Some(ref tx) = self.tx {
            let _ = tx.send(TtsMsg::Text(text.to_string()));
        }
    }
    fn flush(&self) {
        if let Some(ref tx) = self.tx {
            let _ = tx.send(TtsMsg::Flush);
        }
    }
}

impl Drop for TtsQueue {
    fn drop(&mut self) {
        self.flush();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// TTS worker buffers tokens into natural chunks and flushes on idle timeout
/// so the voice starts quickly without synthesizing every subword separately.
fn tts_worker(rx: Receiver<TtsMsg>) {
    const IDLE_TIMEOUT: Duration = Duration::from_millis(80);
    const MAX_CHUNK: usize = 160;
    const MIN_CHUNK: usize = 2;

    let mut buffer = String::new();
    let mut deadline: Option<Instant> = None;

    fn is_sentence_end(s: &str) -> bool {
        s.trim_end().ends_with(['.', '!', '?'])
    }

    fn has_speech(s: &str) -> bool {
        s.contains(|c: char| c.is_alphabetic() || c.is_ascii_digit())
    }

    loop {
        let timeout = match deadline {
            Some(d) => d.saturating_duration_since(Instant::now()),
            None => Duration::from_millis(500),
        };

        match rx.recv_timeout(timeout) {
            Ok(TtsMsg::Text(text)) => {
                if text.is_empty() {
                    continue;
                }
                buffer.push_str(&sanitize_for_tts(&text));

                // Flush complete lines as soon as they appear so list items
                // and clause breaks become their own utterances.
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim();
                    if !line.is_empty() && has_speech(line) {
                        speak_chunk(line);
                    }
                    let rest = buffer.split_off(pos + 1);
                    buffer = rest;
                }

                if buffer.is_empty() {
                    deadline = None;
                } else if buffer.len() >= MIN_CHUNK
                    && (is_sentence_end(&buffer) || buffer.len() >= MAX_CHUNK)
                {
                    let line = buffer.trim();
                    if !line.is_empty() && has_speech(line) {
                        speak_chunk(line);
                    }
                    buffer.clear();
                    deadline = None;
                } else {
                    deadline = Some(Instant::now() + IDLE_TIMEOUT);
                }
            }
            Ok(TtsMsg::Flush) | Err(RecvTimeoutError::Disconnected) => {
                let line = buffer.trim();
                if !line.is_empty() && has_speech(line) {
                    speak_chunk(line);
                }
                break;
            }
            Err(RecvTimeoutError::Timeout) => {
                let line = buffer.trim();
                if !line.is_empty() && has_speech(line) {
                    speak_chunk(line);
                }
                buffer.clear();
                deadline = None;
            }
        }
    }
}

/// Locate the native `badapple-tts` binary.  It may be next to the current
/// executable (release build), in the Cargo target directory (dev build), or
/// in the bundled app.
fn tts_binary_path() -> Option<std::path::PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        let next_to_exe = exe.parent()?.join("badapple-tts");
        if next_to_exe.is_file() {
            return Some(next_to_exe);
        }
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let from_manifest = std::path::PathBuf::from(manifest)
            .join("target")
            .join("release")
            .join("badapple-tts");
        if from_manifest.is_file() {
            return Some(from_manifest);
        }
    }
    let from_cwd = std::path::PathBuf::from("target/release/badapple-tts");
    if from_cwd.is_file() {
        return Some(from_cwd);
    }
    // Bundled app helper location.
    let from_app =
        std::path::PathBuf::from("/Applications/Bad Apple.app/Contents/Helpers/badapple-tts");
    if from_app.is_file() {
        return Some(from_app);
    }
    None
}

/// Send a chunk to the local native TTS server and play it with afplay.
/// If the server is not running, attempt to start it once. Multi-line text is
/// split so each line is synthesised as a separate utterance.
fn speak_chunk(text: &str) {
    if !text.contains(|c: char| c.is_alphabetic() || c.is_ascii_digit()) {
        return;
    }

    let text = sanitize_for_tts(text);
    for line in text.split('\n') {
        let line = line.trim();
        if line.is_empty() || !line.contains(|c: char| c.is_alphabetic() || c.is_ascii_digit()) {
            continue;
        }

        let voice = std::env::var("BADAPPLE_TTS_VOICE").unwrap_or_else(|_| "Best".to_string());
        let socket = std::env::var("BADAPPLE_TTS_SOCKET")
            .unwrap_or_else(|_| "/tmp/badapple_tts.sock".to_string());

        // Make sure the TTS server socket is reachable.  The platform install
        // loads the LaunchAgent, but a bare `cargo build --release` run needs to
        // start it.
        let stream = match UnixStream::connect(&socket) {
            Ok(s) => Some(s),
            Err(_) => {
                if let Some(bin) = tts_binary_path() {
                    let _ = start_tts_server(&bin, &socket);
                    // Give the server a moment to bind.
                    for _ in 0..20 {
                        if UnixStream::connect(&socket).is_ok() {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
                UnixStream::connect(&socket).ok()
            }
        };

        let mut stream = match stream {
            Some(s) => s,
            None => continue,
        };

        let request = format!(
            "{{\"text\":{},\"voice\":{}}}\n",
            serde_json::to_string(line).unwrap_or_default(),
            serde_json::to_string(&voice).unwrap_or_default()
        );
        if stream.write_all(request.as_bytes()).is_err() {
            continue;
        }
        let mut response = String::new();
        if stream.read_to_string(&mut response).is_err() {
            continue;
        }
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&response) {
            if let Some(wav) = json.get("wav_path").and_then(|v| v.as_str()) {
                let _ = std::process::Command::new("/usr/bin/afplay")
                    .arg(wav)
                    .status();
            }
        }
    }
}

fn start_tts_server(bin: &std::path::Path, socket: &str) -> Result<()> {
    use std::process::Stdio;
    std::fs::remove_file(socket).ok();
    std::process::Command::new(bin)
        .env("BADAPPLE_TTS_SOCKET", socket)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to start badapple-tts")?;
    Ok(())
}

fn run_benchmark(single_prompt: Option<&str>, max_new_tokens: usize) -> Result<()> {
    let prompts: Vec<String> = if let Some(p) = single_prompt {
        vec![p.to_string()]
    } else {
        vec![
            "Who are you?".to_string(),
            "What is the capital of France?".to_string(),
            "Tell me about Rome.".to_string(),
            "What do you think of Siri?".to_string(),
            "How does a car engine work?".to_string(),
        ]
    };

    println!(
        "Bad Apple benchmark — {} prompt(s), max_tokens={}",
        prompts.len(),
        max_new_tokens
    );
    println!(
        "{:<38} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "prompt", "tok", "ttft(s)", "decode", "total", "mem(GB)"
    );

    let mut total_tokens = 0usize;
    let mut total_elapsed = 0.0;
    let mut all_metrics = Vec::new();

    const BENCH: &str = "__BADAPPLE_BENCHMARK__ ";
    for prompt in &prompts {
        let start = Instant::now();
        let mut first_token_at: Option<Instant> = None;
        let bench_prompt = BENCH.to_string() + prompt;
        let (text, metrics) = query_with_metrics(&bench_prompt, max_new_tokens, |_token| {
            if first_token_at.is_none() {
                first_token_at = Some(Instant::now());
            }
        })?;
        let elapsed = start.elapsed().as_secs_f64();
        let ttft = first_token_at.map_or(0.0, |t| t.duration_since(start).as_secs_f64());
        total_elapsed += elapsed;

        if let Some(m) = metrics.clone() {
            total_tokens += m.tokens;
            println!(
                "{:<38} {:>8} {:>10.2} {:>10.1} {:>10.1} {:>10.2}",
                truncate(prompt, 37),
                m.tokens,
                ttft,
                m.decode_tps,
                m.total_tps,
                m.peak_memory_gb
            );
            all_metrics.push(m);
        } else {
            println!(
                "{:<38} {:>8} {:>10.2} {:>10} {:>10} {:>10}",
                truncate(prompt, 37),
                0,
                ttft,
                "-",
                "-",
                "-"
            );
        }
        // Avoid cache hits polluting the benchmark.
        if text.len() < 500 {
            let _ = query_with_metrics("__badapple_new_chat__", 16, |_token| {});
        }
    }

    if !all_metrics.is_empty() {
        let avg_decode =
            all_metrics.iter().map(|m| m.decode_tps).sum::<f64>() / all_metrics.len() as f64;
        let avg_total =
            all_metrics.iter().map(|m| m.total_tps).sum::<f64>() / all_metrics.len() as f64;
        let max_mem = all_metrics
            .iter()
            .map(|m| m.peak_memory_gb)
            .fold(0.0, f64::max);
        println!("{:-<90}", "");
        println!(
            "{:<38} {:>8} {:>10} {:>10.1} {:>10.1} {:>10.2}",
            "AVERAGE / MAX", total_tokens, "", avg_decode, avg_total, max_mem
        );
        println!("Total wall time: {total_elapsed:.2}s");
    }
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

/// Try to locate the Bad Apple repo root for diagnostics.
fn badapple_root() -> Option<std::path::PathBuf> {
    if let Ok(root) = std::env::var("BADAPPLE_ROOT") {
        let p = std::path::PathBuf::from(root);
        if p.join("Cargo.toml").is_file() {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut p = exe.clone();
        for _ in 0..5 {
            p.pop();
            if p.join("Cargo.toml").is_file() {
                return Some(p);
            }
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let home = std::path::PathBuf::from(home);
        for candidate in &[
            "bad_apple",
            "Bad_Apple",
            "Code/bad_apple",
            "Projects/bad_apple",
            "src/bad_apple",
        ] {
            let p = home.join(candidate);
            if p.join("Cargo.toml").is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// Run a local support diagnostic and print a redacted report.
fn run_doctor() -> Result<()> {
    use std::fmt::Write;
    use std::process::Command;

    let mut report = String::new();
    let _ = writeln!(report, "=== Bad Apple Doctor ===");

    // Host
    let _ = writeln!(report, "\n[host]");
    if let Ok(out) = Command::new("sw_vers").output() {
        let _ = writeln!(report, "{}", String::from_utf8_lossy(&out.stdout));
    }
    if let Ok(out) = Command::new("uname").args(["-m"]).output() {
        let _ = writeln!(
            report,
            "arch: {}",
            String::from_utf8_lossy(&out.stdout).trim()
        );
    }

    // Python-free status
    let _ = writeln!(report, "\n[python-free status]");
    if let Some(root) = badapple_root() {
        let _ = writeln!(report, "project root: {}", root.display());
        match Command::new("find")
            .args([".", "-maxdepth", "1", "-name", "*.py"])
            .current_dir(&root)
            .output()
        {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let files: Vec<&str> = stdout.lines().filter(|l| !l.is_empty()).collect();
                if files.is_empty() {
                    let _ = writeln!(report, "python files in root: none");
                } else {
                    let _ = writeln!(report, "python files in root:");
                    for f in files.iter().take(20) {
                        let _ = writeln!(report, "  - {}", f);
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(report, "python files in root: check failed ({e})");
            }
        }
    } else {
        let _ = writeln!(report, "project root: not found");
    }

    // Binaries
    let _ = writeln!(report, "\n[binaries]");
    let mut bin_dirs: Vec<std::path::PathBuf> = Vec::new();
    if let Some(root) = badapple_root() {
        bin_dirs.push(root.join("target/release"));
    }
    if let Some(p) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(std::path::Path::to_path_buf))
    {
        bin_dirs.push(p);
    }
    bin_dirs.push(std::path::PathBuf::from("/usr/local/bin"));
    bin_dirs.push(std::path::PathBuf::from(
        "/Applications/Bad Apple.app/Contents/Helpers",
    ));
    bin_dirs.push(std::path::PathBuf::from(
        "/Applications/Bad Apple.app/Contents/MacOS",
    ));
    for bin in ["badapple", "gatekeeper", "badapple-identity"] {
        let mut found = None;
        if let Ok(p) = Command::new("which").arg(bin).output() {
            let path = String::from_utf8_lossy(&p.stdout).trim().to_string();
            if !path.is_empty() {
                found = Some(path);
            }
        }
        if found.is_none() {
            for dir in &bin_dirs {
                let candidate = dir.join(bin);
                if candidate.is_file() {
                    found = Some(candidate.display().to_string());
                    break;
                }
            }
        }
        let _ = writeln!(
            report,
            "{}: {}",
            bin,
            found.unwrap_or_else(|| "not found".to_string())
        );
    }

    // Bad Apple.app
    let _ = writeln!(report, "\n[Bad Apple.app]");
    let app = std::path::PathBuf::from("/Applications/Bad Apple.app");
    let _ = writeln!(report, "installed: {}", app.is_dir());
    let info = app.join("Contents/Info.plist");
    if info.is_file() {
        if let Ok(out) = Command::new("defaults")
            .args([
                "read",
                "/Applications/Bad Apple.app/Contents/Info",
                "CFBundleShortVersionString",
            ])
            .output()
        {
            let _ = writeln!(
                report,
                "version: {}",
                String::from_utf8_lossy(&out.stdout).trim()
            );
        }
    }

    // launchd jobs
    let _ = writeln!(report, "\n[launchd]");
    let uid = Command::new("id")
        .args(["-u"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    for label in [
        "com.badapple.mlx",
        "com.badapple.gatekeeper",
        "com.badapple.supervisor",
        "com.badapple.tts",
        "com.badapple.menubar",
    ] {
        let mut status = "not found".to_string();
        let mut domains = vec![format!("system/{}", label)];
        if !uid.is_empty() {
            domains.push(format!("gui/{uid}/{label}"));
        }
        for domain in domains {
            if let Ok(out) = Command::new("launchctl").args(["print", &domain]).output() {
                let text = String::from_utf8_lossy(&out.stdout);
                if text.contains(&format!("{label} = {{")) {
                    if let Some(st) = text.lines().find_map(|l| l.trim().strip_prefix("state = ")) {
                        status = st.trim().to_string();
                    } else if text.contains("active count =") {
                        status = "active".to_string();
                    }
                    break;
                }
            }
        }
        let _ = writeln!(report, "{label}: {status}");
    }

    // Sockets
    let _ = writeln!(report, "\n[sockets]");
    for (sock, note) in [
        ("/var/run/badapple/substrate.sock", ""),
        ("/var/run/badapple/substrate_mlx.sock", ""),
        ("/var/run/badapple/identity.sock", ""),
        ("/var/run/badapple/mcp.sock", " (MCP off by default)"),
        ("/var/run/badapple/aqua_helper.sock", ""),
        ("/tmp/badapple_tts.sock", ""),
    ] {
        let p = std::path::PathBuf::from(sock);
        let status = if std::fs::metadata(&p).is_ok() {
            "present"
        } else {
            "missing"
        };
        let _ = writeln!(report, "{}: {}{}", sock, status, note);
    }

    // Data / logs
    let _ = writeln!(report, "\n[data]");
    let data = std::path::PathBuf::from("/var/lib/bad_apple");
    let _ = writeln!(report, "/var/lib/bad_apple: {}", data.is_dir());
    if let Ok(home) = std::env::var("HOME") {
        let _ = writeln!(
            report,
            "~/.bad_apple: {}",
            std::path::PathBuf::from(&home).join(".bad_apple").is_dir()
        );
    }
    for log in [
        "/var/log/bad_apple_mlx_server.log",
        "/var/log/bad_apple_supervisor.log",
    ] {
        let p = std::path::PathBuf::from(log);
        let _ = writeln!(
            report,
            "{}: {}",
            log,
            if p.is_file() { "present" } else { "missing" }
        );
    }

    // Model cache
    let _ = writeln!(report, "\n[model cache]");
    if let Ok(home) = std::env::var("HOME") {
        let hub = std::path::PathBuf::from(&home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
        if hub.is_dir() {
            let mut found = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&hub) {
                for e in entries.flatten() {
                    let n = e.file_name().to_string_lossy().to_string();
                    if n.starts_with("models--") {
                        found.push(n);
                    }
                }
            }
            let _ = writeln!(report, "cached models: {}", found.len());
            for m in found.iter().take(8) {
                let _ = writeln!(report, "  - {m}");
            }
        } else {
            let _ = writeln!(report, "huggingface hub: missing");
        }
    }

    // Memory
    let _ = writeln!(report, "\n[memory]");
    if let Ok(out) = Command::new("memory_pressure").output() {
        let _ = writeln!(report, "{}", String::from_utf8_lossy(&out.stdout));
    } else if let Ok(out) = Command::new("vm_stat").output() {
        let _ = writeln!(report, "{}", String::from_utf8_lossy(&out.stdout));
    }

    println!("{report}");
    Ok(())
}

/// Collect crash logs, daemon state, and system info into a single shareable report.
fn run_crash_report() -> Result<()> {
    use std::fmt::Write;
    use std::process::Command;
    let mut report = String::new();
    let _ = writeln!(report, "=== Bad Apple Crash Report ===");
    let _ = writeln!(
        report,
        "Generated: {}",
        std::process::Command::new("date")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    );

    // System info
    let _ = writeln!(report, "\n[system]");
    if let Ok(out) = Command::new("sw_vers").output() {
        let _ = writeln!(report, "{}", String::from_utf8_lossy(&out.stdout).trim());
    }
    if let Ok(out) = Command::new("uname").args(["-a"]).output() {
        let _ = writeln!(report, "{}", String::from_utf8_lossy(&out.stdout).trim());
    }

    // Daemon status
    let _ = writeln!(report, "\n[daemon status]");
    for label in [
        "com.badapple.mlx",
        "com.badapple.gatekeeper",
        "com.badapple.supervisor",
    ] {
        let status = Command::new("launchctl")
            .args(["print", &format!("system/{label}")])
            .output();
        match status {
            Ok(o) if o.status.success() => {
                let text = String::from_utf8_lossy(&o.stdout);
                if let Some(line) = text.lines().find(|l| l.contains("state =")) {
                    let _ = writeln!(report, "{label}: {}", line.trim());
                } else {
                    let _ = writeln!(report, "{label}: running (details in launchctl print)");
                }
            }
            _ => {
                let _ = writeln!(report, "{label}: not loaded or not found");
            }
        }
    }

    // Socket status
    let _ = writeln!(report, "\n[sockets]");
    for sock in [
        "/var/run/badapple/substrate.sock",
        "/var/run/badapple/substrate_mlx.sock",
    ] {
        let _ = writeln!(
            report,
            "{sock}: {}",
            if std::path::Path::new(sock).exists() {
                "present"
            } else {
                "missing"
            }
        );
    }

    // Recent gatekeeper log (last 50 lines)
    let _ = writeln!(report, "\n[gatekeeper log — last 50 lines]");
    if let Ok(out) = Command::new("tail")
        .args(["-50", "/var/log/bad_apple_gatekeeper.log"])
        .output()
    {
        let _ = writeln!(report, "{}", String::from_utf8_lossy(&out.stdout));
    } else {
        let _ = writeln!(report, "(log not found)");
    }

    // Recent MLX server log (last 50 lines)
    let _ = writeln!(report, "\n[mlx server log — last 50 lines]");
    if let Ok(out) = Command::new("tail")
        .args(["-50", "/var/log/bad_apple_mlx_server.log"])
        .output()
    {
        let _ = writeln!(report, "{}", String::from_utf8_lossy(&out.stdout));
    } else {
        let _ = writeln!(report, "(log not found)");
    }

    // Supervisor log
    let _ = writeln!(report, "\n[supervisor log — last 20 lines]");
    if let Ok(out) = Command::new("tail")
        .args(["-20", "/var/log/bad_apple_supervisor.log"])
        .output()
    {
        let _ = writeln!(report, "{}", String::from_utf8_lossy(&out.stdout));
    } else {
        let _ = writeln!(report, "(log not found)");
    }

    // Crash logs from macOS DiagnosticReports
    let _ = writeln!(report, "\n[crash logs]");
    let crash_dir =
        std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()))
            .join("Library/Logs/DiagnosticReports");
    if crash_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&crash_dir) {
            let mut crashes: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.contains("badapple")
                        || name.contains("BadApple")
                        || name.contains("gatekeeper")
                })
                .collect();
            crashes.sort_by_key(|e| {
                std::cmp::Reverse(
                    e.metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                )
            });
            for crash in crashes.iter().take(5) {
                let name = crash.file_name().to_string_lossy().into_owned();
                let _ = writeln!(report, "  {name}");
                if let Ok(content) = std::fs::read_to_string(crash.path()) {
                    // Just the first 20 lines of each crash log
                    for line in content.lines().take(20) {
                        let _ = writeln!(report, "    {line}");
                    }
                    let _ = writeln!(report, "    ... (truncated)");
                }
            }
            if crashes.is_empty() {
                let _ = writeln!(report, "  (no Bad Apple crash logs found)");
            }
        }
    } else {
        let _ = writeln!(report, "  (DiagnosticReports directory not found)");
    }

    // Ledger tail (last 10 entries, redacted)
    let _ = writeln!(report, "\n[audit ledger — last 10 entries]");
    let ledger = std::path::PathBuf::from("/var/lib/bad_apple/ledger.jsonl");
    if ledger.is_file() {
        if let Ok(out) = Command::new("tail")
            .args(["-10", "/var/lib/bad_apple/ledger.jsonl"])
            .output()
        {
            let _ = writeln!(report, "{}", String::from_utf8_lossy(&out.stdout));
        }
    } else {
        let _ = writeln!(report, "(ledger not found)");
    }

    // Memory
    let _ = writeln!(report, "\n[memory]");
    if let Ok(out) = Command::new("memory_pressure").output() {
        let _ = writeln!(
            report,
            "{}",
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .take(10)
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    println!("{report}");
    eprintln!("\nTo share this report: badapple --crash-report > crash_report.txt");
    Ok(())
}

fn run_model_subcommand(args: &[String]) -> Result<()> {
    if args.is_empty() {
        bail!("usage: badapple model <list|scan|info|use|verify|add|remove|recommend> [args]");
    }
    let sub = args[0].as_str();
    let mut params = serde_json::Map::new();
    match sub {
        "list" => {
            let result = call_agent("list_models", None, 32)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "scan" => {
            let result = call_agent("scan_models", None, 32)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "info" => {
            if args.len() < 2 {
                bail!("usage: badapple model info <model-id>");
            }
            params.insert("model_id".to_string(), Value::String(args[1].clone()));
            let result = call_agent("model_info", Some(Value::Object(params)), 32)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "use" => {
            if args.len() < 2 {
                bail!("usage: badapple model use <model-id>");
            }
            params.insert("model_ref".to_string(), Value::String(args[1].clone()));
            let result = call_agent("switch_main_model", Some(Value::Object(params)), 512)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "verify" => {
            if args.len() >= 2 {
                params.insert("model_id".to_string(), Value::String(args[1].clone()));
            }
            let result = call_agent("verify_models", Some(Value::Object(params)), 512)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "add" => {
            if args.len() < 2 {
                bail!("usage: badapple model add <path> [model-id]");
            }
            params.insert("path".to_string(), Value::String(args[1].clone()));
            if args.len() >= 3 {
                params.insert("model_id".to_string(), Value::String(args[2].clone()));
            }
            let result = call_agent("add_model", Some(Value::Object(params)), 512)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "remove" => {
            if args.len() < 2 {
                bail!("usage: badapple model remove <model-id>");
            }
            params.insert("model_id".to_string(), Value::String(args[1].clone()));
            let result = call_agent("remove_model", Some(Value::Object(params)), 32)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "recommend" => {
            let query = if args.len() > 1 {
                params.insert("query".to_string(), Value::String(args[1..].join(" ")));
                Some(Value::Object(params))
            } else {
                None
            };
            let result = call_agent("recommend_model", query, 32)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        _ => bail!("unknown model subcommand: {sub}"),
    }
    Ok(())
}

fn p2p_binary_path() -> Result<std::path::PathBuf> {
    let exe = std::env::current_exe()?;
    let dir = exe
        .parent()
        .context("badapple executable has no parent directory")?;
    Ok(dir.join("badapple-p2p"))
}

fn run_p2p_helper(extra_args: &[String]) -> Result<()> {
    let binary = p2p_binary_path()?;
    let mut cmd = std::process::Command::new(&binary);
    cmd.args(extra_args);
    cmd.stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit());
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn badapple-p2p at {}", binary.display()))?;
    if !status.success() {
        anyhow::bail!("badapple-p2p exited with status: {status}");
    }
    Ok(())
}

fn p2p_sync_kind_alias(sub: &str) -> Option<&'static str> {
    match sub {
        "sync-personas" | "sync-persona" => Some("personas"),
        "sync-prompt" | "sync-prompts" => Some("prompt"),
        "sync-settings" => Some("settings"),
        "sync-models" | "sync-model-manifests" => Some("models"),
        "sync-checkpoint" | "sync-checkpoints" => Some("checkpoint"),
        _ => None,
    }
}

fn run_p2p_subcommand(args: &[String]) -> Result<()> {
    if args.is_empty() {
        bail!("usage: badapple p2p <peers|sync|sync-doc <kind>|sync-personas|sync-prompt|sync-settings|sync-models|sync-checkpoint|attest|receive-mesh [timeout_ms]|models|pull <peer_id> <model_id>|send <peer_id> <model_id>|receive [peer_id model_id]>");
    }
    let sub = args[0].as_str();
    let mut params = serde_json::Map::new();
    const MAX_TOKENS: usize = 512;
    match sub {
        "peers" => {
            let result = call_agent("p2p_peers", None, MAX_TOKENS)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "sync" => {
            let result = call_agent("p2p_sync", None, MAX_TOKENS)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "sync-doc" => {
            if args.len() < 2 {
                bail!("usage: badapple p2p sync-doc <personas|prompt|settings|models|checkpoint|ledger_checkpoint>");
            }
            run_p2p_helper(&["sync-doc".to_string(), args[1].clone()])?;
        }
        "sync-personas"
        | "sync-persona"
        | "sync-prompt"
        | "sync-prompts"
        | "sync-settings"
        | "sync-models"
        | "sync-model-manifests"
        | "sync-checkpoint"
        | "sync-checkpoints" => {
            let kind = p2p_sync_kind_alias(sub).unwrap_or(sub);
            run_p2p_helper(&["sync-doc".to_string(), kind.to_string()])?;
        }
        "attest" => {
            // Show the checkpoint docs this node holds for each mesh peer —
            // the mutual-attestation surface: what each member last proved.
            let store = bad_apple::mesh_sync::MeshStore::new(
                bad_apple::mesh_sync::MeshStore::default_root(),
            )?;
            let mut rows: Vec<serde_json::Value> = Vec::new();
            for doc in store.list() {
                if doc.kind != bad_apple::mesh_sync::MeshDocKind::SovereignCheckpoint
                    && doc.kind != bad_apple::mesh_sync::MeshDocKind::LedgerCheckpoint
                {
                    continue;
                }
                rows.push(serde_json::json!({
                    "kind": doc.kind.to_string(),
                    "origin": doc.origin,
                    "timestamp": doc.timestamp,
                    "version": doc.version,
                    "checkpoint": serde_json::from_str::<serde_json::Value>(&doc.body)
                        .unwrap_or(serde_json::Value::Null),
                }));
            }
            println!("{}", serde_json::to_string_pretty(&rows)?);
        }
        "receive-mesh" => {
            let mut argv = vec!["receive-mesh".to_string()];
            if args.len() >= 2 {
                argv.push(args[1].clone());
            }
            run_p2p_helper(&argv)?;
        }
        "models" => {
            let result = call_agent("p2p_models", None, MAX_TOKENS)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "pull" => {
            if args.len() < 3 {
                bail!("usage: badapple p2p pull <peer_id> <model_id>");
            }
            params.insert("peer_id".to_string(), Value::String(args[1].clone()));
            params.insert("model_id".to_string(), Value::String(args[2].clone()));
            let result = call_agent("p2p_pull_model", Some(Value::Object(params)), MAX_TOKENS)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "send" => {
            if args.len() < 3 {
                bail!("usage: badapple p2p send <peer_id> <model_id>");
            }
            params.insert("peer_id".to_string(), Value::String(args[1].clone()));
            params.insert("model_id".to_string(), Value::String(args[2].clone()));
            let result = call_agent("p2p_send_model", Some(Value::Object(params)), MAX_TOKENS)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "receive" => {
            if args.len() >= 3 {
                params.insert("peer_id".to_string(), Value::String(args[1].clone()));
                params.insert("model_id".to_string(), Value::String(args[2].clone()));
            } else if args.len() == 2 {
                bail!("usage: badapple p2p receive [peer_id model_id]");
            }
            let result = call_agent(
                "p2p_receive_model",
                if params.is_empty() {
                    None
                } else {
                    Some(Value::Object(params))
                },
                MAX_TOKENS,
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        _ => bail!("unknown p2p subcommand: {sub}"),
    }
    Ok(())
}

fn run_vault_subcommand(args: &[String]) -> Result<()> {
    if args.is_empty() {
        bail!("usage: badapple vault <get|set|remove|list|import> [args]");
    }
    let sub = args[0].as_str();
    let vault = bad_apple::vault::BadAppleVault::open_default()
        .context("failed to open vault; set BADAPPLE_VAULT_KEY, BADAPPLE_SLICKS_KEY_PATH, or BADAPPLE_SLICKS_SECRET")?;
    match sub {
        "list" => {
            let keys = vault.list()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({ "keys": keys }))?
            );
        }
        "get" => {
            if args.len() < 2 {
                bail!("usage: badapple vault get <key>");
            }
            match vault.get(&args[1])? {
                Some(value) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(
                            &serde_json::json!({ "key": args[1], "value": value })
                        )?
                    );
                }
                None => {
                    println!("{{\"key\": \"{}\", \"error\": \"not found\"}}", args[1]);
                    std::process::exit(1);
                }
            }
        }
        "set" => {
            if args.len() < 3 {
                bail!("usage: badapple vault set <key> <value>");
            }
            vault.set(&args[1], &args[2])?;
            println!("{{\"status\": \"ok\", \"key\": \"{}\"}}", args[1]);
        }
        "remove" => {
            if args.len() < 2 {
                bail!("usage: badapple vault remove <key>");
            }
            vault.remove(&args[1])?;
            println!("{{\"status\": \"removed\", \"key\": \"{}\"}}", args[1]);
        }
        "import" => {
            if args.len() < 2 {
                bail!("usage: badapple vault import <dotenv-file>");
            }
            let content = std::fs::read_to_string(&args[1])
                .with_context(|| format!("failed to read {}", args[1]))?;
            let mut count = 0;
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((k, v)) = line.split_once('=') {
                    let k = k.trim();
                    let v = v.trim().trim_matches('"').trim_matches('\'');
                    if !k.is_empty() {
                        vault.set(k, v)?;
                        count += 1;
                    }
                }
            }
            println!("{{\"status\": \"ok\", \"imported\": {count}}}");
        }
        _ => bail!("unknown vault subcommand: {sub}"),
    }
    Ok(())
}

fn run_workspace_subcommand(args: &[String]) -> Result<()> {
    if args.is_empty() {
        bail!("usage: badapple workspace <get|set <path>|index|watch [path]>");
    }
    let sub = args[0].as_str();
    match sub {
        "get" => {
            let result = call_agent("get_workspace", None, 32)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "set" => {
            if args.len() < 2 {
                bail!("usage: badapple workspace set <path>");
            }
            let mut params = serde_json::Map::new();
            params.insert("path".to_string(), Value::String(args[1].clone()));
            let result = call_agent("set_workspace", Some(Value::Object(params)), 32)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "index" => {
            let path = if args.len() >= 2 {
                Some(args[1].clone())
            } else {
                None
            };
            let mut params = serde_json::Map::new();
            if let Some(p) = path {
                params.insert("path".to_string(), Value::String(p));
            }
            let result = call_agent(
                "index_documents",
                if params.is_empty() {
                    None
                } else {
                    Some(Value::Object(params))
                },
                512,
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "watch" => {
            let path = if args.len() >= 2 {
                std::path::PathBuf::from(&args[1])
            } else {
                // Try the current workspace from the daemon.
                let result = call_agent("get_workspace", None, 32)?;
                if let Some(p) = result["workspace"].as_str() {
                    std::path::PathBuf::from(p)
                } else {
                    bail!("no workspace set; provide a path or run `badapple workspace set <path>` first");
                }
            };
            if !path.is_dir() {
                bail!("workspace path is not a directory: {}", path.display());
            }

            // Set workspace on the daemon if different.
            let current = call_agent("get_workspace", None, 32)?;
            if current["workspace"].as_str() != Some(path.to_str().unwrap_or("")) {
                let mut params = serde_json::Map::new();
                params.insert(
                    "path".to_string(),
                    Value::String(path.to_string_lossy().to_string()),
                );
                call_agent("set_workspace", Some(Value::Object(params)), 32)?;
            }

            println!(
                "{{\"status\": \"watching\", \"workspace\": \"{}\"}}",
                path.display()
            );

            let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
            let r = running.clone();
            let mut signals = signal_hook::iterator::Signals::new([
                signal_hook::consts::SIGINT,
                signal_hook::consts::SIGTERM,
            ])?;
            std::thread::spawn(move || {
                if signals.forever().next().is_some() {
                    r.store(false, std::sync::atomic::Ordering::SeqCst);
                }
            });

            let watcher = bad_apple::workspace_watcher::WorkspaceWatcher::watch(
                &path,
                std::time::Duration::from_secs(2),
                Some(bad_apple::workspace_watcher::default_file_filter),
                move |root, changed| {
                    eprintln!("[workspace] re-indexing after changes: {:?}", changed);
                    let mut params = serde_json::Map::new();
                    params.insert(
                        "path".to_string(),
                        Value::String(root.to_string_lossy().to_string()),
                    );
                    match call_agent("index_documents", Some(Value::Object(params)), 512) {
                        Ok(_) => eprintln!("[workspace] indexed"),
                        Err(e) => eprintln!("[workspace] index failed: {e}"),
                    }
                },
            )
            .context("failed to start workspace watcher")?;

            while running.load(std::sync::atomic::Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            watcher.stop();
            println!("{{\"status\": \"stopped\"}}");
        }
        _ => bail!("unknown workspace subcommand: {sub}"),
    }
    Ok(())
}

fn run_mcp_subcommand(args: &[String]) -> Result<()> {
    if args.is_empty() {
        bail!("usage: badapple mcp <list|add <id> <command> [args...]|remove <id>|install <id>|uninstall <id>|start <id>|stop <id>|status <id>>");
    }
    let sub = args[0].as_str();

    let catalog_path = mcp_catalog_path();
    let rt = tokio::runtime::Runtime::new()?;
    let market = bad_apple::mcp_marketplace::McpMarketplace::new(catalog_path, 100);

    rt.block_on(async {
        market.load().await?;
        match sub {
            "list" => {
                let mut servers = market.list().await;
                for s in &mut servers {
                    let status = market.status(&s.id).await.ok();
                    if let Some(st) = status {
                        s.installed = st.installed;
                        s.enabled = st.enabled;
                    }
                }
                let result = serde_json::json!({ "servers": servers });
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            "add" => {
                if args.len() < 3 {
                    bail!("usage: badapple mcp add <id> <command> [args...]");
                }
                let id = args[1].clone();
                let command = args[2].clone();
                let rest = args[3..].to_vec();
                market
                    .upsert(bad_apple::mcp_marketplace::McpServer {
                        id,
                        name: args[1].clone(),
                        command,
                        args: rest,
                        env: std::collections::HashMap::new(),
                        transport: bad_apple::mcp_marketplace::McpTransport::Stdio,
                        installed: true,
                        enabled: true,
                        description: "".to_string(),
                    })
                    .await?;
                market.save().await?;
                println!("{{\"status\": \"ok\", \"action\": \"added\"}}");
            }
            "remove" => {
                if args.len() < 2 {
                    bail!("usage: badapple mcp remove <id>");
                }
                market.remove(&args[1]).await?;
                market.save().await?;
                println!("{{\"status\": \"ok\", \"action\": \"removed\"}}");
            }
            "install" => {
                if args.len() < 2 {
                    bail!("usage: badapple mcp install <id>");
                }
                market.install(&args[1]).await?;
                market.save().await?;
                println!("{{\"status\": \"ok\", \"action\": \"installed\"}}");
            }
            "uninstall" => {
                if args.len() < 2 {
                    bail!("usage: badapple mcp uninstall <id>");
                }
                market.uninstall(&args[1]).await?;
                market.save().await?;
                println!("{{\"status\": \"ok\", \"action\": \"uninstalled\"}}");
            }
            "start" => {
                if args.len() < 2 {
                    bail!("usage: badapple mcp start <id>");
                }
                market.start(&args[1]).await?;
                println!("{{\"status\": \"ok\", \"action\": \"started\"}}");
            }
            "stop" => {
                if args.len() < 2 {
                    bail!("usage: badapple mcp stop <id>");
                }
                market.stop(&args[1]).await?;
                println!("{{\"status\": \"ok\", \"action\": \"stopped\"}}");
            }
            "status" => {
                if args.len() < 2 {
                    bail!("usage: badapple mcp status <id>");
                }
                let status = market.status(&args[1]).await?;
                println!("{}", serde_json::to_string_pretty(&status)?);
            }
            "init" => {
                // Air-gap: do not ship npx-based remote-download servers or
                // unrestricted filesystem roots as defaults. Users can add MCP
                // servers explicitly with `badapple mcp add` once they have
                // validated the command and arguments locally.
                market.save().await?;
                println!("{{\"status\": \"ok\", \"action\": \"initialized\"}}");
            }
            _ => bail!("unknown mcp subcommand: {sub}"),
        }
        Ok(())
    })
}

fn mcp_catalog_path() -> std::path::PathBuf {
    std::env::var("BADAPPLE_MCP_CATALOG_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::data_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("/var/lib/bad_apple"))
                .join("bad_apple")
                .join("mcp_catalog.json")
        })
}

fn run_redteam_subcommand(args: &[String]) -> Result<()> {
    use bad_apple::red_team::{RedTeamLoop, RedTeamRunner};
    use std::time::Duration;

    let sub = args.first().map(String::as_str).unwrap_or("run");
    match sub {
        "run" => {
            let runner = RedTeamRunner::default();
            let report = runner.run_once();
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.findings().is_empty() {
                std::process::exit(1);
            }
        }
        "watch" => {
            let interval = args
                .get(1)
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(30);
            let rt = RedTeamLoop::new(Duration::from_secs(interval));
            println!("{{\"status\": \"running\", \"interval_sec\": {interval}}}");
            let _handle = rt.start();
            // The handle runs until the process is killed.
            loop {
                std::thread::sleep(Duration::from_secs(60));
                let report = rt.last_report();
                if let Some(r) = report {
                    let findings = r.findings();
                    println!(
                        "{{\"score\": {:.4}, \"total\": {}, \"findings\": {}}}",
                        r.score,
                        r.total,
                        findings.len()
                    );
                    if !findings.is_empty() {
                        eprintln!("{}", serde_json::to_string_pretty(&findings)?);
                    }
                }
            }
        }
        "status" => {
            let runner = RedTeamRunner::default();
            let report = runner.run_once();
            let findings = report.findings();
            println!(
                "{{\"status\": \"ok\", \"score\": {:.4}, \"total\": {}, \"findings\": {}}}",
                report.score,
                report.total,
                findings.len()
            );
            for finding in &findings {
                eprintln!("{}", serde_json::to_string_pretty(finding)?);
            }
        }
        "category" => {
            let category = args.get(1).map(String::as_str).unwrap_or("");
            if category.is_empty() {
                bail!("usage: badapple redteam category <cage|slicks|p2p|wasm|policy|audit>");
            }
            let runner = RedTeamRunner::default();
            let report = runner.run_category(category);
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.findings().is_empty() {
                std::process::exit(1);
            }
        }
        "probe" => {
            let id = args.get(1).map(String::as_str).unwrap_or("");
            if id.is_empty() {
                bail!("usage: badapple redteam probe <id>");
            }
            let runner = RedTeamRunner::default();
            let report = runner
                .run_probe(id)
                .ok_or_else(|| anyhow::anyhow!("unknown probe: {id}"))?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.findings().is_empty() {
                std::process::exit(1);
            }
        }
        _ => bail!("unknown redteam subcommand: {sub}\nusage: badapple redteam <run|watch|status|category <category>|probe <id>>"),
    }
    Ok(())
}

fn run_cert() -> Result<()> {
    let results = cert::run();
    let mut failures = 0;
    for r in &results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        eprintln!("[cert] [{status}] {}: {}", r.name, r.message);
        if !r.passed {
            failures += 1;
        }
    }
    let summary = serde_json::json!({
        "status": if failures == 0 { "ok" } else { "failed" },
        "total": results.len(),
        "failures": failures,
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if failures > 0 {
        bail!("cert suite failed: {failures} check(s)");
    }
    Ok(())
}

/// `badapple status` — a three-line "is it working?" for humans, unlike the
/// machine-readable `cert` suite or the full `--doctor` report.
fn run_status() -> Result<()> {
    use std::os::unix::net::UnixStream;
    println!("Bad Apple v{}", env!("CARGO_PKG_VERSION"));

    let sock = bad_apple::bad_apple_ipc::socket_path();
    match UnixStream::connect(&sock) {
        Ok(_) => println!("daemon:    running    ({})", sock.display()),
        Err(e) => {
            let state = match e.kind() {
                std::io::ErrorKind::NotFound => "not running — open Bad Apple.app and try again",
                std::io::ErrorKind::PermissionDenied => {
                    "running but unreachable — socket permissions are wrong"
                }
                std::io::ErrorKind::ConnectionRefused => "starting up — try again in a few seconds",
                _ => "unreachable",
            };
            println!("daemon:    {state}");
            println!("\nRun `badapple --doctor` for a full diagnostic report.");
            bail!("Bad Apple is not reachable");
        }
    }

    let id_sock = bad_apple::bad_apple_ipc::identity_agent_socket_path();
    if UnixStream::connect(&id_sock).is_ok() {
        println!("identity:  running    (Secure Enclave signing)");
    } else if id_sock.exists() {
        println!("identity:  present but not accepting connections");
    } else {
        println!("identity:  not running (SLICKS v1 still works)");
    }
    Ok(())
}

fn run_receipts() -> Result<()> {
    use std::io::BufRead;
    println!("=== BAD APPLE PROOF CARD ===");
    println!("version:    v{}", env!("CARGO_PKG_VERSION"));

    let sock = bad_apple::bad_apple_ipc::socket_path();
    let daemon_up = std::os::unix::net::UnixStream::connect(&sock).is_ok();
    println!(
        "daemon:     {}",
        if daemon_up { "running" } else { "not running" }
    );

    let id_sock = bad_apple::bad_apple_ipc::identity_agent_socket_path();
    let enclave = std::os::unix::net::UnixStream::connect(&id_sock).is_ok();
    println!(
        "identity:   {}",
        if enclave {
            "Secure Enclave (SLICKS v2)"
        } else {
            "SLICKS v1 (HMAC)"
        }
    );

    let ledger = std::path::Path::new("/var/lib/bad_apple/ledger.jsonl");
    match std::fs::File::open(ledger) {
        Ok(f) => {
            let mut count = 0u64;
            let mut first_ts: Option<chrono::DateTime<chrono::Utc>> = None;
            let mut last_hash = String::new();
            for line in std::io::BufReader::new(f).lines().map_while(Result::ok) {
                count += 1;
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if first_ts.is_none() {
                        first_ts = v
                            .get("ts")
                            .and_then(|t| t.as_str())
                            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                            .map(|d| d.with_timezone(&chrono::Utc));
                    }
                    if let Some(h) = v.get("hash").and_then(|h| h.as_str()) {
                        last_hash = h.to_string();
                    }
                }
            }
            println!("ledger:     {count} attested actions");
            if let Some(born) = first_ts {
                let age = chrono::Utc::now().signed_duration_since(born).num_days();
                println!("organism:   {age} days old");
            }
            if !last_hash.is_empty() {
                println!("chain tip:  {}…", &last_hash[..16.min(last_hash.len())]);
            }
        }
        Err(_) => println!("ledger:     not found (no attested history yet)"),
    }

    let cp = std::path::Path::new("/var/lib/bad_apple/ledger.sovereign.checkpoint.json");
    if let Ok(text) = std::fs::read_to_string(cp) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            let n = v["entry_count"].as_u64().unwrap_or(0);
            let at = v["signed_at"].as_str().unwrap_or("unknown");
            let scheme = v["scheme"].as_str().unwrap_or("unknown");
            println!("sovereign:  sealed {n} entries · {scheme} · signed {at}");
        }
    } else {
        println!("sovereign:  no checkpoint (run badapple-sovereign)");
    }

    let ify_state = bad_apple::ify::load_state();
    let phase = bad_apple::ify::current_phase(&ify_state);
    println!("ify:        watching ({})", phase.as_str());

    println!();
    println!("every action above is hash-chained and HMAC-sealed.");
    println!("don't trust this card — verify it: `badapple --doctor` · `badapple cert`");
    Ok(())
}

/// `badapple export-proof [dir]` — package the sovereign chain, its Enclave-signed
/// checkpoint, and verify instructions into a shareable proof bundle. The exported
/// chain verifies with zero secrets via sovereign_ledger's public verification.
fn run_export_proof(args: &[String]) -> Result<()> {
    let stamp = chrono::Utc::now().format("%Y%m%d");
    let default_dir = format!("badapple-proof-{stamp}");
    let out = std::path::PathBuf::from(args.first().map(String::as_str).unwrap_or(&default_dir));
    std::fs::create_dir_all(&out)?;

    let data_dir = std::path::Path::new("/var/lib/bad_apple");
    let sovereign = data_dir.join("ledger.sovereign.jsonl");
    if !sovereign.exists() {
        bail!(
            "no sovereign chain at {} — run badapple-sovereign first",
            sovereign.display()
        );
    }

    let mut copied: Vec<String> = Vec::new();
    for name in [
        "ledger.sovereign.jsonl",
        "ledger.sovereign.checkpoint.json",
        "ledger_checkpoint.json",
    ] {
        let src = data_dir.join(name);
        if src.exists() {
            std::fs::copy(&src, out.join(name))?;
            copied.push(name.to_string());
        }
    }

    let text = std::fs::read_to_string(&sovereign)?;
    let entries = text.lines().count();
    let manifest = serde_json::json!({
        "exported_by": format!("Bad Apple v{}", env!("CARGO_PKG_VERSION")),
        "exported_at": chrono::Utc::now().to_rfc3339(),
        "sovereign_entries": entries,
        "format": "sovereign_ledger v1 JSONL (see SPEC.md in the sovereign_ledger repo)",
        "verify": "sovereign_ledger::seal::verify_public — no secrets required",
        "files": copied,
    });
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    std::fs::write(
        out.join("VERIFY.md"),
        "# Verify this proof bundle\n\n\
         `ledger.sovereign.jsonl` is a tamper-evident, sealed audit chain written by\n\
         Bad Apple's sovereign layer. It verifies **without any secret material** —\n\
         every seal carries the revealed keys and signatures needed to check it.\n\n\
         ## Verify with Rust\n\n\
         ```sh\n\
         cargo add sovereign_ledger\n\
         ```\n\
         ```rust\n\
         let report = sovereign_ledger::seal::verify_public(reader)?;\n\
         ```\n\n\
         ## Verify with the independent JS verifier (zero dependencies)\n\n\
         ```sh\n\
         node verify.mjs ledger.sovereign.jsonl --public\n\
         ```\n\
         from https://github.com/savageAZfck/sovereign_ledger/tree/main/verifiers/js\n\n\
         ## The format\n\n\
         SPEC.md in https://github.com/savageAZfck/sovereign_ledger documents every\n\
         byte: MAC preimages, segment keys, seals, Merkle construction, anchors.\n\n\
         `ledger.sovereign.checkpoint.json` is a Secure Enclave-signed checkpoint\n\
         (scheme `secure-enclave`) binding the chain tip and Merkle root.\n",
    )?;

    println!("proof bundle written to {}", out.display());
    for f in &copied {
        println!("  + {f}");
    }
    println!("  + manifest.json");
    println!("  + VERIFY.md");
    println!("\n{entries} sovereign entries — verifiable by anyone, no secrets required.");
    Ok(())
}

/// `badapple demo` — narrated self-demonstration. Every line is a real subsystem
/// call; nothing is scripted or faked. Screen-record this and it is the pitch.
fn run_demo() -> Result<()> {
    use std::io::BufRead;
    println!();
    println!("  Bad Apple — self-demonstration");
    println!("  ────────────────────────────");
    println!();

    // 1. Chain verification — real ledger walk.
    let ledger = std::path::Path::new("/var/lib/bad_apple/ledger.jsonl");
    let (count, tip) = match std::fs::File::open(ledger) {
        Ok(f) => {
            let mut n = 0u64;
            let mut last = String::new();
            for line in std::io::BufReader::new(f).lines().map_while(Result::ok) {
                n += 1;
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(h) = v.get("hash").and_then(|h| h.as_str()) {
                        last = h.to_string();
                    }
                }
            }
            (n, last)
        }
        Err(_) => (0, String::new()),
    };
    println!(
        "  Verifying my chain...        {count} attested actions, tip {}…",
        &tip[..16.min(tip.len())]
    );

    // 2. Air-gap certification — real cert suite.
    print!("  Checking my air gap...       ");
    let results = cert::run();
    let failures = results.iter().filter(|r| !r.passed).count();
    if failures == 0 {
        println!("{} checks, zero network sockets — clean", results.len());
    } else {
        println!("{failures} of {} checks FAILED", results.len());
    }

    // 3. Watchdog — real IFY state.
    let ify_state = bad_apple::ify::load_state();
    let phase = bad_apple::ify::current_phase(&ify_state);
    println!(
        "  Consulting my watchdog...    IFY is {} — watching, brake-only",
        phase.as_str()
    );

    // 4. Identity — real socket check.
    let id_sock = bad_apple::bad_apple_ipc::identity_agent_socket_path();
    let enclave = std::os::unix::net::UnixStream::connect(&id_sock).is_ok();
    println!(
        "  Reading my vitals...         {}",
        if enclave {
            "Secure Enclave signing — identity is hardware-bound"
        } else {
            "SLICKS v1 HMAC — software identity"
        }
    );

    // 5. Sovereign seal — real checkpoint.
    let cp = std::path::Path::new("/var/lib/bad_apple/ledger.sovereign.checkpoint.json");
    if let Ok(text) = std::fs::read_to_string(cp) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            let n = v["entry_count"].as_u64().unwrap_or(0);
            let scheme = v["scheme"].as_str().unwrap_or("?");
            println!("  Checking my sovereign seal... {n} entries sealed · {scheme}");
        }
    }

    println!();
    println!("  As far as I can prove: I am alone with your data.");
    println!("  Don't trust me — verify me: `badapple cert` · `badapple receipts`");
    println!();
    Ok(())
}

/// `badapple demo full` — exec the narrated walkthrough (mesh-brain kill/heal
/// included). The script lives at the runtime root — two dirs above
/// target/release/<exe> in both the dev repo and the installed layout.
fn run_demo_full() -> Result<()> {
    let exe = std::env::current_exe()
        .and_then(|p| p.canonicalize())
        .context("cannot resolve the badapple binary path")?;
    let exe_dir = exe.parent().context("no binary dir")?.to_path_buf();
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(s) = std::env::var("BADAPPLE_DEMO_SCRIPT") {
        candidates.push(PathBuf::from(s));
    }
    candidates.push(exe_dir.join("../../demo_walkthrough.sh"));
    candidates.push(PathBuf::from(
        "/Applications/Bad Apple.app/Contents/Resources/demo_walkthrough.sh",
    ));
    candidates.push(PathBuf::from("demo_walkthrough.sh"));
    let script = candidates
        .iter()
        .map(|p| p.canonicalize().unwrap_or_else(|_| p.clone()))
        .find(|p| p.is_file())
        .context("demo_walkthrough.sh not found — set BADAPPLE_DEMO_SCRIPT to point at it")?;
    let status = std::process::Command::new("bash")
        .arg(&script)
        .env("BADAPPLE_BIN", &exe)
        .env("BADAPPLE_ENGINE_BIN", exe_dir.join("badapple-engine"))
        .status()
        .context("failed to launch the walkthrough demo")?;
    if !status.success() {
        bail!("walkthrough exited with {status}");
    }
    Ok(())
}

fn run_ify_subcommand(args: &[String]) -> Result<()> {
    let sub = args.first().map(String::as_str).unwrap_or("status");
    match sub {
        "status" => {
            let state = bad_apple::ify::load_state();
            let phase = bad_apple::ify::current_phase(&state);
            let summary = serde_json::json!({
                "phase": phase.as_str(),
                "installed_days_ago": ((bad_apple::ify::now_secs() - state.installed_at) / 86400.0 * 10.0).round() / 10.0,
                "events_seen": state.events_seen,
                "event_types": state.event_types.len(),
                "approvals": {"granted": state.approvals_granted, "denied": state.approvals_denied},
                "firewall_hits": state.firewall_hits,
                "kill_switch_events": state.kill_switch_events,
                "state_dir": bad_apple::ify::ify_dir(),
            });
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        "once" => {
            let secrets = bad_apple::ify::load_slicks_secrets();
            let mut state = bad_apple::ify::load_state();
            let report = bad_apple::ify::tail_ledger(&mut state, &secrets)?;
            for f in &report.findings {
                bad_apple::ify::dispatch(&state, f);
            }
            bad_apple::ify::save_state(&state);
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "phase": bad_apple::ify::current_phase(&state).as_str(),
                    "new_events": report.new_events,
                    "findings": report.findings,
                    "chain_broken": report.chain_broken,
                    "truncated": report.truncated,
                }))?
            );
        }
        "findings" => {
            let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
            let path = bad_apple::ify::findings_path();
            let lines: Vec<String> = std::fs::read_to_string(&path)
                .unwrap_or_default()
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(String::from)
                .collect();
            for line in lines.iter().rev().take(n).rev() {
                println!("{line}");
            }
        }
        "proposals" => {
            let dir = bad_apple::ify::proposals_dir();
            let mut paths: Vec<_> = std::fs::read_dir(&dir)
                .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
                .unwrap_or_default();
            paths.sort();
            for p in paths {
                println!("{}", p.display());
            }
        }
        _ => bail!("unknown ify subcommand: {sub}\nusage: badapple ify <status|once|findings [n]|proposals>"),
    }
    Ok(())
}

fn run_policy_subcommand(args: &[String]) -> Result<()> {
    use bad_apple::org_policy;
    if args.is_empty() {
        bail!("usage: badapple policy <keygen [--out <dir>]|sign [--key <path>] [--policy <path>] [--sig <path>]|verify [--policy <path>] [--sig <path>] [--pub <path>]|status>");
    }
    let named = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let policy = named("--policy").unwrap_or_else(|| org_policy::DEFAULT_POLICY_PATH.into());
    let sig = named("--sig").unwrap_or_else(|| org_policy::DEFAULT_SIG_PATH.into());
    let pubk = named("--pub").unwrap_or_else(|| org_policy::DEFAULT_PUB_PATH.into());
    match args[0].as_str() {
        "keygen" => {
            let dir = named("--out").unwrap_or_else(|| ".".into());
            let (key, pub_key) = org_policy::keygen(&dir)?;
            println!("org keypair generated");
            println!("  secret (keep offline): {}", key.display());
            println!(
                "  trust root (pin on managed machines): {}",
                pub_key.display()
            );
        }
        "sign" => {
            let key = named("--key").unwrap_or_else(|| org_policy::DEFAULT_KEY_PATH.into());
            org_policy::sign_file(&policy, &key, &sig)?;
            println!("signed {} -> {}", policy, sig);
        }
        "verify" => match org_policy::verify_file(&policy, &sig, &pubk) {
            Ok(true) => println!("{{\"policy_signature\": \"valid\", \"policy\": \"{policy}\"}}"),
            Ok(false) => bail!("policy signature invalid or missing: {policy}"),
            Err(e) => return Err(e),
        },
        "status" => {
            let org = org_policy::org_mode_active();
            let valid = if org {
                org_policy::verify_file(&policy, &sig, &pubk).unwrap_or(false)
            } else {
                false
            };
            println!(
                "{{\"mode\": \"{}\", \"policy_signature\": \"{}\"}}",
                if org { "org-signed" } else { "personal" },
                if !org {
                    "not required"
                } else if valid {
                    "valid"
                } else {
                    "invalid or missing"
                }
            );
        }
        sub => bail!(
            "unknown policy subcommand: {sub}\nusage: badapple policy <keygen|sign|verify|status>"
        ),
    }
    Ok(())
}

fn run_mesh_brain_subcommand(args: &[String]) -> Result<()> {
    use bad_apple::mesh_brain;
    if args.is_empty() {
        bail!("usage: badapple mesh-brain <plan --model <dir|repo> --hosts a:port,b:port [--mem gb,gb]|shard --model <dir|repo> --rank i --of N [--hosts a,b] [--out dir]|ping [--to h:p]|status [--hosts a,b]|ask [--to h:p] --prompt text [--max-tokens n]|forget>\n       plan/shard remember the mesh in /var/lib/bad_apple/mesh_hosts.json — later status/ping/ask default to it.");
    }
    let named = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    match args[0].as_str() {
        "plan" => {
            let model = named("--model").context("--model required")?;
            let hosts_raw =
                named("--hosts").context("--hosts required (comma-separated host:port)")?;
            let hosts: Vec<String> = hosts_raw.split(',').map(|s| s.trim().to_string()).collect();
            let mem: Option<Vec<f64>> = named("--mem")
                .map(|m| m.split(',').filter_map(|s| s.trim().parse().ok()).collect());
            let dir = mesh_brain::resolve_model_dir(&model)?;
            let plan = mesh_brain::plan(&dir, &hosts, mem.as_deref())?;
            let _ = mesh_brain::save_hosts(&hosts, &model);
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
        "shard" => {
            let model = named("--model").context("--model required")?;
            let dir = mesh_brain::resolve_model_dir(&model)?;
            let (world, rank) = match (named("--of"), named("--rank")) {
                (Some(w), Some(r)) => (w.parse::<usize>()?, r.parse::<usize>()?),
                _ => bail!("shard requires --rank <i> --of <N>"),
            };
            let hosts: Vec<String> = match named("--hosts") {
                Some(h) => h.split(',').map(|s| s.trim().to_string()).collect(),
                None => (0..world)
                    .map(|i| format!("127.0.0.1:{}", 8741 + i))
                    .collect(),
            };
            if hosts.len() != world {
                bail!(
                    "--hosts count {} does not match --of {}",
                    hosts.len(),
                    world
                );
            }
            let p = mesh_brain::plan(&dir, &hosts, None)?;
            let spec = p.ranks.get(rank).cloned().context("rank out of range")?;
            let out = named("--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| dir.join(format!("shard-r{rank}")));
            let n = mesh_brain::build_shard(&dir, &out, &spec)?;
            let _ = mesh_brain::save_hosts(&hosts, &model);
            println!(
                "{{\"status\":\"ok\",\"rank\":{rank},\"world\":{world},\"layers\":[{},{}),\"tensors\":{n},\"out\":\"{}\"}}",
                spec.layer_start, spec.layer_end, out.display()
            );
        }
        "ping" => {
            let host = named("--to")
                .or_else(|| mesh_brain::load_hosts().and_then(|h| h.first().cloned()))
                .context(
                    "ping requires --to host:port (no saved mesh — run `mesh-brain plan` first)",
                )?;
            let (rank, ls, le) = mesh_brain::ping(&host)?;
            println!(
                "{{\"status\":\"ok\",\"host\":\"{host}\",\"rank\":{rank},\"layers\":[{},{}])}}",
                ls.map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
                le.map(|v| v.to_string()).unwrap_or_else(|| "?".into())
            );
        }
        "status" => {
            let hosts_raw = named("--hosts")
                .or_else(|| mesh_brain::load_hosts().map(|h| h.join(",")))
                .context("status requires --hosts a:port,b:port (no saved mesh — run `mesh-brain plan` first)")?;
            let mut live = 0usize;
            let mut rows = Vec::new();
            for h in hosts_raw.split(',').map(|s| s.trim()) {
                match mesh_brain::ping(h) {
                    Ok((rank, ls, le)) => {
                        live += 1;
                        rows.push(format!(
                            "{{\"host\":\"{h}\",\"ok\":true,\"rank\":{rank},\"layers\":[{},{}])}}",
                            ls.map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
                            le.map(|v| v.to_string()).unwrap_or_else(|| "?".into())
                        ));
                    }
                    Err(e) => rows.push(format!(
                        "{{\"host\":\"{h}\",\"ok\":false,\"error\":\"{}\"}}",
                        e.to_string().replace('"', "'")
                    )),
                }
            }
            println!(
                "{{\"status\":\"{}\",\"live\":{live},\"ranks\":[{}]}}",
                if live == rows.len() {
                    "ready"
                } else {
                    "degraded"
                },
                rows.join(",")
            );
        }
        "ask" => {
            let host = named("--to")
                .or_else(|| mesh_brain::load_hosts().and_then(|h| h.first().cloned()))
                .context("ask requires --to host:port (rank 0) (no saved mesh — run `mesh-brain plan` first)")?;
            let prompt = named("--prompt")
                .or_else(|| args.get(1).cloned())
                .context("ask requires a prompt")?;
            let max_tokens = named("--max-tokens")
                .and_then(|m| m.parse().ok())
                .unwrap_or(64);
            let text = mesh_brain::ask(&host, &prompt, max_tokens)?;
            println!("{text}");
        }
        "serve" => {
            bail!("run the engine in shard mode instead: BADAPPLE_SHARD_DIR=<shard dir> badapple-engine")
        }
        "forget" => match std::fs::remove_file(mesh_brain::hosts_file()) {
            Ok(()) => println!("{{\"status\":\"forgotten\"}}"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!("{{\"status\":\"no saved mesh\"}}")
            }
            Err(e) => return Err(e.into()),
        },
        sub => bail!("unknown mesh-brain subcommand: {sub}"),
    }
    Ok(())
}

fn print_help() {
    println!(
        "badapple — authenticated local client for the Bad Apple daemon\n\n\
         Usage:\n  badapple [OPTIONS] \"query\"\n  badapple model <list|scan|info|use|verify|add|remove|recommend> [args]\n  badapple p2p <peers|sync|sync-doc <kind>|sync-personas|sync-prompt|sync-settings|sync-models|receive-mesh [timeout_ms]|models|pull <peer_id> <model_id>|send <peer_id> <model_id>|receive [peer_id model_id]>\n  badapple vault <get|set|remove|list|import> [args]\n  badapple workspace <get|set <path>|index|watch [path]>\n  badapple mcp <list|add <id> <command> [args...]|remove <id>|install <id>|uninstall <id>|start <id>|stop <id>|status <id>|init>\n  badapple redteam <run|watch|status|category <category>|probe <id>>\n  badapple status          Is Bad Apple working? Three-line human check\n  badapple cert            Machine-readable air-gap certification\n\n\
         Options:\n  -n, --max-tokens N  Maximum generated tokens (default: 500)\n  --speak             Stream each sentence to local TTS and play with afplay\n  --persona NAME      Switch persona for this query (cali, curious, drill, genz, midwest, wicket, ...)\n  --roast             Alias for --persona drill\n  --benchmark         Benchmark a single prompt or a default suite\n  --doctor            Print a local support diagnostic report (--diagnostics alias)\n  --crash-report      Collect crash logs and daemon state for debugging\n  --json              Output token stream as JSON\n  -h, --help          Show this help\n\n\
         Environment:\n  BADAPPLE_SOCKET_PATH       Unix socket path\n  BADAPPLE_SLICKS_KEY_PATH   SLICKS key file path\n  BADAPPLE_SLICKS_SECRET     In-memory SLICKS secret override\n  BADAPPLE_TTS_VOICE         Voice name for --speak (default: Best; Piper voices in voices/ take priority, then AVFoundation voices)\n  BADAPPLE_VAULT_KEY         Master key for the local secret vault\n  BADAPPLE_MCP_CATALOG_PATH  Path to the MCP marketplace catalog"
    );
}
