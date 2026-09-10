use anyhow::{bail, Context, Result};
use bad_apple::bad_apple_ipc::{call_agent, query_with_metrics};
use bad_apple::cert;
use serde_json::Value;
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
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

    if prompt_parts.first().map(std::string::String::as_str) == Some("cert") {
        return run_cert();
    }

    let prompt = if prompt_parts.is_empty() && !benchmark_mode {
        bail!("usage: badapple [OPTIONS] \"query\"\n       badapple model <list|scan|info|use|verify|add|remove> [args]\n       badapple p2p <peers|sync|sync-doc <kind>|sync-personas|sync-prompt|sync-settings|sync-models|receive-mesh [timeout_ms]|models|pull <peer_id> <model_id>|send <peer_id> <model_id>|receive [peer_id model_id]>\n       badapple vault <get|set|remove|list|import> [args]\n       badapple workspace <get|set <path>|index|watch [path]>\n       badapple mcp <list|add <id> <command> [args...]|remove <id>|install <id>|uninstall <id>|start <id>|stop <id>|status <id>>\n       badapple redteam <run|watch|status|category <category>|probe <id>>\n       badapple cert");
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
        _ => None,
    }
}

fn run_p2p_subcommand(args: &[String]) -> Result<()> {
    if args.is_empty() {
        bail!("usage: badapple p2p <peers|sync|sync-doc <kind>|sync-personas|sync-prompt|sync-settings|sync-models|receive-mesh [timeout_ms]|models|pull <peer_id> <model_id>|send <peer_id> <model_id>|receive [peer_id model_id]>");
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
                bail!("usage: badapple p2p sync-doc <personas|prompt|settings|models>");
            }
            run_p2p_helper(&["sync-doc".to_string(), args[1].clone()])?;
        }
        "sync-personas"
        | "sync-persona"
        | "sync-prompt"
        | "sync-prompts"
        | "sync-settings"
        | "sync-models"
        | "sync-model-manifests" => {
            let kind = p2p_sync_kind_alias(sub).unwrap_or(sub);
            run_p2p_helper(&["sync-doc".to_string(), kind.to_string()])?;
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

fn print_help() {
    println!(
        "badapple — authenticated local client for the Bad Apple daemon\n\n\
         Usage:\n  badapple [OPTIONS] \"query\"\n  badapple model <list|scan|info|use|verify|add|remove|recommend> [args]\n  badapple p2p <peers|sync|sync-doc <kind>|sync-personas|sync-prompt|sync-settings|sync-models|receive-mesh [timeout_ms]|models|pull <peer_id> <model_id>|send <peer_id> <model_id>|receive [peer_id model_id]>\n  badapple vault <get|set|remove|list|import> [args]\n  badapple workspace <get|set <path>|index|watch [path]>\n  badapple mcp <list|add <id> <command> [args...]|remove <id>|install <id>|uninstall <id>|start <id>|stop <id>|status <id>|init>\n  badapple redteam <run|watch|status|category <category>|probe <id>>\n\n\
         Options:\n  -n, --max-tokens N  Maximum generated tokens (default: 500)\n  --speak             Stream each sentence to local TTS and play with afplay\n  --persona NAME      Switch persona for this query (cali, curious, drill, genz, midwest, wicket, ...)\n  --roast             Alias for --persona drill\n  --benchmark         Benchmark a single prompt or a default suite\n  --doctor            Print a local support diagnostic report (--diagnostics alias)\n  --crash-report      Collect crash logs and daemon state for debugging\n  --json              Output token stream as JSON\n  -h, --help          Show this help\n\n\
         Environment:\n  BADAPPLE_SOCKET_PATH       Unix socket path\n  BADAPPLE_SLICKS_KEY_PATH   SLICKS key file path\n  BADAPPLE_SLICKS_SECRET     In-memory SLICKS secret override\n  BADAPPLE_TTS_VOICE         Voice name for --speak (default: Best; Piper voices in voices/ take priority, then AVFoundation voices)\n  BADAPPLE_VAULT_KEY         Master key for the local secret vault\n  BADAPPLE_MCP_CATALOG_PATH  Path to the MCP marketplace catalog"
    );
}
