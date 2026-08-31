use anyhow::{bail, Context, Result};
use bad_apple::bad_apple_ipc::{call_agent, query_with_metrics};
use serde_json::Value;
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut prompt_parts = Vec::new();
    let mut max_new_tokens = 240;
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

    let prompt = if prompt_parts.is_empty() && !benchmark_mode {
        bail!("usage: badapple [OPTIONS] \"query\"\n       badapple model <list|scan|info|use|verify|add|remove> [args]\n       badapple p2p <peers|sync|models|pull <peer_id> <model_id>|send <peer_id> <model_id>|receive [peer_id model_id]>");
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

    fn is_break_point(buf: &str) -> bool {
        if buf.len() >= MAX_CHUNK {
            return true;
        }
        if buf.len() < MIN_CHUNK {
            return false;
        }
        if buf.ends_with(['.', '!', '?', ':', ';', '\n']) {
            return true;
        }
        if buf.ends_with("—") || buf.ends_with("...") || buf.ends_with("…") {
            return true;
        }
        false
    }

    loop {
        let timeout = match deadline {
            Some(d) => d.saturating_duration_since(Instant::now()),
            None => Duration::from_millis(500),
        };

        match rx.recv_timeout(timeout) {
            Ok(TtsMsg::Text(text)) => {
                buffer.push_str(&text);
                if is_break_point(&buffer) {
                    speak_chunk(&buffer);
                    buffer.clear();
                    deadline = None;
                } else {
                    deadline = Some(Instant::now() + IDLE_TIMEOUT);
                }
            }
            Ok(TtsMsg::Flush) | Err(RecvTimeoutError::Disconnected) => {
                if !buffer.is_empty() {
                    speak_chunk(&buffer);
                }
                break;
            }
            Err(RecvTimeoutError::Timeout) => {
                if !buffer.is_empty() {
                    speak_chunk(&buffer);
                    buffer.clear();
                    deadline = None;
                }
            }
        }
    }
}

/// Send a chunk to the local Piper TTS server and play it with afplay.
fn speak_chunk(text: &str) {
    let voice =
        std::env::var("BADAPPLE_TTS_VOICE").unwrap_or_else(|_| "en_US-amy-medium".to_string());
    let socket = std::env::var("BADAPPLE_TTS_SOCKET")
        .unwrap_or_else(|_| "/tmp/badapple_tts.sock".to_string());
    let request = format!(
        "{{\"text\":{},\"voice\":{}}}\n",
        serde_json::to_string(text).unwrap_or_default(),
        serde_json::to_string(&voice).unwrap_or_default()
    );
    let mut stream = match UnixStream::connect(&socket) {
        Ok(s) => s,
        Err(_) => return,
    };
    if stream.write_all(request.as_bytes()).is_err() {
        return;
    }
    let mut response = String::new();
    if stream.read_to_string(&mut response).is_err() {
        return;
    }
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&response) {
        if let Some(wav) = json.get("wav_path").and_then(|v| v.as_str()) {
            let _ = std::process::Command::new("afplay").arg(wav).status();
        }
    }
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

    // Python / venv
    let _ = writeln!(report, "\n[python]");
    if let Ok(out) = Command::new("python3").args(["--version"]).output() {
        let _ = writeln!(report, "{}", String::from_utf8_lossy(&out.stdout).trim());
    }
    let venv: std::path::PathBuf = std::env::var("VIRTUAL_ENV")
        .map(std::path::PathBuf::from)
        .ok()
        .filter(|p| p.is_dir())
        .or_else(|| badapple_root().map(|r| r.join(".venv")))
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| std::path::PathBuf::from(h).join(".local/share/badapple/venv"))
        })
        .unwrap_or_default();
    let _ = writeln!(report, "venv: {}", venv.display());
    let _ = writeln!(
        report,
        "venv python ok: {}",
        venv.join("bin/python").is_file()
    );

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
    for sock in [
        "/var/run/badapple/substrate.sock",
        "/var/run/badapple/substrate_mlx.sock",
        "/var/run/badapple/identity.sock",
        "/var/run/badapple/mcp.sock",
        "/var/run/badapple/aqua_helper.sock",
        "/tmp/badapple_tts.sock",
    ] {
        let p = std::path::PathBuf::from(sock);
        let _ = writeln!(
            report,
            "{}: {}",
            sock,
            if std::fs::metadata(&p).is_ok() {
                "present"
            } else {
                "missing"
            }
        );
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

fn run_p2p_subcommand(args: &[String]) -> Result<()> {
    if args.is_empty() {
        bail!("usage: badapple p2p <peers|sync|models|pull <peer_id> <model_id>|send <peer_id> <model_id>|receive [peer_id model_id]>");
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

fn print_help() {
    println!(
        "badapple — authenticated local client for the Bad Apple daemon\n\n\
         Usage:\n  badapple [OPTIONS] \"query\"\n  badapple model <list|scan|info|use|verify|add|remove|recommend> [args]\n  badapple p2p <peers|sync|models|pull <peer_id> <model_id>|send <peer_id> <model_id>|receive [peer_id model_id]>\n\n\
         Options:\n  -n, --max-tokens N  Maximum generated tokens (default: 240)\n  --speak             Stream each sentence to local TTS and play with afplay\n  --persona NAME      Switch persona for this query (wicket, drill, genz, midwest, ...)\n  --roast             Alias for --persona drill\n  --benchmark         Benchmark a single prompt or a default suite\n  --doctor            Print a local support diagnostic report (--diagnostics alias)\n  --crash-report      Collect crash logs and daemon state for debugging\n  --json              Output token stream as JSON\n  -h, --help          Show this help\n\n\
         Environment:\n  BADAPPLE_SOCKET_PATH       Unix socket path\n  BADAPPLE_SLICKS_KEY_PATH   SLICKS key file path\n  BADAPPLE_SLICKS_SECRET     In-memory SLICKS secret override\n  BADAPPLE_TTS_VOICE         Voice name for --speak (default: en_US-amy-medium)"
    );
}
