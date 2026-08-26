use anyhow::{bail, Context, Result};
use bad_apple::bad_apple_ipc::query_with_metrics;
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::mpsc::{channel, Sender};
use std::thread;
use std::time::Instant;

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
            "--doctor" => {
                doctor_mode = true;
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

    let prompt = if prompt_parts.is_empty() && !benchmark_mode {
        bail!("usage: badapple [OPTIONS] \"query\"");
    } else {
        prompt_parts.join(" ")
    };

    // Apply voice and persona sentinels.
    let wrap = |p: String| {
        let mut p = p;
        if voice_mode && !p.starts_with("__BADAPPLE_VOICE__ ") {
            p = format!("__BADAPPLE_VOICE__ {}", p);
        }
        if let Some(ref name) = persona {
            if !p.starts_with("__BADAPPLE_PERSONA__") {
                p = format!("__BADAPPLE_PERSONA__{}__ {}", name, p);
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

    // Give TTS a moment to finish before the program exits.
    if tts.is_some() {
        std::thread::sleep(std::time::Duration::from_millis(800));
    }
    Ok(())
}

struct TtsQueue {
    tx: Option<Sender<String>>,
    worker: Option<thread::JoinHandle<()>>,
}

impl TtsQueue {
    fn new() -> Self {
        let (tx, rx) = channel::<String>();
        let worker = thread::spawn(move || {
            while let Ok(text) = rx.recv() {
                speak_chunk(&text);
            }
        });
        Self {
            tx: Some(tx),
            worker: Some(worker),
        }
    }
    fn push(&self, text: &str) {
        if let Some(ref tx) = self.tx {
            let _ = tx.send(text.to_string());
        }
    }
}

impl Drop for TtsQueue {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
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
        println!("Total wall time: {:.2}s", total_elapsed);
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
        .or_else(|| {
            std::env::current_exe().ok().map(|mut p| {
                // bad_apple/target/release/badapple -> bad_apple
                for _ in 0..3 {
                    p.pop();
                }
                p.join(".venv")
            })
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
    let bin_dirs: Vec<std::path::PathBuf> = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .into_iter()
        .chain(std::iter::once(std::path::PathBuf::from("/usr/local/bin")))
        .collect();
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
    for label in [
        "com.badapple.mlx",
        "com.badapple.gatekeeper",
        "com.badapple.supervisor",
        "com.badapple.tts",
        "com.badapple.menubar",
    ] {
        if let Ok(out) = Command::new("launchctl").args(["list", label]).output() {
            let _ = writeln!(
                report,
                "{}: {}",
                label,
                String::from_utf8_lossy(&out.stdout).trim()
            );
        }
    }

    // Sockets
    let _ = writeln!(report, "\n[sockets]");
    for sock in [
        "/var/run/badapple/substrate_mlx.sock",
        "/var/run/badapple/mcp.sock",
        "/var/run/badapple/aqua_helper.sock",
        "/tmp/badapple_tts.sock",
    ] {
        let p = std::path::PathBuf::from(sock);
        let _ = writeln!(
            report,
            "{}: {}",
            sock,
            if p.is_file() { "present" } else { "missing" }
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
                let _ = writeln!(report, "  - {}", m);
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

    println!("{}", report);
    Ok(())
}

fn print_help() {
    println!(
        "badapple — authenticated local client for the Bad Apple daemon\n\n\
         Usage:\n  badapple [OPTIONS] \"query\"\n\n\
         Options:\n  -n, --max-tokens N  Maximum generated tokens (default: 240)\n  --speak             Stream each sentence to local TTS and play with afplay\n  --persona NAME      Switch persona for this query (wicket, drill, genz, midwest, ...)\n  --roast             Alias for --persona drill\n  --benchmark         Benchmark a single prompt or a default suite\n  --doctor            Print a local support diagnostic report\n  --json              Output token stream as JSON\n  -h, --help          Show this help\n\n\
         Environment:\n  BADAPPLE_SOCKET_PATH       Unix socket path\n  BADAPPLE_SLICKS_KEY_PATH   SLICKS key file path\n  BADAPPLE_SLICKS_SECRET     In-memory SLICKS secret override\n  BADAPPLE_TTS_VOICE         Voice name for --speak (default: en_US-amy-medium)"
    );
}
