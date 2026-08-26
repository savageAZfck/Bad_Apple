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
        if tts.is_some() {
            if !suffix.trim().is_empty() {
                tts.as_ref().unwrap().push(suffix);
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

fn print_help() {
    println!(
        "badapple — authenticated local client for the Bad Apple daemon\n\n\
         Usage:\n  badapple [OPTIONS] \"query\"\n\n\
         Options:\n  -n, --max-tokens N  Maximum generated tokens (default: 240)\n  --speak             Stream each sentence to local TTS and play with afplay\n  --persona NAME      Switch persona for this query (wicket, drill, genz, midwest, ...)\n  --roast             Alias for --persona drill\n  --benchmark         Benchmark a single prompt or a default suite\n  --json              Output token stream as JSON\n  -h, --help          Show this help\n\n\
         Environment:\n  BADAPPLE_SOCKET_PATH       Unix socket path\n  BADAPPLE_SLICKS_KEY_PATH   SLICKS key file path\n  BADAPPLE_SLICKS_SECRET     In-memory SLICKS secret override\n  BADAPPLE_TTS_VOICE         Voice name for --speak (default: en_US-amy-medium)"
    );
}
