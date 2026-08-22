use anyhow::{bail, Context, Result};
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut prompt_parts = Vec::new();
    let mut max_new_tokens = 120;
    let mut speak_stream = std::env::var("BADAPPLE_SPEAK").is_ok();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            "--speak" => {
                speak_stream = true;
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

    if prompt_parts.is_empty() {
        bail!("usage: badapple [--max-tokens N] \"query\"");
    }

    let prompt = prompt_parts.join(" ");
    let json_stream = std::env::var("BADAPPLE_STREAM_JSON").is_ok();
    let voice_mode = std::env::var("BADAPPLE_VOICE").is_ok() || speak_stream;
    let prompt = if voice_mode && !prompt.starts_with("__BADAPPLE_VOICE__ ") {
        format!("__BADAPPLE_VOICE__ {}", prompt)
    } else {
        prompt
    };
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let mut emitted = String::new();
    let final_text = bad_apple::bad_apple_ipc::stream_query(&prompt, max_new_tokens, |token| {
        emitted.push_str(token);
        if json_stream {
            let line = format!("{{\"type\":\"token\",\"text\":{}}}\n", serde_json::to_string(token).unwrap_or_default());
            let _ = stdout.write_all(line.as_bytes());
        } else if speak_stream {
            let _ = speak_chunk(token);
        } else {
            let _ = stdout.write_all(token.as_bytes());
        }
        let _ = stdout.flush();
    })?;

    if json_stream {
        let line = format!("{{\"type\":\"done\",\"text\":{}}}\n", serde_json::to_string(&final_text).unwrap_or_default());
        stdout.write_all(line.as_bytes())?;
    } else if let Some(suffix) = final_text.strip_prefix(&emitted) {
        stdout.write_all(suffix.as_bytes())?;
    }
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

/// Send a chunk to the local Piper TTS server and play it with afplay.
fn speak_chunk(text: &str) {
    let voice = std::env::var("BADAPPLE_TTS_VOICE").unwrap_or_else(|_| "en_US-amy-medium".to_string());
    let socket = std::env::var("BADAPPLE_TTS_SOCKET").unwrap_or_else(|_| "/tmp/badapple_tts.sock".to_string());
    let request = format!("{{\"text\":{},\"voice\":{}}}\n", serde_json::to_string(text).unwrap_or_default(), serde_json::to_string(&voice).unwrap_or_default());
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


fn print_help() {
    println!(
        "badapple — authenticated local client for the Bad Apple daemon\n\n\
         Usage:\n  badapple [OPTIONS] \"query\"\n\n\
         Options:\n  -n, --max-tokens N  Maximum generated tokens (default: 256)\n  --speak             Stream each sentence to local TTS and play with afplay\n  -h, --help          Show this help\n\n\
         Environment:\n  BADAPPLE_SOCKET_PATH       Unix socket path\n  BADAPPLE_SLICKS_KEY_PATH   SLICKS key file path\n  BADAPPLE_SLICKS_SECRET     In-memory SLICKS secret override\n  BADAPPLE_TTS_VOICE         Voice name for --speak (default: en_US-lessac-high)"
    );
}
