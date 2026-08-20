use anyhow::{bail, Context, Result};
use std::io::{self, Write};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut prompt_parts = Vec::new();
    let mut max_new_tokens = 64;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(());
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
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let mut emitted = String::new();
    let final_text = bad_apple::bad_apple_ipc::stream_query(&prompt, max_new_tokens, |token| {
        emitted.push_str(token);
        let _ = stdout.write_all(token.as_bytes());
        let _ = stdout.flush();
    })?;

    if let Some(suffix) = final_text.strip_prefix(&emitted) {
        stdout.write_all(suffix.as_bytes())?;
    }
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn print_help() {
    println!(
        "badapple — authenticated local client for the Bad Apple daemon\n\n\
         Usage:\n  badapple [OPTIONS] \"query\"\n\n\
         Options:\n  -n, --max-tokens N  Maximum generated tokens (default: 256)\n  -h, --help          Show this help\n\n\
         Environment:\n  BADAPPLE_SOCKET_PATH       Unix socket path\n  BADAPPLE_SLICKS_KEY_PATH   SLICKS key file path\n  BADAPPLE_SLICKS_SECRET     In-memory SLICKS secret override"
    );
}
