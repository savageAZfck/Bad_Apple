# AGENTS.md — Bad Apple

This file captures the project-specific commands and conventions learned while working on Bad Apple, so the next agent (or future you) does not have to rediscover them.

## Project layout

- `badapple_mlx_server.py` — MLX daemon (Python). Loads the 9B Qwen 3.5 model, DFlash draft, and serves the SLICKS Unix socket.
- `src/bin/badapple.rs` + `src/` — Rust CLI client that talks to the daemon.
- `src/platform/apple_bridge/com.badapple.mlx.plist` — launchd daemon config.
- `prompt.txt` — Hot-reloadable system prompt. Edits take effect on the next query without restarting the model.
- `BAD_APPLE.md` — Technical overview and live performance numbers.
- `BAD_APPLE_BUYERS.md` — Buyer-facing pitch doc.

## Build

```bash
cargo build --release
```

## Start / restart the daemon

```bash
osascript -e 'do shell script "cp /Users/savag3/bad_apple/src/platform/apple_bridge/com.badapple.mlx.plist /Library/LaunchDaemons/ && launchctl unload /Library/LaunchDaemons/com.badapple.mlx.plist 2>/dev/null; launchctl load -w /Library/LaunchDaemons/com.badapple.mlx.plist" with administrator privileges'
```

Wait ~45 s for the model bundle and embedding model to load. Check the tail of the log:

```bash
tail -n 20 /var/log/bad_apple_mlx_server.log
```

## Test a query

```bash
# text
target/release/badapple "What is 2+2?"

# voice (text output only, no audio)
BADAPPLE_VOICE=1 target/release/badapple "What do you think of Siri?"

# voice with TTS (Piper server must be running)
target/release/badapple --speak "What do you think of Siri?"

# set max tokens
target/release/badapple -n 120 "Write me a poem about bare metal"
```

## Edit the persona

Edit `prompt.txt`. The daemon hot-reloads it on the next query. Voice mode uses `VOICE_SYSTEM_PROMPT` inside `badapple_mlx_server.py`, which requires a daemon restart to change.

## Check performance

```bash
# live log
tail -n 20 /var/log/bad_apple_mlx_server.log

# memory pressure on macOS
memory_pressure
```

Look for log lines like:

```
[perf] 31 tokens @ 20.1 decode t/s (5.4 total t/s), draft_accept_ratio=58%, block_tokens=6, peak_memory=5.72 GB
```

## Python venv

For one-off MLX/DFlash debugging:

```bash
source .venv/bin/activate
```

## Current DFlash settings

Set in `src/platform/apple_bridge/com.badapple.mlx.plist`:

- `BADAPPLE_DFLASH=1`
- `BADAPPLE_DFLASH_VERIFY_LEN_CAP=6`
- `BADAPPLE_DFLASH_BLOCK_TOKENS=6`
- `BADAPPLE_DFLASH_QUANTIZE_KV=1`

## Common gotchas

- `BAD_APPLE.md` is the live technical doc; keep it in sync with the architecture.
- `BAD_APPLE_BUYERS.md` is the buyer-facing doc; commit it when the numbers change.
- DFlash is deterministic for the same prompt. To get variance, vary the seed or the prompt phrasing.
- The 4B voice bundle has been removed in favor of the unified 9B brain; do not reintroduce it.
