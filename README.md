# Bad Apple

> **Local AI that runs on your Mac — no cloud after the first download.**

Bad Apple is an on-device, air-gapped AI assistant for macOS. It runs a local Qwen 3.5 9B model on Apple Silicon using [MLX](https://github.com/ml-explore/mlx), with a smaller DFlash speculative draft for faster token generation. It answers questions, runs local tools, indexes your files, switches personas, and speaks back through a local Piper TTS server.

After the models are downloaded once, **no prompt, response, or action leaves your machine**.

---

## What it does

- **Local 9B reasoning** — `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` runs on your GPU.
- **DFlash speculative decoding** — `z-lab/Qwen3.5-9B-DFlash` blocks speed up generation.
- **Streaming output** — tokens stream to the terminal, TTS, or the menu bar as they are generated.
- **Local RAG** — indexes your text files with `bge-small-en-v1.5` and retrieves relevant chunks.
- **Persona packs** — `personas.json` with Default, Wicket, Gen Z, Drill, Midwest Aunt; switch at runtime.
- **Teachable quips** — `teach The cloud is just hamsters on a wheel` and the persona remembers.
- **Streaming output firewall** — Aho-Corasick blocklist for secrets, PII, and custom patterns.
- **Hash-chained audit ledger** — every query, tool, and response is logged with SHA-256 chaining.
- **Human-in-the-loop approvals** — destructive tools (`run_shell`, `run_applescript`, `write_file`, `index_documents`) require approval.
- **Semantic cache** — common answers are cached by embedding similarity and served instantly.
- **Local TTS** — Piper neural TTS via `badapple_tts_server.py`; plays through `afplay`.
- **Menu bar app** — `Bad Apple.app` sits in the status bar, listens for voice, switches persona/roast, and runs benchmarks.
- **SLICKS-secured IPC** — every client completes an HMAC-SHA256 challenge-response over a Unix socket.

---

## Quick start

### Requirements

- Apple Silicon Mac (M1 or later)
- macOS 26 or later
- Rust toolchain with Cargo
- Python 3.12 with `mlx`, `mlx-lm`, and the packages in `requirements.txt`
- Models are downloaded on first run:
  - `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit`
  - `z-lab/Qwen3.5-9B-DFlash`
  - `BAAI/bge-small-en-v1.5` (for RAG/cache)

### Build

```bash
cargo build --release
```

### Install and start the daemons

```bash
sudo cp src/platform/apple_bridge/com.badapple.gatekeeper.plist /Library/LaunchDaemons/
sudo cp src/platform/apple_bridge/com.badapple.mlx.plist /Library/LaunchDaemons/
sudo cp src/platform/apple_bridge/com.badapple.tts.plist /Library/LaunchDaemons/
sudo launchctl load -w /Library/LaunchDaemons/com.badapple.gatekeeper.plist
sudo launchctl load -w /Library/LaunchDaemons/com.badapple.mlx.plist
sudo launchctl load -w /Library/LaunchDaemons/com.badapple.tts.plist
```

Wait ~45 s for the 9B model and embedding model to load. Check the log:

```bash
tail -n 20 /var/log/bad_apple_mlx_server.log
```

### Run your first prompt

```bash
target/release/badapple "What time is it?"
target/release/badapple -n 240 "Write me a poem about bare metal"
```

### Voice / TTS

```bash
target/release/badapple --speak "What do you think of Siri?"
```

### Personas and roast

```bash
target/release/badapple --persona wicket "Who are you?"
target/release/badapple --roast "What do you think of Siri?"
target/release/badapple "switch to midwest"
```

### Benchmark

```bash
target/release/badapple --benchmark
```

### Menu bar

Build and install the status-bar host:

```bash
src/platform/apple_desktop/build_bad_apple_menu_bar.sh
cp -R target/release/Bad\ Apple.app /Applications/
open -a "Bad Apple"
```

---

## Architecture

```text
badapple CLI / menu bar / Siri
              │
              ▼
  /var/run/badapple/substrate.sock
              │
              ▼
     gatekeeper (SLICKS proxy, fast actions)
              │
              ▼
   badapple_mlx_server.py (9B Qwen3.5 + DFlash + tools + RAG)
              │
              ▼
    badapple_tts_server.py (Piper TTS)
```

- The **gatekeeper** is the authenticated front door. It handles SLICKS, fast local actions, and proxies generation to the MLX server.
- The **MLX server** loads the 9B target + DFlash draft once and keeps them hot. It handles conversation, memory, tools, RAG, approvals, cache, firewall, and audit.
- The **TTS server** runs Piper on a Unix socket and returns WAV paths.
- The **menu bar app** is a Swift/Objective-C status-bar host that listens for voice, sends prompts through the bundled `badapple` helper, and speaks responses.

---

## Tools

All tools are local:

- `get_current_time`
- `list_directory`
- `read_file`
- `write_file` — writes to `~/.bad_apple/notes/`
- `search_content` — `grep -R`
- `search_local_files` — Spotlight via `mdfind`
- `run_shell` — proposed, then executed after approval
- `run_applescript` — proposed, then executed after approval
- `index_documents` — indexes a directory into the RAG store

---

## Security & privacy

- **Air-gapped at runtime** — no network calls for inference, actions, or TTS.
- **Authenticated** — every client completes a SLICKS HMAC-SHA256 challenge-response with prompt binding.
- **Fail-closed tools** — shell/AppleScript and file writes require explicit user approval unless `BADAPPLE_AUTOPILOT=1` is set.
- **Streaming firewall** — secrets and PII patterns are blocked or redacted in generated text.
- **Audit ledger** — append-only, hash-chained JSONL at `/var/lib/bad_apple/ledger.jsonl`.
- **Local-only audio** — TTS is synthesized on-device.

---

## Project structure

```text
badapple_mlx_server.py          # 9B MLX inference daemon
badapple_extras.py              # personas, firewall, audit, cache, approvals
badapple_tts_server.py          # Piper TTS daemon
personas.json                   # persona packs
prompt.txt                      # hot-reloadable system prompt
src/bin/badapple.rs             # Rust CLI client
src/bin/gatekeeper.rs           # SLICKS proxy + fast action gate
src/bad_apple_ipc.rs            # SLICKS protocol
src/platform/apple_bridge/      # launchd plists, Siri bridge
src/platform/apple_desktop/     # menu bar app source
BAD_APPLE.md                    # technical deep-dive
```

---

## Performance

Measured on a 16 GB Apple Silicon M-series Mac:

- **First token**: ~3.5–6.5 s for 450–750 token prompts.
- **Decode throughput**: ~13–25 tok/s, with spikes to ~36 tok/s on high-acceptance turns.
- **Peak memory**: ~5.7–6.5 GB with the 9B model + DFlash draft loaded.
- **RAG embeddings**: `bge-small-en-v1.5` on CPU.

---

## License

LicenseRef-Proprietary. See `LICENSE.txt`.
