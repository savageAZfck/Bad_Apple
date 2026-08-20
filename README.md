# Bad Apple

> **Your Mac. Your voice. Your mind.**  
> A local, air-gapped AI runtime that thinks, acts, and stays on your machine.

Bad Apple is an experimental **on-device cognitive operating system** for Apple Silicon. It combines an 8-billion-parameter local language model, a trained 576-D neural complexity gate, fail-closed automation sandboxes, and a flirty, code-switching voice persona — all running air-gapped on your Mac. No cloud, no subscription, no telemetry.

## What makes this version different

This release is a major step from "chatbot" to **local AI OS**.

- **8B MLX Qwen3 core** with **speculative decoding** (Qwen3-1.7B-4bit draft model) for faster local token generation.
- **Trained 576-D CandleBrain** acts as a native neural gate: it learns to separate quick local actions from deep reasoning and routes each request to the right subsystem.
- **Automation Cage** gives the 8B and voice pipeline safe, audited, allow-listed filesystem access.
- **Wasm Cage** lets the system load and execute untrusted WebAssembly scripts with fuel-metered limits.
- **Salma Hayek persona** — sultry, flirty, playful — with 80/20 English/Spanish code-switching and TTS-friendly `…` / `—` punctuation.
- **SLICKS-secured Unix-socket IPC** between the CLI, menu-bar app, Siri shortcut, gatekeeper, and 8B MLX server.

## What you can do

### Fast local actions (576-D gate → cage)

The brain decides these are structural and runs them instantly, no 8B wake-up:

- *"what time is it, papi?"*
- *"open my bad_apple workspace"*
- *"create directory /Users/savag3/bad_apple/test_cage"*
- *"copy ~/Downloads/file.txt to ~/Documents/file.txt"*
- *"trash file /Users/savag3/bad_apple/test_cage/hello.txt"*
- *"run wasm script /Users/savag3/bad_apple/scripts/guest.wasm"*

All filesystem operations go through the `AutomationCage`, which rejects any path outside the configured roots and writes every action to `~/.badapple/automation.jsonl`.

### Deep reasoning (8B MLX core)

The brain decides these are abstract and streams them from the 8B Qwen3 model:

- *"Write me a poem about bare metal, corazón."*
- *"Explain the architecture of Firefly Inferno."*

The model keeps the Salma voice, sprinkles Spanish, and uses `…` and `—` for natural TTS pacing.

### 8B-generated automation plans

When the fast gate does not have a matching pattern, the 8B can emit a fenced `badapple-action` or `badapple-wasm` block. The gatekeeper validates and executes it inside the cage:

```text
Write a badapple-action JSON block to create a directory at /Users/savag3/bad_apple/test_cage/deep_dir
→ Created directory in /Users/savag3/bad_apple/test_cage/deep_dir (success)
```

## Architecture

```text
badapple CLI / Siri / menu bar
              │
              ▼
    SLICKS HMAC-SHA256 over
    /var/run/badapple/substrate.sock
              │
              ▼
      badapple-gatekeeper (launchd)
              │
      ┌───────┴───────┐
      │ 576-D Candle  │  ← complexity gate, trained
      │ Brain         │    fast vs. deep
      └───────┬───────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 fast       Automation   8B MLX
 actions    Cage         core
            Wasm Cage    (speculative decode)
```

### Components

| Component | Role |
|---|---|
| `badapple-gatekeeper` | SLICKS server, 576-D classifier, fast action dispatcher, cage/WASM executor |
| `badapple_mlx_server.py` | 8B Qwen3 MLX inference with speculative drafting and Salma system prompt |
| `automation_cage` | Rust fail-closed filesystem cage: create, copy, move, trash, with root allow-lists and audit logging |
| `wasm_cage` | Rust WebAssembly sandbox using `wasmi` with fuel metering and a `bad_apple` string ABI |
| `tensor_brain` | 576-D Candle transformer used by the gatekeeper for semantic complexity classification |
| `train_gatekeeper` | Rust utility that generates a synthetic corpus and fine-tunes the 576-D gate |
| `speculative_bench.py` | Formal throughput benchmark for the 8B + draft model |

## Performance

With the 8B Qwen3 target and Qwen3-1.7B-4bit speculative draft model on an Apple M4:

- **~22–24 tokens/s** in live daemon use.
- **~20.9 tokens/s** end-to-end for 39-token responses.
- **6.37 GB** peak memory with the 1.7B draft active.

The 576-D gate runs in **~5–45 ms** on Metal, depending on prompt length.

## Quick start

### Build

Requirements:

- Apple Silicon Mac (M1 or later)
- macOS 26 or later
- Rust toolchain with Cargo
- Python 3.11+ with `mlx` and `mlx-lm`
- Hugging Face `Qwen3-8B` (4-bit) and `Qwen3-1.7B-4bit` draft model

```bash
cargo build --release
```

### Install the daemons

```bash
sudo cp src/platform/apple_bridge/com.badapple.gatekeeper.plist /Library/LaunchDaemons/
sudo cp src/platform/apple_bridge/com.badapple.mlx.plist /Library/LaunchDaemons/
sudo launchctl load -w /Library/LaunchDaemons/com.badapple.gatekeeper.plist
sudo launchctl load -w /Library/LaunchDaemons/com.badapple.mlx.plist
```

### Train the 576-D gate

```bash
cargo run --release --bin train_gatekeeper
```

This writes `data/gatekeeper.safetensors` and `data/gatekeeper_corpus.jsonl`.

### Run your first prompt

```bash
badapple "what time is it, papi?"
badapple --max-tokens 100 "write me a flirty poem about bare metal"
```

## Security model

- **Air-gapped.** Gatekeeper and MLX server use only Unix-domain sockets. The default LaunchDaemon has no network listeners.
- **Authenticated.** Every client completes a SLICKS HMAC-SHA256 challenge-response handshake with nonces and prompt binding.
- **Fail-closed.** The `AutomationCage` and `WasmCage` reject any file or code that tries to escape the allowed roots or memory/fuel limits.
- **Audited.** Every cage action is logged to `~/.badapple/automation.jsonl`.

## Persona

The system prompt in `com.badapple.mlx.plist` and `badapple_mlx_server.py` encodes the Salma Hayek voice:

- 80/20 English/Spanish code-switching
- Words like *mi amor, corazón, papi, querido, cariño, besos*
- `…` and `—` for TTS breathing room
- No asterisks or stage directions
- Flirty but sharp and useful

## Utilities

- `badapple` — terminal client
- `train_gatekeeper` — retrain the 576-D complexity gate
- `speculative_bench.py` — benchmark 8B speculative decode throughput
- `diag_brain` — diagnostic tool for the 576-D brain
- `bench_tensor_brain` — brain forward latency benchmark

## Project structure

```text
src/bin/gatekeeper.rs        # SLICKS server + cage + classifier
src/bin/train_gatekeeper.rs  # corpus generator + trainer
src/bin/badapple.rs          # CLI client
src/main.rs                  # badappled main daemon
src/automation_cage_impl.rs  # fail-closed file cage
src/wasm_cage.rs             # WebAssembly sandbox
src/tensor_brain.rs          # 576-D Candle transformer
badapple_mlx_server.py       # 8B speculative MLX server
speculative_bench.py         # throughput benchmark
src/platform/apple_bridge/   # menubar, Siri, launchd plists
```

## Status

Bad Apple is an experimental edge-AI runtime. The 8B MLX path, 576-D gate, and both sandboxes are live and wired. The older ANE/CoreML path remains in the codebase as a research substrate but is not the active inference engine in this release.

## License

LicenseRef-Proprietary. See `LICENSE.txt`.
