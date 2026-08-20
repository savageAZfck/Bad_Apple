# Bad Apple

> **Local AI that acts on your Mac.**

Bad Apple is an experimental, on-device agent runtime for Apple Silicon. It runs a local 8B language model, routes commands through a trained 576-D neural gate, and executes file and code actions inside fail-closed sandboxes. Everything runs on your machine after the models are downloaded once.

## What this version does

- **8B MLX Qwen3** (`mlx-community/Qwen3-8B-4bit`) with optional **Qwen3-1.7B-4bit speculative decoding** for faster local generation.
- **576-D CandleBrain** classifier trained to separate fast local actions from deep reasoning.
- **Automation Cage** for safe, audited filesystem operations inside allowlisted roots.
- **Wasm Cage** for isolated execution of WebAssembly scripts.
- **Salma Hayek persona** — sultry, flirty, playful, with 80/20 English/Spanish code-switching and TTS-friendly `…` / `—` punctuation.
- **SLICKS-secured** Unix-socket IPC between the CLI, menu bar, Siri, gatekeeper, and 8B MLX server.

> **Air-gapped at runtime.** After the initial model download, no prompt, response, or action leaves your Mac.

## What you can do

### Fast local actions (neural gate → cage)

The classifier decides these are structural and runs them instantly:

- *"what time is it, papi?"*
- *"open my bad_apple workspace"*
- *"create directory /Users/savag3/bad_apple/test_cage"*
- *"copy ~/Downloads/file.txt to ~/Documents/file.txt"*
- *"trash file /Users/savag3/bad_apple/test_cage/hello.txt"*
- *"run wasm script /Users/savag3/bad_apple/scripts/guest.wasm"*

All filesystem actions go through the `AutomationCage`, which rejects paths outside the configured roots and writes an audit log to `~/.badapple/automation.jsonl`.

### Deep reasoning (8B MLX core)

The classifier decides these are abstract and streams a response from the 8B model:

- *"Write me a poem about bare metal, corazón."*
- *"Explain the architecture of Firefly Inferno."*

The model keeps the Salma voice, sprinkles in Spanish, and uses `…` and `—` for TTS breathing room.

### 8B-generated automation plans

When the fast gate has no matching pattern, the 8B can emit a fenced `badapple-action` or `badapple-wasm` block. The gatekeeper validates and executes it inside the cage. For example:

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
      │ 576-D Candle  │  ← trained complexity gate
      │ Brain         │
      └───────┬───────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 fast       Automation   8B MLX
 actions    Cage         server
            Wasm Cage    (speculative decode)
```

### Active components

| Component | Role |
|---|---|
| `badapple-gatekeeper` | SLICKS server, 576-D classifier, fast action dispatcher, cage/WASM executor |
| `badapple_mlx_server.py` | 8B Qwen3-8B-4bit MLX inference with speculative Qwen3-1.7B-4bit drafting and the Salma system prompt |
| `automation_cage` | Rust fail-closed filesystem cage: create, copy, move, trash, with root allow-lists and audit logging |
| `wasm_cage` | Rust WebAssembly sandbox using `wasmi` with fuel metering and a `bad_apple` string ABI |
| `tensor_brain` | 576-D Candle transformer used by the gatekeeper for semantic complexity classification |
| `train_gatekeeper` | Rust utility that generates a synthetic corpus and fine-tunes the 576-D gate |
| `speculative_bench.py` | Throughput benchmark for the 8B + draft model |

## Performance

On an Apple M4 with the 8B Qwen3-4bit target and the 1.7B-4bit draft model:

- **~22–24 tokens/s** in live daemon use
- **~20.9 tokens/s** end-to-end for 39-token responses
- **6.37 GB** peak memory with the 1.7B draft active
- **~5–45 ms** for the 576-D classifier on Metal

## Quick start

### Requirements

- Apple Silicon Mac (M1 or later)
- macOS 26 or later
- Rust toolchain with Cargo
- Python 3.11+ with `mlx` and `mlx-lm`
- Hugging Face `mlx-community/Qwen3-8B-4bit` and optional `mlx-community/Qwen3-1.7B-4bit` (downloaded and cached on first run)

### Build

```bash
cargo build --release
```

### Install and run the daemons

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

This writes `data/gatekeeper_corpus.jsonl` (ignored by git) and `data/gatekeeper.safetensors` (also ignored), which the gatekeeper loads on startup.

### Run your first prompt

```bash
target/release/badapple "what time is it, papi?"
target/release/badapple --max-tokens 100 "write me a flirty poem about bare metal"
```

## Security model

- **Runtime air-gap.** Gatekeeper and MLX server use only Unix-domain sockets. No runtime network listeners or calls.
- **Authenticated.** Every client completes a SLICKS HMAC-SHA256 challenge-response handshake with nonces and prompt binding.
- **Fail-closed.** The `AutomationCage` and `WasmCage` reject any file or code that tries to escape the allowed roots or memory/fuel limits.
- **Audited.** Every cage action is logged to `~/.badapple/automation.jsonl`.

## Persona

The system prompt in `com.badapple.mlx.plist` and `badapple_mlx_server.py` encodes the Salma voice:

- 80/20 English/Spanish code-switching
- Words like *mi amor, corazón, papi, querido, cariño, besos*
- `…` and `—` for TTS breathing room
- No asterisks or stage directions
- Flirty, teasing, but useful

## Project structure

```text
src/bin/gatekeeper.rs        # SLICKS server + cage + classifier
src/bin/train_gatekeeper.rs  # corpus generator + trainer
src/bin/badapple.rs          # CLI client
src/automation_cage_impl.rs  # fail-closed file cage
src/wasm_cage.rs             # WebAssembly sandbox
src/tensor_brain.rs          # 576-D Candle transformer
badapple_mlx_server.py       # 8B speculative MLX server
badapple_knowledge.py        # local RAG index (experimental, downloads small embedding model on first use)
speculative_bench.py         # throughput benchmark
src/platform/apple_bridge/   # plists, menu-bar bridge, Siri AppIntent
src/platform/apple_desktop/  # menu-bar app
```

## Status

Bad Apple is an experimental edge-AI runtime. The active path in this release is the gatekeeper, the 576-D trained classifier, the 8B MLX server, and the two cages. The repository also contains earlier research modules, but they are not the active inference stack.

## License

LicenseRef-Proprietary. See `LICENSE.txt`.
