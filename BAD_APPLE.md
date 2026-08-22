# Bad Apple — Local AI for macOS

Bad Apple is an on-device, air-gapped AI assistant for macOS. It runs Qwen 3.5 on Apple Silicon using MLX, answers questions, runs local tools, indexes your files, and speaks responses through a local neural TTS server — all without sending anything to the cloud after the models are downloaded once.

---

## Brains / Models

The runtime now uses a single Qwen 3.5 9B 4-bit brain for both text and voice. There is no separate voice bundle to swap in and out, which removes the old multi-second mode-switching lag.

| Component | Model | Size | Role |
|---|---|---|---|
| Target LLM | `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` | ~6.2 GB | All text and voice reasoning |
| DFlash draft | `z-lab/Qwen3.5-9B-DFlash` | ~2.4 GB | Speculative token blocks for the 9B target |
| RAG embeddings | `BAAI/bge-small-en-v1.5` | small | Local sentence-transformer on CPU |
| TTS voice | `en_US-amy-medium` (default) | small | Piper neural TTS server (`badapple_tts_server.py`) |

All models are downloaded and cached on the Mac. At runtime, **no prompt, response, or action leaves the machine**.

---

## Speed / Token Throughput

Live numbers from the daemon log on a 16 GB Apple Silicon M-series Mac with the single 9B brain loaded.

### Text mode

| Prompt | Tokens in | First token | Tokens out | Decode t/s | Draft acceptance | Peak memory |
|---|---|---:|---:|---:|---:|---:|
| `Who are you?` | ~620 | 5.7 s | 49 | 25.7 | 82% | 6.37 GB |
| `What is 2+2?` | ~580 | 5.4 s | 31 | 20.1 | 58% | 5.72 GB |
| `What is 7+7?` | ~700 | 5.4 s | 31 | 20.1 | 58% | 5.72 GB |
| `What is 8+8?` | ~650 | 4.4 s | 40 | 18.4 | 50% | 6.42 GB |
| `What is the capital of Germany?` | ~560 | 4.4 s | 35 | 15.8 | 63% | 6.29 GB |
| `Tell me about Rome.` | ~460 | 3.9 s | 51 | 13.8 | 55% | 5.79 GB |

- Typical first-token latency: **~3.5–6.5 s** for 450–750 token prompts.
- Typical decode throughput: **~13–25 tok/s**, with spikes to ~36 tok/s on high-acceptance turns.
- Peak memory stays **~5.7–6.5 GB**, leaving the rest of 16 GB free.

### Voice mode

| Prompt | Tokens in | First token | Tokens out | Decode t/s | Peak memory |
|---|---|---:|---:|---:|---:|
| `What is the capital of Spain?` | ~430 | 2.8 s | 27 | 10.5 | 5.78 GB |
| `Tell me about Rome.` | ~430 | 3.9 s | 51 | 13.8 | 5.79 GB |
| `Who are you?` | ~460 | 3.1 s | 16 | 13.1 | 6.22 GB |

- Voice first-token latency: **~2.8–5.7 s** for 430–460 token voice prompts.
- Voice decode: **~9–15 tok/s**.
- No separate voice model is loaded, so switching from text to voice is now a prompt change, not a model swap.

### Speculative decoding

DFlash uses a block-diffusion draft model to propose tokens in parallel and the target model to verify them in a single forward pass. This works better with Qwen 3.5's hybrid attention/GatedDeltaNet architecture than the native `mlx-lm` `draft_model` path.

- `BADAPPLE_DFLASH=1`
- `BADAPPLE_DFLASH_VERIFY_LEN_CAP=6`
- `BADAPPLE_DFLASH_BLOCK_TOKENS=6`
- `BADAPPLE_DFLASH_QUANTIZE_KV=1`

---

## Features

### Core

- **Local Qwen 3.5 inference** on Apple Silicon GPU (MLX)
- **Single 9B brain** for both text and voice, no dual-model swap
- **DFlash speculative decoding** for faster generation
- **Streaming token output** to terminal or TTS
- **SLICKS authenticated Unix socket** (HMAC-SHA256 challenge-response)
- **launchd-managed daemon** (`com.badapple.mlx`) that runs as root and auto-restarts

### Conversation & memory

- **Multi-turn conversation history** saved to local JSONL
- **Persistent user memory**: records user facts and recalls them in future turns
- **RAG / local document search**: indexes your text files with `BAAI/bge-small-en-v1.5` and retrieves relevant chunks
- **Hot-reloadable system prompt** via `prompt.txt` without restarting the 9B model
- **Persona packs** (`personas.json`): switch at runtime with `switch to <persona>`
- **Teachable quips** with `teach <line>`
- **Streaming output firewall** (Aho-Corasick blocklist) for PII, secrets, and custom patterns
- **Hash-chained audit ledger** (`/var/lib/bad_apple/ledger.jsonl`) with PII redaction
- **Semantic cache** (`BAAI/bge-small-en-v1.5`) for instant repeated-answer hits
- **Human-in-the-loop approvals** for destructive tools

### Tools (local, no cloud)

The server can run these directly, either through a fast deterministic parser or by a model-generated `tool_call` block:

- `get_current_time` — local system time
- `list_directory` — list files in a path
- `read_file` — read a file (with size limit)
- `write_file` — write a note to `~/.bad_apple/notes/`
- `search_content` — `grep -R` over a directory
- `search_local_files` — Spotlight search via `mdfind`
- `run_shell` — run a sandboxed shell command
- `run_applescript` — execute AppleScript on macOS
- `index_documents` — index a directory into the local RAG store

### Menu bar app

`Bad Apple.app` lives in the macOS status bar. Right-click the apple icon for:

- **New Chat** — clears conversation history
- **Chat History** — opens the transcript window
- **Voice Listening** — toggle always-on voice wake
- **Roast Mode** — alias for the `drill` persona on the next voice query
- **Persona** — switch between Default, Wicket, Gen Z, Drill, and Midwest Aunt
- **Benchmark** — runs the default prompt suite and shows results
- **Voice (Piper / Apple)** and **Accent** — TTS engine and voice selection
- **Quit**

Voice queries respect the selected persona and roast mode by passing `--persona <name>` / `--roast` to the bundled `badapple` CLI helper.

### Voice / TTS

- `--speak` streams each sentence to the local **Piper TTS server** and plays with `afplay`
- Voice can be changed via `BADAPPLE_TTS_VOICE` (default `en_US-amy-medium`)
- `BADAPPLE_TTS_LENGTH_SCALE`, `BADAPPLE_TTS_NOISE_SCALE`, etc. control voice speed and tone
- TTS server runs under `com.badapple.tts`

### Persona

Default persona is a sassy, flirty California beach girl. She:

- Uses English-only slang and endearments (`babe`, `hun`, `bestie`, `dude`, `stoked`, `chill`)
- Stays short: 1–2 punchy paragraphs
- **Roasts cloud AI and Siri** when bragging about bare metal or when asked directly. She varies the target (Siri, Alexa, Google, ChatGPT, Gemini, Cortana, Bixby, "the cloud", server farms, data centers, "some rented GPU in Nevada") and the insult ("ratchet old bitch", "washed-up cloud snitch", "data-hungry narc", "internet junkie", "corporate eavesdropper", etc.)

Built-in persona packs (in `personas.json`) include Wicket (witty Londoner), Gen Z Hype, Drill Rapper, and Midwest Aunt.  Switch at runtime with `switch to <persona>` or set `BADAPPLE_PERSONA=<name>`.

---

## Security & governance

| Layer | Mechanism | Where it lives |
|---|---|---|
| Output firewall | Streaming Aho-Corasick blocklist on generated text | `badapple_extras.py` |
| Audit | Append-only SHA-256–chained JSONL with secret/PII redaction | `/var/lib/bad_apple/ledger.jsonl` |
| Approvals | Proposal/approve workflow for `run_shell`, `run_applescript`, `write_file`, `index_documents` | `badapple_extras.py` + `badapple_mlx_server.py` |
| Cache | Persona-scoped semantic cache; no cache for tool queries or voice | `badapple_extras.py` |

---

## `badapple` CLI

`target/release/badapple` is the authenticated Rust client.

```bash
badapple --max-tokens 240 "What is the capital of France?"
badapple --roast "What do you think of Siri?"            # --persona drill
badapple --persona wicket "Who are you?"
badapple --speak "Tell me a joke."                        # stream to TTS queue
badapple --benchmark                                       # or: badapple --benchmark "prompt"
badapple --json "Explain recursion."                       # token stream as JSON
```

New flags:
- `--benchmark` — runs the default prompt suite and prints a table with TTFT, decode tok/s, total tok/s, and peak memory. Pass a prompt to benchmark a single query.
- `--roast` / `--persona <name>` — injects the `__BADAPPLE_PERSONA__<name>__` sentinel so the server switches persona for that query.
- `--speak` now queues chunks to a background TTS worker so the model is not blocked waiting for audio playback; audio plays sequentially without overlapping.
- `--json` streams each token and the final `done` frame as JSON, including `metrics`.

---

## Architecture

```text
┌───────────────────────────────────────┐
│  badapple CLI  │  Bad Apple menu bar  │
│  target/release/badapple              │
│  (Rust client)                        │
└───────────┬───────────────────────────┘
            │ Unix socket + SLICKS auth
            ▼
┌───────────────────────────────────────┐
│  com.badapple.gatekeeper              │
│  target/release/gatekeeper            │
│  - SLICKS proxy, fast actions,        │
│    automation cage                    │
└───────────┬───────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  com.badapple.mlx                     │
│  badapple_mlx_server.py               │
│  - loads single 9B target + DFlash    │
│  - runs tools, memory, RAG, TTS       │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  com.badapple.tts                     │
│  badapple_tts_server.py (Piper)       │
└───────────────────────────────────────┘
```

- **Client socket (via gatekeeper)**: `/var/run/badapple/substrate.sock`
- **MLX socket**: `/var/run/badapple/substrate_mlx.sock`
- **Logs**: `/var/log/bad_apple_mlx_server.log`
- **Conversation & memory**: `~/.bad_apple/`
- **RAG index**: `/var/lib/bad_apple/knowledge`

---

## Security & Privacy

- **Air-gapped at runtime**: no network calls for inference or actions
- **Authenticated**: every client proves itself with SLICKS HMAC-SHA256
- **Fail-closed tools**: file writes are restricted to `BADAPPLE_NOTES_DIR`; shell/AppleScript calls are gated
- **Local-only audio**: TTS happens on-device

---

## How to use

```bash
# Build
cargo build --release

# Text query
target/release/badapple "What time is it?"

# Voice query (text-to-speech)
target/release/badapple --speak "What time is it?"

# Force voice mode with env var
BADAPPLE_VOICE=1 target/release/badapple "What time is it?"

# Set max tokens
target/release/badapple -n 240 "Write me a poem about bare metal"
```

---

## Tuning / Environment Variables

| Variable | Default / Current | Purpose |
|---|---|---|
| `BADAPPLE_MAIN_MODEL` | `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` | Target model for all modes |
| `BADAPPLE_DRAFT_MODEL` | `z-lab/Qwen3.5-9B-DFlash` | DFlash draft for the 9B target |
| `BADAPPLE_DFLASH` | `1` | Enable DFlash speculative decoding |
| `BADAPPLE_DFLASH_VERIFY_LEN_CAP` | `6` | Tokens to verify per target forward |
| `BADAPPLE_DFLASH_BLOCK_TOKENS` | `6` | Draft block size per step |
| `BADAPPLE_DFLASH_QUANTIZE_KV` | `1` | Quantize key/value cache |
| `BADAPPLE_PROMPT_FILE` | `prompt.txt` | Hot-reloadable system prompt |
| `BADAPPLE_TTS_VOICE` | `en_US-amy-medium` | Default TTS voice |

---

## Caveats

- **DFlash is deterministic for the same prompt**: identical questions get identical answers. A per-query random seed is set, but the primary source of variance is different phrasing.
- **Throughput is workload-dependent**: DFlash acceptance swings from ~45% to ~80%, so tok/s swings with it. Sustained 22–33 tok/s is possible on high-acceptance turns but not guaranteed for every prompt on this hardware.
- **First-token latency is dominated by prefill**: long prompts or large knowledge chunks push the first token toward the 5–7 s range.
- **Max-token cutoffs are cleaned up**: if the model runs out of output tokens mid-sentence, the response is trimmed to the last complete sentence so it doesn't end on a dangling word.
- The older `automation_cage`, `wasm_cage`, and 576-D `tensor_brain` classifier still exist in the repo but are **not the active path**. The active stack is the gatekeeper + MLX server + CLI + TTS server + menu bar.
