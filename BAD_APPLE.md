# Bad Apple — Local AI for macOS

Bad Apple is an on-device, air-gapped AI assistant for macOS. It runs Qwen 3.5 on Apple Silicon using MLX, answers questions, runs local tools, indexes your files, and speaks responses through a local neural TTS server — all without sending anything to the cloud after the models are downloaded once.

---

## Brains / Models

The runtime now uses a single Qwen 3.5 9B 4-bit brain for both text and voice. There is no separate voice bundle to swap in and out, which removes the old multi-second mode-switching lag.

| Component | Model | Size | Role |
|---|---|---|---|
| Target LLM | `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` | ~6.2 GB | All text and voice reasoning |
| Fast tier / tiny brain | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | ~0.3 GB | Instant answers for greetings, identity, time, simple math, and deterministic queries |
| MLX-LM speculative draft (optional) | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | ~0.3 GB | Optional small draft for the 9B brain; set `BADAPPLE_SPECULATIVE_DRAFT=auto` to enable |
| RAG embeddings | `BAAI/bge-small-en-v1.5` | small | Local sentence-transformer on CPU |
| TTS voice | `en_US-amy-medium` (default) | small | Piper neural TTS server (`badapple_tts_server.py`) |

All models are downloaded and cached on the Mac. At runtime, **no prompt, response, or action leaves the machine**.

---

## Speed / Token Throughput

Live numbers from the daemon log on a 16 GB Apple Silicon M-series Mac with the single 9B brain loaded.

### Text mode (9B)

| Prompt | Prompt tokens | First token | Tokens out | Decode t/s | Peak memory |
|---|---|---:|---:|---:|---:|
| `Who are you?` | ~540 | 9.4 s | 31 | 16.5 | 5.84 GB |
| `What is the capital of France?` | ~540 | 9.5 s | 25 | 15.9 | 5.85 GB |
| `Tell me about Rome.` | ~540 | 7.6 s | 48 | 15.6 | 5.85 GB |
| `What do you think of Siri?` | ~540 | 7.1 s | 41 | 15.6 | 5.85 GB |
| `How does a car engine work?` | ~540 | 8.3 s | 41 | 15.6 | 5.85 GB |

- Typical first-token latency: **~7–9.5 s** for ~540 token prompts.
- Typical decode throughput: **~15.5–16.5 tok/s**.
- Benchmark total wall time for the 5-prompt suite: **~49 s** on the M3 Max test Mac.
- Peak memory stays **~5.8–5.9 GB**.
- The `total` column in the benchmark is **end-to-end tok/s including TTFT**, not the raw decode rate; look at the `decode` column for the model's actual token generation speed.

### Fast tier (0.5B)

When `BADAPPLE_FAST_TIER=1`, simple queries route through `mlx-community/Qwen2.5-0.5B-Instruct-4bit`.

| Prompt | Prompt tokens | First token | Tokens out | Decode t/s | Total t/s | Peak memory |
|---|---|---:|---:|---:|---:|---:|
| `Who are you?` | 35 | 0.23 s | 47 | 235.2 | 108.6 | 0.33 GB |
| `What is the capital of France?` | 38 | 0.19 s | 8 | 152.0 | 32.9 | 0.33 GB |
| `Tell me about Rome.` | 36 | 0.18 s | 120 | 132.8 | 110.6 | 0.33 GB |
| `What do you think of Siri?` | 38 | 0.66 s | 94 | 90.6 | 55.6 | 0.33 GB |
| `How does a car engine work?` | 38 | 0.47 s | 120 | 120.9 | 81.9 | 0.33 GB |

- Typical first-token latency: **~0.2–0.7 s**.
- Typical decode throughput: **~90–235 tok/s**.
- Peak memory: **~0.33 GB**.

Fast tier handles greetings, identity, time, simple math, and other deterministic/patterned queries. Non-trivial reasoning falls through to the 9B brain.

### Voice mode

Voice uses the 9B brain by default. When fast tier is on, short voice greetings and commands can also hit the 0.5B model.

### Inference tuning

- `BADAPPLE_DFLASH=0` — DFlash is off because it does not reliably beat plain `mlx-lm` on this quant.
- `BADAPPLE_SPECULATIVE_DRAFT=auto` — when set, Bad Apple scans the HF cache for a small compatible draft model and uses it with `mlx-lm` speculative decoding.
- `BADAPPLE_FAST_TIER=1` — the 0.5B fast model is enabled for appropriate queries.
- `prefill_step_size=4096` and `max_kv_size=4096` keep prompt encoding in a single shot and bound KV-cache growth.
- `prompt.txt` is hot-reloaded and kept compact; the 9B chat template only receives a focused subset of tool schemas per query, cutting prefill latency for tool-heavy prompts.

---

## What Bad Apple can do

Bad Apple is a private, on-device AI assistant for macOS. It runs the Qwen 3.5 9B brain and a 0.5B fast tier on Apple Silicon using MLX, answers questions, runs local tools, indexes files, and speaks responses through a local Piper TTS server. After the models are downloaded once, **no prompt, response, or action leaves the Mac**.

When asked, it can say:

> I can answer questions, run local tools, search files, write notes, run shell/AppleScript/Shortcuts, index documents, manage working memory, switch personas, speak, stream JSON, and run benchmarks — all on your Mac, babe.

## Features

### Core

- **Local Qwen 3.5 9B inference** on Apple Silicon GPU (MLX)
- **0.5B fast tier** for instant greetings, identity, time, simple math, and deterministic queries
- **Single 9B brain** for both text and voice, no dual-model swap
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
- `run_shell` — run a sandboxed shell command (read-only by default; `ls`, `cat`, `head`, `tail`, `find`, `grep`, `wc`, `file`, `pwd`, `mdfind`, `ps`, `df`, `du)
- `run_applescript` — execute AppleScript on macOS
- `run_shortcut` — run a macOS Shortcuts shortcut
- `index_documents` — index a directory into the local RAG store
- `git_status`, `git_diff`, `git_log`, `git_commit` — local git helpers
- `read_working_memory`, `write_working_memory`, `clear_working_memory` — scratchpad at `/var/lib/bad_apple/working_memory.txt`

### Feature roadmap status

The current build covers the following roadmap phases:

- **Phase 1 — Vision / screen understanding** ✅: `capture_and_describe_screen`, `screen_capture`, `describe_image`, and `extract_text_from_image` use the local `mlx-vlm` Qwen2-VL-2B model. Screen capture is now routed straight to the VLM, avoiding a full 9B tool-call generation.
- **Phase 2 — Persistent workspace mode** ✅: `workspace_status` reports build system, git branch, last commit, README summary, and recent files. The active workspace is set via `set workspace to <path>`.
- **Phase 3 — Local email / calendar / reminders** ✅: `today_events`, `upcoming_events`, `list_reminders`, `add_reminder`, `unread_emails`, and `search_mail` talk to the local macOS Calendar, Reminders, and Mail apps via AppleScript.
- **Phase 4 — Working memory dashboard** ✅: read/write/clear working memory directly with natural-language commands.
- **Phase 5 — Plugin / signed tool registry** ✅: `PluginRegistry` in `badapple_plugins.py` loads signed plugin manifests and exposes their tools as first-class `run_tool`/`_run_approved_tool` actions.
- **Phase 6 — Long-horizon episodic memory graph** ✅: `MemoryGraph` stores facts, entities, relations, episodes, and workflows, and recalls them via semantic search.
- **Phase 7 — MCP server expansion** ✅: `badapple_mcp_server.py` exposes a Unix-socket MCP transport at `/var/run/badapple/mcp.sock` for external tool clients.
- **Phase 8 — Encrypted P2P sync** 🔄: `badapple_p2p.py` is implemented but left off by default for the air-gap `cert_suite`; enable with `BADAPPLE_P2P=1`.
- **Phase 9 — Local image generation** ✅: `generate_image` uses the cached local `mflux` FLUX.2-klein-4B model.
- **Phase 10 — Streaming first-token preview** ✅: token streaming is live via `--json` and the `stream_queue` in `generate_with_tools`.
- **Phase 11 — Personal on-device LoRA fine-tuning** ✅: `lora_add_example`, `lora_train`, `lora_adapters`, and `lora_generate` use `mlx-lm` on local datasets.
- **Phase 12 — Dream / offline consolidation** ✅: `consolidate_memory` runs a memory-graph deduplication and re-embedding pass.
- **Phase 13 — Policy language for the cage** ✅: `Policy` in `badapple_extras.py` declares which tools are allowed, require approval, and how arguments are validated; loaded from `policy.yaml`.
- **Phase 14 — Adversarial output classifier** ✅: `StreamingFirewall` blocks PII, secrets, and custom blocklist patterns in generated output with a streaming Aho-Corasick automaton.
- **Phase 15 — Self-hosting model registry** 🔄: the active model is controlled by `BADAPPLE_MAIN_MODEL`; a pluggable local registry is the next remaining step.

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
│  - loads single 9B target + optional small draft │
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
| `BADAPPLE_DRAFT_MODEL` | `z-lab/Qwen3.5-9B-DFlash` | Legacy DFlash draft for the 9B target (unused) |
| `BADAPPLE_SPECULATIVE_DRAFT` | `auto` | Enable `mlx-lm` speculative decoding with an auto-detected small draft |
| `BADAPPLE_NUM_DRAFT_TOKENS` | `2` | Tokens to draft per verification step |
| `BADAPPLE_DFLASH_QUANTIZE_KV` | `1` | Quantize key/value cache |
| `BADAPPLE_PROMPT_FILE` | `prompt.txt` | Hot-reloadable system prompt |
| `BADAPPLE_TTS_VOICE` | `en_US-amy-medium` | Default TTS voice |

---

## Caveats

- **Speculative decoding is optional and requires a small compatible draft model**: without a cached draft, plain `mlx-lm` is used.
- **Throughput is workload-dependent**: DFlash acceptance swings from ~45% to ~80%, so tok/s swings with it. Sustained 22–33 tok/s is possible on high-acceptance turns but not guaranteed for every prompt on this hardware.
- **First-token latency is dominated by prefill**: long prompts or large knowledge chunks push the first token toward the 5–7 s range.
- **Max-token cutoffs are cleaned up**: if the model runs out of output tokens mid-sentence, the response is trimmed to the last complete sentence so it doesn't end on a dangling word.
- The older `automation_cage`, `wasm_cage`, and 576-D `tensor_brain` classifier still exist in the repo but are **not the active path**. The active stack is the gatekeeper + MLX server + CLI + TTS server + menu bar.
