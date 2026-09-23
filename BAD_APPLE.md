# Bad Apple — Bare-Metal AI Operating System for macOS

Bad Apple is a self-hosted, air-gapped personal AGI operating system for macOS. It runs a 7B Qwen 2.5 Coder as the default model on Apple Silicon using MLX, with a 9B Qwen 3.5 model as a switchable option. It answers questions, runs local tools, indexes your files, and speaks responses through a local neural TTS server — all without sending anything to the cloud after the models are downloaded once.

Internally it is built as a supervised **actor OS**: every major subsystem — resources, circuit breakers, workspace, persona, P2P, MCP, health, audit, cache, and model registry — runs as a dedicated actor. The Rust CLI, the menu bar, and any MCP client authenticate to the daemon through **SLICKS v2**, a hardware-bound challenge/response protocol signed by the Apple Secure Enclave.

---

## Brains / Models

The runtime now uses a 7B Qwen 2.5 Coder 4-bit brain as the default for both text and voice, with a 9B Qwen 3.5 4-bit brain available as a switchable option. There is no separate voice bundle to swap in and out, which removes the old multi-second mode-switching lag.

| Component | Model | Size | Role |
|---|---|---|---|
| Target LLM (default) | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | ~4.2 GB | Coding, general chat, and tool reasoning |
| Target LLM (switchable) | `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` | ~6.2 GB | Heavier general reasoning: `badapple model use main_9b` |
| Fast tier / tiny brain | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | ~0.3 GB | Instant answers for greetings, identity, time, simple math, and deterministic queries |
| MLX-LM speculative draft (optional) | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | ~0.3 GB | Optional small draft for the main brain; set `BADAPPLE_SPECULATIVE_DRAFT=auto` to enable |
| RAG embeddings | `BAAI/bge-small-en-v1.5` | small | Local sentence-transformer on CPU |
| TTS voice | `en_US-amy-medium` (default) | small | Native AVFoundation TTS server (`badapple-tts`) |

All models are downloaded and cached on the Mac. At runtime, **no prompt, response, or action leaves the machine**.

---

## Speed / Token Throughput

Live numbers from the daemon log on a 16 GB Apple Silicon M-series Mac. The default is the 7B brain; the 9B numbers are from the optional `main_9b` profile.

### Text mode (7B, default)

Observed ranges on a 16 GB Apple Silicon Mac with the 7B Coder brain loaded:

| Prompt | Prompt tokens | First token | Tokens out | Decode t/s | Peak memory |
|---|---|---|---:|---:|---:|---:|
| `Who are you?` | ~1160 | 0.97 s | 16 | 19.2 | 4.12 GB |
| `What is the capital of France?` | ~1160 | 0.43 s | 16 | 18.0 | 4.12 GB |
| `Tell me about Rome.` | ~1160 | 0.40 s | 16 | 21.8 | 4.12 GB |
| `What do you think of Siri?` | ~1160 | 0.40 s | 16 | 16.1 | 4.12 GB |
| `How does a car engine work?` | ~1160 | 0.42 s | 16 | 22.8 | 4.12 GB |

- Typical first-token latency: **~0.4–1.0 s** for warm cached prompts; **~14 s** for a cold cache/model-load start on a 16 GB Mac.
- Typical decode throughput: **~16–23 tok/s**, with a suite average of **19.6 tok/s** on this quant once the model is warm and no build is running.
- Peak memory stays **~4.1 GB**.
- **Important:** running `cargo build`, `swift build`, or packaging immediately before benchmarking will increase swap and can drop tok/s by 25-40%. Run the benchmark after the build has finished and the system has had ~30-60 s to settle.

### Text mode (9B, switchable)

| Prompt | Prompt tokens | First token | Tokens out | Decode t/s | Peak memory |
|---|---|---:|---:|---:|---:|
| `Who are you?` | ~1800 | 0.00 s | 0 | — | 6.08 GB |
| `What is the capital of France?` | ~1800 | 4.23 s | 20 | 10.2 | 6.08 GB |
| `Tell me about Rome.` | ~1800 | 3.98 s | 20 | 10.2 | 6.08 GB |
| `What do you think of Siri?` | ~1800 | 4.28 s | 66 | 10.3 | 6.08 GB |
| `How does a car engine work?` | ~1800 | 4.92 s | 66 | 10.3 | 6.08 GB |

- Typical first-token latency: **~4.0–4.9 s** for ~1800 token prompts once the system-prompt KV cache is loaded (down from ~7–9.5 s without the cache).
- The `Who are you?` prompt is handled by fast meta-response logic, so it does not run the 9B brain and reports 0 output tokens.
- Typical decode throughput: **~10.2–10.3 tok/s** on this quant.
- Benchmark total wall time for the 5-prompt suite: **~26.8 s** on the M3 Max test Mac.
- Peak memory stays **~6.0–6.1 GB**.
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

Fast tier handles greetings, identity, time, simple math, and other deterministic/patterned queries. Non-trivial reasoning falls through to the 7B Coder brain by default, or to the 9B brain if it is active.

### Voice mode

Voice uses the 7B brain by default. When fast tier is on, short voice greetings and commands can also hit the 0.5B model. Switch to the 9B brain with `badapple model use main_9b`.

### Inference tuning

- `BADAPPLE_DFLASH=0` — DFlash is off because it does not reliably beat plain `mlx-lm` on this quant.
- `BADAPPLE_SPECULATIVE_DRAFT=auto` — when set, Bad Apple scans the HF cache for a small compatible draft model and uses it with `mlx-lm` speculative decoding.
- `BADAPPLE_FAST_TIER=1` — the 0.5B fast model is enabled for appropriate queries.
- `prefill_step_size=2048` and `max_kv_size=2048` are the installer defaults on a 16 GB Mac; they keep prompt encoding in one or two shots and bound KV-cache growth. Set them higher if you have 32 GB+ and need longer context.
- `prompt.txt` is hot-reloaded and kept compact; the 7B/9B chat template only receives a focused subset of tool schemas per query, cutting prefill latency for tool-heavy prompts.

---

## What Bad Apple can do

Bad Apple is a private, on-device personal AGI for macOS. It runs a 7B Qwen 2.5 Coder brain and a 0.5B fast tier on Apple Silicon using MLX, with a 9B Qwen 3.5 brain as a switchable option. It answers questions, runs local tools, indexes files, and speaks responses through a native AVFoundation TTS server. After the models are downloaded once, **no prompt, response, or action leaves the Mac**.

When asked, it can say:

> I can answer questions, run local tools, search files, write notes, run shell/AppleScript/Shortcuts, index documents, manage working memory, switch personas, run a self-improvement check, speak, stream JSON, and run benchmarks — all on your Mac, babe.

## Features

### Core

- **Local Qwen 2.5 Coder 7B inference** on Apple Silicon GPU (MLX); 9B Qwen 3.5 switchable
- **0.5B fast tier** for instant greetings, identity, time, simple math, and deterministic queries
- **Single main brain** for both text and voice, no dual-model swap
- **Streaming token output** to terminal or TTS
- **SLICKS v2 authenticated Unix socket** (Secure Enclave–signed challenge/response, with HMAC-SHA256 v1 fallback)
- **launchd-managed daemon** (`com.badapple.mlx`) that runs as root and auto-restarts

### Conversation & memory

- **Multi-turn conversation history** saved to local JSONL
- **Persistent user memory**: records user facts and recalls them in future turns
- **RAG / local document search**: indexes your text files with `BAAI/bge-small-en-v1.5` and retrieves relevant chunks
- **Hot-reloadable system prompt** via `prompt.txt` without restarting the main brain
- **Persona packs** (`personas.json`): switch at runtime with `switch to <persona>`
- **Teachable quips** with `teach <line>`
- **Streaming output firewall** (Aho-Corasick blocklist) for PII, secrets, and custom patterns
- **Hash-chained audit ledger** (`/var/lib/bad_apple/ledger.jsonl`) with PII redaction, plus a hardened sovereign copy (`ledger.sovereign.jsonl`) re-verified daily and anchored to Secure Enclave–signed checkpoints
- **Semantic cache** (`BAAI/bge-small-en-v1.5`) for instant repeated-answer hits
- **Human-in-the-loop approvals** for destructive tools
- **Council of Minds**: 14 deterministic strategist/financial seats vote on every gated action — passed votes execute under autopilot, failed votes escalate to the human; semantic `council <question>` sessions voice all seats; deliberations journaled on the ledger
- **Bounded Curious self-improvement autopilot** — when `curious` persona is active and autopilot is on, the engine runs a local self-check (cert suite, doctor, output firewall, git status, source TODO/FIXME scan) and writes a proposal note to `~/.bad_apple/notes/proposed_patches/`. Trigger manually with `badapple "curious check"`.
- **Human layer** (`~/.bad_apple/human/state.json`, mode 0600): persistent relationship preferences, shared commitments with due dates, and ongoing life threads. Private mode blocks all writes. The active attention mode (`available`/`focus`/`quiet`/`sleep`) governs whether proactive events speak, notify, or are held in Human Home; due commitments surface as `commitment_due` notifications through the same governed channel.

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
- `human_home` — read the shared-life state (due commitments, waiting items, preferences, life threads, held notifications)
- `remember_preference`, `forget_preference` — store/remove explicitly stated preferences
- `add_commitment`, `update_commitment` — record and complete commitments (`owner` user or bad_apple, optional `due_at`)
- `manage_life_thread` — create/update ongoing life threads
- `set_attention_mode` — set `available`, `focus`, `quiet`, or `sleep` delivery

### Feature roadmap status

The current build covers the following roadmap phases:

- **Phase 1 — Vision / screen understanding** ✅: `capture_and_describe_screen`, `screen_capture`, `describe_image`, and `extract_text_from_image` use the local `mlx-vlm` Qwen2-VL-2B model. Screen capture is now routed straight to the VLM, avoiding a full 9B tool-call generation.
- **Phase 2 — Persistent workspace mode** ✅: `workspace_status` reports build system, git branch, last commit, README summary, and recent files. The active workspace is set via `set workspace to <path>`.
- **Phase 3 — Local email / calendar / reminders** ✅: `today_events`, `upcoming_events`, `list_reminders`, `add_reminder`, `unread_emails`, and `search_mail` talk to the local macOS Calendar, Reminders, and Mail apps via AppleScript.
- **Phase 4 — Working memory dashboard** ✅: read/write/clear working memory directly with natural-language commands.
- **Phase 5 — Plugin / signed tool registry** ✅: `PluginRegistry` loads signed plugin manifests and exposes their tools as first-class `run_tool`/`_run_approved_tool` actions.
- **Phase 6 — Long-horizon episodic memory graph** ✅: `MemoryGraph` stores facts, entities, relations, episodes, and workflows, and recalls them via semantic search.
- **Phase 7 — MCP server expansion** ✅: A Unix-socket MCP transport at `/var/run/badapple/mcp.sock` exposes tools and resources to MCP clients, with request size limits, tool allowlists, and per-call timeouts. The MCP marketplace runs as a supervised actor.
- **Phase 8 — Encrypted P2P sync** ✅: P2P sync is implemented with AES-256-GCM link-local sync and Secure Enclave–signed origin authentication. It is off by default for the air-gap certification suite; enable with `BADAPPLE_P2P=1`.
- **Phase 9 — Local image generation** ✅: `generate_image` uses the cached local `mflux` FLUX.2-klein-4B model.
- **Phase 10 — Streaming first-token preview** ✅: token streaming is live via `--json` and the `stream_queue` in `generate_with_tools`.
- **Phase 11 — Personal on-device LoRA fine-tuning** ✅: `lora_add_example`, `lora_train`, `lora_adapters`, and `lora_generate` use `mlx-lm` on local datasets.
- **Phase 12 — Dream / offline consolidation** ✅: `consolidate_memory` runs a memory-graph deduplication and re-embedding pass.
- **Phase 13 — Policy language for the cage** ✅: `Policy` in `policy.yaml` declares which tools are allowed, require approval, and how arguments are validated.
- **Phase 14 — Adversarial output classifier** ✅: `StreamingFirewall` blocks PII, secrets, and custom blocklist patterns in generated output with a streaming Aho-Corasick automaton.
- **Phase 15 — Self-hosting model registry** ✅: `badapple model <list|scan|info|use|verify|add|remove|recommend>` manages the local cache, records SHA-256 provenance, and signs manifests with the Secure Enclave.
- **Phase 16 — P2P model manifest gossip + file transfer** ✅: the link-local mesh shares signed model manifests and streams the actual model weight files between peers. `badapple p2p <peers|sync|models|pull|send|receive>` discovers neighbors, syncs memory, pulls a manifest, and sends/receives the full model over encrypted local TCP.
- **Phase 17 — Council of Minds** ✅: a 14-seat deterministic council (4 financial minds + 10 strategists) votes on every gated action before execution. Under autopilot, passed votes run and failed votes escalate to the human as proposals; in manual mode the verdict advises the approval prompt. `badapple "council <question>"` runs the semantic session — the local model voices all fourteen seats and returns a verdict. Every deliberation is journaled on the audit ledger.

### Menu bar app

`Bad Apple.app` lives in the macOS status bar. The first time it runs, a plain-English onboarding panel walks you through installing the small background helper. After that, right-click the apple icon for:

- **Status...** — plain-English snapshot of brain, memory, P2P, and MCP
- **Human Home...** — read-only window showing the human layer: NOW (due commitments), WAITING ON YOU, I'M HANDLING, REMEMBERING, LIFE THREADS, and HELD FOR LATER
- **New Chat** — clears conversation history
- **Chat History** — opens the transcript window
- **Voice Listening** — toggle always-on voice wake
- **Roast Mode** — alias for the `drill` persona on the next voice query
- **Persona** — switch between Default, Wicket, Gen Z, Drill, and Midwest Aunt
- **Benchmark** — runs the default prompt suite and shows results
- **Voice (native / Apple)** and **Accent** — TTS engine and voice selection
- **Quit**

Voice queries respect the selected persona and roast mode by passing `--persona <name>` / `--roast` to the bundled `badapple` CLI helper.

### Voice / TTS

- `--speak` streams each sentence to the local **native TTS server** (`badapple-tts`) and plays with `afplay`
- Voice can be changed via `BADAPPLE_TTS_VOICE` (default `en_US-amy-medium`)
- `BADAPPLE_TTS_LENGTH_SCALE` and `BADAPPLE_TTS_VOLUME` control voice speed and volume
- TTS server runs under `com.badapple.tts`

### Persona

Default persona is a sassy, flirty California beach girl. She:

- Uses English-only slang and endearments (`babe`, `hun`, `homie`, `dude`, `stoked`, `chill`)
- Stays short: 1–2 punchy paragraphs
- **Roasts cloud AI and Siri** when bragging about bare metal or when asked directly. She varies the target (Siri, Alexa, Google, ChatGPT, Gemini, Cortana, Bixby, "the cloud", server farms, data centers, "some rented GPU in Nevada") and the insult ("ratchet old bitch", "washed-up cloud snitch", "data-hungry narc", "internet junkie", "corporate eavesdropper", etc.)

Built-in persona packs (in `personas.json`) include Wicket (witty Londoner), Gen Z Hype, Drill Rapper, and Midwest Aunt.  Switch at runtime with `switch to <persona>` or set `BADAPPLE_PERSONA=<name>`.

---

## Security & governance

| Layer | Mechanism | Where it lives |
|---|---|---|
| Output firewall | Streaming Aho-Corasick blocklist on generated text | `badapple-engine` |
| Audit | Append-only SHA-256–chained JSONL with secret/PII redaction; independent HMAC-SHA256 sovereign copy + Secure Enclave checkpoints via `badapple-sovereign` | `/var/lib/bad_apple/ledger.jsonl` |
| Watchdog | IFY (`badapple-ify`): verifies each new ledger line as it arrives, learns event/tool/approval baselines over a 14-day gestation, surfaces anomalies as approval-gated proposals, and in autopilot phase can pull the kill switch on critical findings (chain break, ledger truncation). See `IFY.md` | `~/.bad_apple/ify/` |
| Approvals | Proposal/approve workflow for `run_shell`, `run_applescript`, `write_file`, `index_documents` | `badapple-engine` |
| Council | 14 deterministic seats (4 financial minds + 10 strategists) vote on every gated action via an 8-feature action encoder (destructiveness, irreversibility, blast radius, sensitivity, privilege, scope, cost, novelty); under autopilot, passed votes execute and failed votes escalate to a human proposal; every deliberation is journaled as `council_deliberation` with per-seat votes and rationales | `BadAppleCouncil.swift` |
| Cache | Persona-scoped semantic cache; no cache for tool queries or voice | `badapple-engine` |

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

# Self-hosting model registry (SE-signed provenance)
badapple model list
badapple model scan
badapple model info caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit
badapple model use caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit
badapple model verify
badapple model add /path/to/local/model
badapple model remove <id>
badapple model recommend

# Encrypted link-local P2P edge mesh
badapple p2p peers
badapple p2p sync
badapple p2p models
badapple p2p pull <peer_id> <model_id>
badapple p2p send <peer_id> <model_id>
badapple p2p receive [peer_id] [model_id]
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
            │ Unix socket + SLICKS v2 auth (Secure Enclave / HMAC fallback)
            ▼
┌───────────────────────────────────────┐
│  com.badapple.gatekeeper              │
│  target/release/gatekeeper            │
│  - SLICKS v2 proxy, fast actions,     │
│    automation cage                    │
└───────────┬───────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  com.badapple.mlx                     │
│  badapple-engine (Swift MLX)          │
│  - loads single 9B target + optional small draft │
│  - actor-ized subsystems              │
│  - runs tools, memory, RAG, TTS, P2P, │
│    MCP marketplace                    │
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  com.badapple.tts                     │
│  badapple-tts (AVFoundation)          │
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
- **Hardware-bound identity**: every client and the daemon prove themselves with SLICKS v2 (Secure Enclave ECDSA), with SLICKS v1 (HMAC-SHA256) as a fallback
- **Actor-isolated subsystems**: resources, circuit breakers, persona, workspace, P2P, MCP, cache, audit, model, and health run as supervised actors
- **Fail-closed tools**: file writes are restricted to `BADAPPLE_NOTES_DIR`; shell/AppleScript calls are gated
- **Local-only audio**: TTS happens on-device

---

## How to use

```bash
# Consumer install (one-click)
# 1. Unzip the release.
# 2. Drag "Bad Apple.app" to /Applications.
# 3. Double-click "Install Bad Apple" in the zip folder and enter your Mac password.

# Build from source
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
| `BADAPPLE_MAIN_MODEL` | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | Target model for all modes |
| `BADAPPLE_DRAFT_MODEL` | (none) | Legacy DFlash draft for the 9B target (unused) |
| `BADAPPLE_SPECULATIVE_DRAFT` | (none) | Enable `mlx-lm` speculative decoding with an auto-detected small draft |
| `BADAPPLE_NUM_DRAFT_TOKENS` | `2` | Tokens to draft per verification step |
| `BADAPPLE_MAX_KV_SIZE` | `2048` (installer default on 16 GB) | Max KV cache size in tokens |
| `BADAPPLE_PREFILL_STEP_SIZE` | `2048` (installer default on 16 GB) | Max tokens to prefill in one step |
| `BADAPPLE_DFLASH_QUANTIZE_KV` | `1` | Quantize key/value cache |
| `BADAPPLE_PROMPT_FILE` | `prompt.txt` | Hot-reloadable system prompt |
| `BADAPPLE_TTS_VOICE` | `en_US-amy-medium` | Default TTS voice |

---

## Caveats

- **Speculative decoding is optional and requires a small compatible draft model**: without a cached draft, plain `mlx-lm` is used.
- **Throughput is workload-dependent**: DFlash acceptance swings from ~45% to ~80%, so tok/s swings with it. Sustained 22–33 tok/s is possible on high-acceptance turns but not guaranteed for every prompt on this hardware.
- **First-token latency is dominated by prefill**: a cold model start can take ~14 s on a 16 GB Mac; warm cached prompts are usually sub-second.
- **Max-token cutoffs are cleaned up**: if the model runs out of output tokens mid-sentence, the response is trimmed to the last complete sentence so it doesn't end on a dangling word.
- `automation_cage` and `wasm_cage` are **not dead code** despite earlier docs claiming otherwise: `src/bin/gatekeeper.rs`'s `post_process_response()` runs on every single response, for both SLICKS v1 and v2 clients, scanning the model's output for ` ```badapple-action ` / ` ```badapple-wasm ` blocks and executing them in the respective sandbox. The 576-D `tensor_brain` classifier (`SemanticRouter`) and its `FastActionResolver` fast-path *are* effectively bypassed for SLICKS v2 clients (the default whenever the identity agent socket is present, i.e. the normal case on an installed machine) — v2 requests skip straight to `forward_v2_to_mlx`, which proxies the handshake end-to-end and only applies cage/wasm post-processing to the final text, without the native complexity classification or regex fast-action matching. That classify+fast-action path remains live for SLICKS v1 clients (`BADAPPLE_SLICKS2=0`, or no identity agent running). The active stack is the gatekeeper (proxy + cage/wasm post-processing, always; native classifier + fast actions, v1-only) + MLX server + CLI + TTS server + menu bar.
