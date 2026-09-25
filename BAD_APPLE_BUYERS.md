# Bad Apple — Bare-Metal AI Operating System for macOS

## What it is

Bad Apple is a self-hosted, air-gapped personal AGI operating system for macOS. It runs a 7B Qwen 2.5 Coder brain as the default reasoning and coding model — accelerated by a resident 0.5B speculative draft and a reusable prompt-prefix KV cache — with a 9B Qwen 3.5 general model as a switchable option, all on Apple Silicon using MLX. It answers questions, runs local tools, indexes your files, watches conditions you set, holds standing orders, anticipates your calendar, speaks responses through a native TTS server, and can shard a single large model layer-range across peer Macs over an encrypted mesh — all without sending prompts, responses, or actions to a cloud service after the initial model download.

## How to install (consumer)

1. Download the unsigned release zip from GitHub.
2. Unzip it and drag `Bad Apple.app` into `/Applications`.
3. Double-click `Install Bad Apple` and enter your Mac password when asked.
4. Launch `Bad Apple.app` from `/Applications`. A friendly onboarding window will guide you through the first run.

## The pitch

- **Air-gapped by default**: no prompt, no action, no memory leaves your Mac.
- **Hardware-bound identity**: SLICKS v2 signs every client–daemon connection and P2P frame with the Apple Secure Enclave.
- **Actor-ized OS**: resources, circuit breakers, persona, workspace, P2P, MCP, cache, audit, model, and health run as supervised actors.
- **Two brains, one daemon**: a resident 0.5B draft accelerates the 7B Coder brain (speculative decoding + a reusable prompt-prefix KV cache, ~47% faster turns measured); the 9B brain is available for heavier general reasoning (`badapple model use main_9b`).
- **Persistent attention**: watchers monitor files, processes, text, and mail senders and can fire goals; standing orders and scheduled tasks run through the agent loop; calendar lookahead preps for upcoming events.
- **Senses**: opt-in ambient hearing (on-device speech recognition), screen ocular, clipboard recall, and bounded meeting capture.
- **Perimeter sentinel**: diffs launch-agent/persistence surfaces and network listeners; builds evidence trails for anomalies.
- **Native task board**: `⌘T` in the menu bar shows live task/step state; failed agent steps replan instead of aborting.
- **Aqua bridge**: local mail, calendar, reminders, and messages tools with policy-gated approvals.
- **Local tooling**: search files, run AppleScript, run Shortcuts, get the time, write notes, index documents, git helpers, and more.
- **Persistent memory + RAG**: remembers user facts and searches indexed local documents.
- **Hot-reloadable persona**: edit `prompt.txt` without restarting the 7B/9B model.
- **MCP + local marketplace**: exposes tools to MCP clients and can run local stdio MCP servers under the same policy gate.
- **Curious self-improvement**: in `curious` persona with autopilot on, Bad Apple runs a bounded self-check (audit, firewall, git, source markers) and writes a proposal note so it can improve on its own, all local and policy-gated.
- **Council of Minds**: a 14-seat deterministic council — four financial minds (Buffett, Dalio, Musk, Jobs) and ten strategists (Sun Tzu, Clausewitz, Musashi, Machiavelli, Napoleon, Hannibal, Aurelius, Boyd, Genghis Khan, Patton) — votes on every gated action. Under autopilot, passed votes execute and failed votes come back to the human for approval; every deliberation lands on the audit ledger with per-seat rationales. Ask the council anything directly with `council <question>`.

## Models loaded

| Component | Model | Size | Role |
|---|---|---|---|
| Target LLM (default) | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | ~4.2 GB | Coding, general chat, and tool reasoning |
| Target LLM (switchable) | `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` | ~6.2 GB | Heavier general reasoning: `badapple model use main_9b` |
| Fast tier | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | ~0.3 GB | Greetings, identity, time, simple math, deterministic queries |
| Embeddings | `BAAI/bge-small-en-v1.5` | small | Local document / memory retrieval on CPU |
| TTS voice | `en_US-amy-medium` (default) | small | Native neural speech on a local socket |

All models are cached on disk after the first download. Nothing is re-downloaded at runtime.

## Performance (live M-series Apple Silicon, 16 GB unified memory)

### 7B Qwen 2.5 Coder (default)

Live 7B numbers on a 16 GB Apple Silicon Mac after the v0.1.7 KV-cache, supervisor-health, and event-driven Curious self-repair fixes, measured once the model was warm and build pressure had settled:

| Prompt | Prompt tokens | First token | Tokens out | Decode t/s | Peak memory |
|---|---|---|---:|---:|---:|
| `Who are you?` | ~1160 | 0.97 s | 16 | 19.2 | 4.12 GB |
| `What is the capital of France?` | ~1160 | 0.43 s | 16 | 18.0 | 4.12 GB |
| `Tell me about Rome.` | ~1160 | 0.40 s | 16 | 21.8 | 4.12 GB |
| `What do you think of Siri?` | ~1160 | 0.40 s | 16 | 16.1 | 4.12 GB |
| `How does a car engine work?` | ~1160 | 0.42 s | 16 | 22.8 | 4.12 GB |

- Typical first-token latency: **~0.4–1.0 s** for warm cached prompts; **~14 s** for a cold model start.
- Typical decode throughput: **~16–23 tok/s**, with a 5-prompt suite average of **19.6 tok/s**.
- Peak memory: **~4.1 GB**.
- Numbers are measured after the build has finished and the system has settled; tok/s can drop 25-40% if benchmarked during or immediately after a Rust/Swift build.

### 9B Qwen 3.5 (switchable)

| Prompt | Prompt tokens | First token | Tokens out | Decode tok/s | Peak memory |
|---|---|---:|---:|---:|---:|
| `Who are you?` | ~1800 | 0.00 s | 0 | — | 6.08 GB |
| `What is the capital of France?` | ~1800 | 4.23 s | 20 | 10.2 | 6.08 GB |
| `Tell me about Rome.` | ~1800 | 3.98 s | 20 | 10.2 | 6.08 GB |
| `What do you think of Siri?` | ~1800 | 4.28 s | 66 | 10.3 | 6.08 GB |
| `How does a car engine work?` | ~1800 | 4.92 s | 66 | 10.3 | 6.08 GB |

- Typical first-token latency: **~4.0–4.9 s** once the system-prompt KV cache is loaded.
- Typical decode throughput: **~10.2–10.3 tok/s**.
- Benchmark total wall time: **~26.8 s** for the 5-prompt suite on an M3 Max test Mac.
- Peak memory: **~6.0–6.1 GB**.

### 0.5B fast tier

| Prompt | Prompt tokens | First token | Tokens out | Decode tok/s | Total tok/s | Peak memory |
|---|---|---:|---:|---:|---:|---:|
| `Who are you?` | 35 | 0.23 s | 47 | 235.2 | 108.6 | 0.33 GB |
| `What is the capital of France?` | 38 | 0.19 s | 8 | 152.0 | 32.9 | 0.33 GB |
| `Tell me about Rome.` | 36 | 0.18 s | 120 | 132.8 | 110.6 | 0.33 GB |
| `What do you think of Siri?` | 38 | 0.66 s | 94 | 90.6 | 55.6 | 0.33 GB |
| `How does a car engine work?` | 38 | 0.47 s | 120 | 120.9 | 81.9 | 0.33 GB |

- Typical first-token latency: **~0.2–0.7 s**.
- Typical decode throughput: **~90–235 tok/s**.
- Peak memory: **~0.33 GB**.

## What Bad Apple can do

When asked, it should say:

> Here's what I can do, babe:
> 1. Answer questions, explain, summarize, brainstorm, and chat — fully local and air-gapped.
> 2. Run local tools: shell, AppleScript, file read/write/search, and macOS Shortcuts with your approval.
> 3. Use local MCP servers with per-tool write approvals.
> 4. Index documents for RAG, remember facts, and manage a workspace / project context.
> 5. Run multi-step agent tasks, a bounded Curious self-improvement check, and capture ambient context.
> 6. Engage the kill switch and the air-gap hard switch.
> 7. Pre-download models, switch personas, run benchmarks, and stream JSON.
> 8. Speak responses through the local TTS server and integrate with macOS Shortcuts and Siri.
> 9. Show a local web dashboard / control center at http://127.0.0.1:8787.
> 10. Sync with other Bad Apple peers over an encrypted, Secure Enclave–signed P2P mesh — off by default.
> 11. Send a full model from one Mac to another over the local network; files are verified against the signed manifest.
> 12. Self-hosting model registry with `badapple model` CLI: list, verify, add, remove, and switch local models; manifests are signed by the Secure Enclave.
> 13. Run as a supervised actor OS: every subsystem is isolated, restartable, and observable.
> Everything stays on your Mac.

### 1. Core AI
1. Answer questions, explain, summarize, brainstorm, and write short notes
2. Run a local 7B Qwen 2.5 Coder brain with a resident 0.5B speculative draft on Apple Silicon; 9B Qwen 3.5 is switchable
3. Switch persona at runtime: `switch to cali`, `wicket`, `genz`, `drill`, `midwest`
4. `switch to roast` or `--roast` for the drill persona
5. `teach <line>` to store a custom quip
6. Multi-turn conversation with local JSONL history
7. Hot-reloadable system prompt via `prompt.txt`
8. `curious check` — runs a bounded self-improvement audit and writes a proposal note; `curious` persona with autopilot can run it on a loop

### 2. Memory and knowledge
1. Working memory / scratchpad
2. Persistent user memory graph and episodic recall
3. Semantic cache for repeated questions
4. Local document indexing and RAG search
5. Workspace / project context

### 3. Voice and interface
1. Text input via the Rust CLI
2. Voice mode with STT / "hey bad apple" wake
3. Local native TTS with `--speak`
4. macOS menu bar app with voice HUD
5. macOS Shortcuts and Siri AppIntents
6. Local web dashboard at `http://127.0.0.1:8787`
7. REST `/api/chat` endpoint and web chat UI

### 4. Tools and agent OS
1. Natural-language tool router
2. Shell command execution and AppleScript execution
3. Run macOS Shortcuts and list installed shortcuts
4. File read / write / search / list / grep
5. Spotlight search via `mdfind`
6. Vision and screen description
7. Image generation
8. Xcode project indexing and local git helpers
9. Multi-step agent tasks and planning
10. Local MCP server marketplace with per-tool write approvals

### 5. Security and control
1. Human-in-the-loop approval gate
2. Autopilot mode — gated actions get a 14-seat council vote: passed votes run, failed votes escalate to the human
3. Private mode to pause persistence
4. Kill switch / emergency stop
5. Air-gap hard switch to block network MCP and downloads
6. Output firewall with blocklist patterns
7. Hash-chained audit ledger with secret/PII redaction, plus an independently verified hardened copy anchored to daily Secure Enclave–signed checkpoints
8. SLICKS v2 Unix socket authentication with Secure Enclave ECDSA (HMAC fallback)
9. Air-gap certification suite that verifies local-only sockets, model provenance, and SE identity

### 6. Mesh and networking
1. P2P encrypted LAN sync (off by default)
2. Peer discovery and sync beacons
3. P2P model manifest gossip: discover models on nearby Bad Apple nodes
4. P2P model file transfer: send and receive a whole model between Macs
5. `badapple p2p <peers|sync|models|pull|send|receive>` CLI

### 7. Model and memory management
1. Self-hosting model registry: `badapple model <list|scan|info|use|verify|add|remove|recommend>`
2. SHA-256 provenance manifests signed with the Secure Enclave
3. Model manager with cache status
4. One-click model pre-download with memory check
3. Runtime model switching and recommendations
4. VRAM governor and memory pressure handling
5. `/api/models` dashboard page

### 8. Observability and platform
1. Health supervisor and launchd daemon management
2. Rust gatekeeper proxy
3. Log rotation and current-process log filtering
4. Audit ledger tail and live log tail in the dashboard
5. Unsigned release packaging and auto-updater

### Menu bar app

`Bad Apple.app` lives in the macOS status bar. Right-click the apple icon for:

- **New Chat** — clears conversation history
- **Chat History** — opens the transcript window
- **Voice Listening** — toggle always-on voice wake
- **Roast Mode** — alias for the `drill` persona on the next voice query
- **Persona** — switch between Default, Wicket, Gen Z, Drill, and Midwest Aunt
- **Fast Tier Only** — toggle the 0.5B fast tier
- **Benchmark** — runs the default prompt suite and shows results
- **Voice (native / Apple)** and **Accent** — TTS engine and voice selection
- **Quit**

## Privacy & security

- Prompts and responses never leave the machine during normal use.
- Tool actions (file search, AppleScript, app open) run locally.
- Conversation history and user memory are stored locally, not synced.
- Client-to-daemon traffic is over a Unix socket with SLICKS HMAC challenge/response.
- The launchd daemon runs as root and auto-restarts.
- Output firewall blocks PII, secrets, and custom patterns.
- Hash-chained audit ledger with redaction at `/var/lib/bad_apple/ledger.jsonl`, hardened daily into an independent HMAC-chained sovereign copy with Secure Enclave–signed checkpoints.
