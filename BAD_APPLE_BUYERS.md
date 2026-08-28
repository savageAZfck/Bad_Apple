# Bad Apple — Buyer's Overview

## What it is

Bad Apple is a private, on-device AI assistant for macOS. It runs a Qwen 3.5 9B brain and a Qwen 2.5 0.5B fast tier on Apple Silicon using MLX, answers questions, runs local tools, indexes your files, and speaks responses through a local Piper TTS server — all without sending prompts, responses, or actions to a cloud service after the initial model download.

## The pitch

- **Air-gapped by default**: no prompt, no action, no memory leaves your Mac.
- **Two brains, one daemon**: the 0.5B fast tier handles instant greetings/time/math; the 9B brain handles reasoning.
- **Local tooling**: search files, run AppleScript, run Shortcuts, get the time, write notes, index documents, git helpers, and more.
- **Persistent memory + RAG**: remembers user facts and searches indexed local documents.
- **Hot-reloadable persona**: edit `prompt.txt` without restarting the 9B model.
- **Authenticated socket**: SLICKS HMAC challenge/response over a Unix socket.

## Models loaded

| Component | Model | Size | Role |
|---|---|---|---|
| Target LLM | `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` | ~6.2 GB | All text and voice reasoning |
| Fast tier | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | ~0.3 GB | Greetings, identity, time, simple math, deterministic queries |
| Embeddings | `BAAI/bge-small-en-v1.5` | small | Local document / memory retrieval on CPU |
| TTS voice | `en_US-amy-medium` (default) | small | Piper neural speech on a local socket |

All models are cached on disk after the first download. Nothing is re-downloaded at runtime.

## Performance (live M-series Apple Silicon, 16 GB unified memory)

### 9B brain

| Prompt | Prompt tokens | First token | Tokens out | Decode tok/s | Peak memory |
|---|---|---:|---:|---:|---:|
| `Who are you?` | ~540 | 9.4 s | 31 | 16.5 | 5.84 GB |
| `What is the capital of France?` | ~540 | 9.5 s | 25 | 15.9 | 5.85 GB |
| `Tell me about Rome.` | ~540 | 7.6 s | 48 | 15.6 | 5.85 GB |
| `What do you think of Siri?` | ~540 | 7.1 s | 41 | 15.6 | 5.85 GB |
| `How does a car engine work?` | ~540 | 8.3 s | 41 | 15.6 | 5.85 GB |

- Typical first-token latency: **~7–9.5 s**.
- Typical decode throughput: **~15.5–16.5 tok/s**.
- Benchmark total wall time: **~49 s** for the 5-prompt suite.
- Peak memory: **~5.8–5.9 GB**.

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
> 5. Run multi-step agent tasks and capture ambient context.
> 6. Engage the kill switch and the air-gap hard switch.
> 7. Pre-download models, switch personas, run benchmarks, and stream JSON.
> 8. Speak responses through the local TTS server and integrate with macOS Shortcuts and Siri.
> 9. Show a local web dashboard / control center at http://127.0.0.1:8787.
> 10. Sync with other Bad Apple peers over P2P — off by default.
> Everything stays on your Mac.

### 1. Core AI
1. Answer questions, explain, summarize, brainstorm, and write short notes
2. Run a local 9B Qwen 3.5 brain and a 0.5B fast tier on Apple Silicon
3. Switch persona at runtime: `switch to default`, `wicket`, `genz`, `drill`, `midwest`
4. `switch to roast` or `--roast` for the drill persona
5. `teach <line>` to store a custom quip
6. Multi-turn conversation with local JSONL history
7. Hot-reloadable system prompt via `prompt.txt`

### 2. Memory and knowledge
1. Working memory / scratchpad
2. Persistent user memory graph and episodic recall
3. Semantic cache for repeated questions
4. Local document indexing and RAG search
5. Workspace / project context

### 3. Voice and interface
1. Text input via the Rust CLI
2. Voice mode with STT / "hey bad apple" wake
3. Local Piper neural TTS with `--speak`
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
2. Autopilot mode to skip approvals
3. Private mode to pause persistence
4. Kill switch / emergency stop
5. Air-gap hard switch to block network MCP and downloads
6. Output firewall with blocklist patterns
7. Hash-chained audit ledger with redaction
8. SLICKS Unix socket authentication with HMAC

### 6. Mesh and networking
1. P2P encrypted LAN sync (off by default)
2. Peer discovery and sync beacons

### 7. Model and memory management
1. Model manager with cache status
2. One-click model pre-download with memory check
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
- **Voice (Piper / Apple)** and **Accent** — TTS engine and voice selection
- **Quit**

## Privacy & security

- Prompts and responses never leave the machine during normal use.
- Tool actions (file search, AppleScript, app open) run locally.
- Conversation history and user memory are stored locally, not synced.
- Client-to-daemon traffic is over a Unix socket with SLICKS HMAC challenge/response.
- The launchd daemon runs as root and auto-restarts.
- Output firewall blocks PII, secrets, and custom patterns.
- Hash-chained audit ledger with redaction at `/var/lib/bad_apple/ledger.jsonl`.
