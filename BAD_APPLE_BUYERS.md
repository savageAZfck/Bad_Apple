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

> I can answer questions, run local tools, search files, write notes, run shell/AppleScript/Shortcuts, index documents, manage working memory, switch personas, speak, stream JSON, and run benchmarks — all on your Mac, babe.

### Core capabilities

- Answer questions, explain concepts, summarize text, brainstorm, write short notes
- Roast cloud AI / Siri / Alexa / Google / ChatGPT / Gemini / etc. when asked about bare metal or identity
- Switch persona at runtime: `switch to wicket`, `switch to drill`, `switch to genz`, `switch to midwest`, `switch to default`
- `switch to roast` or `--roast` for drill persona
- `teach <line>` to store a custom quip
- `--speak` / voice mode to stream responses through local Piper TTS
- `--benchmark` to run the prompt suite
- `--json` to stream tokens as JSON
- Multi-turn conversation with local JSONL history
- Persistent user memory and RAG over indexed local documents
- Hot-reloadable system prompt via `prompt.txt`

### Local tools (no cloud)

- `get_current_time` — local system time
- `list_directory` — list files in a path
- `read_file` — read a file with size limit
- `write_file` — write a note to `~/.bad_apple/notes/`
- `search_content` — `grep -R` over a directory
- `search_local_files` — Spotlight search via `mdfind`
- `run_shell` — sandboxed shell (read-only by default: `ls`, `cat`, `head`, `tail`, `find`, `grep`, `wc`, `file`, `pwd`, `mdfind`, `ps`, `df`, `du`)
- `run_applescript` — execute AppleScript
- `run_shortcut` — run a macOS Shortcuts shortcut
- `index_documents` — index a directory into the local RAG store
- `git_status`, `git_diff`, `git_log`, `git_commit` — local git helpers
- `read_working_memory`, `write_working_memory`, `clear_working_memory` — scratchpad at `/var/lib/bad_apple/working_memory.txt`

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
