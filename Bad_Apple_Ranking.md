# Bad Apple — Complete Feature & Capability Overview

> **Bad Apple is an on-device, air-gapped AI assistant for macOS that runs a 9B Qwen 3.5 large language model on Apple Silicon, executes local tools, indexes your files, speaks responses, and never sends prompts or data to the cloud after the models are downloaded once.**

---

## What Bad Apple Is

Bad Apple is a **local-first generative AI runtime** for macOS. It is built around a single 9B Qwen 3.5 4-bit model running on the Apple Neural Engine / GPU through MLX, an optional small speculative draft model for faster generation, a local neural TTS server, and a Rust-backed SLICKS-secured Unix-socket command layer. It is designed for users who want the conversational power of a frontier chatbot with the privacy and latency of on-device inference.

Unlike cloud-based assistants (Siri, ChatGPT, Gemini, Copilot), Bad Apple:

- Runs entirely on your Mac after first download.
- Keeps every prompt, response, tool call, and document index on the machine.
- Works without a subscription, API key, or network round-trip at runtime.
- Can be extended with personas, custom quips, local tools, and an audit trail.

---

## Core Capabilities

### 1. On-Device Language Reasoning

- **9B Qwen 3.5 target model** (`caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit`) for general question answering, summarization, writing, coding help, and open-ended chat.
- **Optional speculative decoding** — when a small compatible draft model (e.g. `mlx-community/Qwen2.5-0.5B-Instruct-4bit`) is cached, `mlx-lm` can run speculative decoding for faster generation.
- **Single model for text and voice** — no multi-second model swap when switching from text to speech mode.
- **Streaming output** — tokens are emitted as they are generated and can be displayed, saved, or sent to TTS in real time.

### 2. Voice and Neural Text-to-Speech

- **Piper TTS server** (`badapple_tts_server.py`) synthesizes speech locally with models like `en_US-amy-medium`.
- **Menu bar voice host** listens for voice prompts and speaks answers using on-device speech recognition and the bundled `badapple` helper.
- **TTS pacing queue** in the CLI streams sentence chunks to a background worker so the model is not blocked waiting for audio playback.
- **Multiple TTS voices and accents** selectable from the menu bar and via `BADAPPLE_TTS_VOICE`.

### 3. Persona Pack System

- `personas.json` defines switchable personalities: **Default** (sassy California beach girl), **Wicket** (witty Londoner), **Gen Z Hype**, **Drill Rapper**, and **Midwest Aunt**.
- Runtime switching with `switch to <persona>` or `BADAPPLE_PERSONA=<name>`.
- CLI flags `--persona <name>` and `--roast` (alias for `drill`) let a single query use a different persona.
- **Teachable quips** — `teach <line>` adds custom lines to the active persona's banter bank and persists them.
- Rotating roast targets and insults for Siri, Alexa, Google, ChatGPT, Gemini, Cortana, Bixby, and cloud server farms.

### 4. Local Tool Use and Action Execution

Bad Apple can run tools against the local filesystem and system without leaving the Mac:

- `get_current_time` — local system time
- `list_directory` — list files in a path
- `read_file` — read a file with size limits
- `write_file` — write notes to `~/.bad_apple/notes/`
- `search_content` — `grep -R` over a directory
- `search_local_files` — Spotlight search via `mdfind`
- `run_shell` — sandboxed shell command
- `run_applescript` — execute AppleScript
- `index_documents` — index a directory into the local RAG store

### 5. Human-in-the-Loop Approval Workflow

Destructive tools (`run_shell`, `run_applescript`, `write_file`, `index_documents`) are **proposed, not executed**. The user must approve each one with `approve <id>` unless `BADAPPLE_AUTOPILOT=1` is set. This keeps local AI from silently modifying files or running arbitrary commands.

### 6. Retrieval-Augmented Generation (RAG) and Memory

- **Local document index** with `BAAI/bge-small-en-v1.5` embeddings on CPU.
- `index_documents` adds directories or files to the RAG store.
- The model retrieves relevant chunks and cites them in answers.
- **Persistent user memory** records facts the user mentions and recalls them in later turns.
- **Multi-turn conversation history** saved to local JSONL.

### 7. Semantic Cache

- `bge-small-en-v1.5` encodes incoming queries and classifies intent.
- Common questions are answered instantly from a local persona-scoped cache without re-running the 9B model.
- No caching for tool queries or voice prompts, ensuring fresh local actions.

### 8. Streaming Output Firewall

- Aho-Corasick automaton scans generated tokens and final text in real time.
- Default patterns block secrets (SSH keys, private keys, API keys like `sk-...`).
- Custom blocklist via `BADAPPLE_BLOCKLIST` file or `/var/lib/bad_apple/blocklist.txt`.
- Blocked output is redacted before it reaches the user.

### 9. Hash-Chained Audit Ledger

- Every query, tool call, response, approval, and cache hit is appended to `/var/lib/bad_apple/ledger.jsonl`.
- Each record links to the previous record with a SHA-256 `prev_hash` and a keyed HMAC.
- Secrets and PII are redacted before writing.
- Ledger can be verified with `AuditLedger('/var/lib/bad_apple').verify()`.

### 10. Security and Authentication

- **SLICKS** — HMAC-SHA256 challenge-response with client/server nonces and prompt binding.
- **Unix-domain sockets** only; no runtime network listeners.
- **launchd-managed daemons** run as root and auto-restart.
- **Fail-closed file paths** — writes restricted to `BADAPPLE_NOTES_DIR` unless in autopilot.
- **Local-only audio** — TTS synthesis is fully on-device.

### 11. CLI, Menu Bar, and Benchmarking

- **`badapple` CLI** (`target/release/badapple`) is the authenticated Rust client.
- **Menu bar app** (`Bad Apple.app`) lives in the status bar, supports voice wake, persona switching, roast mode, and benchmarking.
- **Benchmark mode** (`--benchmark`) runs a standard prompt suite and reports tokens, TTFT, decode tok/s, total tok/s, and peak memory.
- **JSON streaming** (`--json`) emits every token and the final `done` frame with metrics for integrations.

---

## Performance

Measured on a 16 GB Apple Silicon M-series Mac with the 9B Qwen 3.5 4-bit target and an optional small draft model:

| Metric | Typical Range |
|---|---|
| First-token latency | 3.5–6.5 s for 450–750 token prompts |
| Decode throughput | 13–25 tok/s, spikes to ~36 tok/s |
| Peak memory | 5.7–6.5 GB with target and optional draft loaded |
| Voice first token | 2.8–5.7 s for 430–460 token prompts |
| RAG embeddings | bge-small on CPU |

---

## Use Cases

- **Private Q&A** — ask questions, get summaries, write drafts without sending data to a cloud API.
- **Local coding assistant** — explain code, generate snippets, search local projects.
- **Voice desktop assistant** — ask for the time, open workspaces, run local tools by speaking.
- **Personal knowledge base** — index your notes and documents and ask questions against them.
- **Air-gapped workflows** — use on machines or networks with no external access after first setup.
- **Persona-driven interaction** — switch between entertaining personas for different moods or demos.

---

## Architecture

```text
badapple CLI / menu bar / voice host
              │
              ▼
   /var/run/badapple/substrate.sock  (SLICKS)
              │
              ▼
     gatekeeper (Rust, launchd)
              │
              ▼
   badapple_mlx_server.py  (Python + MLX)
   ├─ 9B Qwen 3.5 + optional small draft
   ├─ RAG / embeddings
   ├─ personas, cache, firewall, audit
   └─ tools + approvals
              │
              ▼
    badapple_tts_server.py  (Piper TTS)
```

---

## Why Bad Apple Stands Out

- **True local inference** — no API calls or telemetry after the first model download.
- **Persona-driven** — switchable, teachable personalities make the assistant entertaining and brandable.
- **Built-in safety** — approvals, fail-closed paths, streaming firewall, and audit ledger by default.
- **Mac-native** — uses MLX, Apple Silicon, launchd, Piper TTS, and a Swift/Objective-C menu bar.
- **Fast speculative decoding** — when a small cached draft is available, speculative decoding can give better throughput than standard token-by-token generation on Qwen 3.5.
- **Extensible local RAG** — index your own files and query them privately.
- **Open-ended tool use** — local shell, AppleScript, and file tools gated by user approval.
- **Benchmark-ready** — built-in metrics for throughput, latency, and memory.

---

## Quick Start

```bash
# Build
cargo build --release

# Install daemons
sudo cp src/platform/apple_bridge/com.badapple.gatekeeper.plist /Library/LaunchDaemons/
sudo cp src/platform/apple_bridge/com.badapple.mlx.plist /Library/LaunchDaemons/
sudo cp src/platform/apple_bridge/com.badapple.tts.plist /Library/LaunchDaemons/
sudo launchctl load -w /Library/LaunchDaemons/com.badapple.gatekeeper.plist
sudo launchctl load -w /Library/LaunchDaemons/com.badapple.mlx.plist
sudo launchctl load -w /Library/LaunchDaemons/com.badapple.tts.plist

# First prompt
target/release/badapple "What time is it?"
target/release/badapple -n 240 "Write me a poem about bare metal"

# Voice
target/release/badapple --speak "What do you think of Siri?"

# Persona / roast
target/release/badapple --persona wicket "Who are you?"
target/release/badapple --roast "Tell me about cloud AI"

# Benchmark
target/release/badapple --benchmark

# Menu bar
src/platform/apple_desktop/build_bad_apple_menu_bar.sh
open -a "Bad Apple"
```

---

## Consumer Readiness Ranking

**Current score: 7.5/10**

| Category | Score | Rationale |
|---|---|---|
| Packaging & distribution | 2.5 / 3 | Unsigned full-release zip, drag-to-Applications DMG with `Install.command`, and a Homebrew Cask formula are in place. A signed/notarized path exists but is not the default artifact. |
| Installation UX | 1.5 / 2 | DMG `Install.command` and `brew install --cask bad-apple` are close to one-click, but both still require administrator approval and a quarantine strip for the unsigned app. |
| First-run experience | 1.75 / 2 | Lazy startup with `BADAPPLE_LAZY_MAIN_MODEL=1` and fast tier means simple queries work instantly. The new `badapple_model_manager.py` + `web/models.html` checklist lets users pre-download 9B, 0.5B, and vision models before first use; FLUX is tracked but still downloads on first image generation. |
| QA & reliability | 1.75 / 2 | `cargo fmt`, `cargo build --release`, `ruff`, `compileall`, and 52 unit tests pass. Dashboard has new `/api/models` endpoints and the models page is wired. Smoke tests still require a running daemon; no VM install test. |
| Security & trust posture | 1.0 / 2 | Strong internal controls (SLICKS, approvals, audit ledger) and an unsigned consumer package means first-time users see a Gatekeeper warning. `package_signed_release.sh` + `SIGNING.md` document the notarized path. |

### What moved the needle from 6 → 7.5

1. **DMG installer** — `src/platform/apple_desktop/package_dmg.sh` produces a consumer `.dmg` with `Bad Apple.app`, `Applications` alias, and `Install.command` that copies the app and installs the platform LaunchDaemons.
2. **Homebrew Cask** — `homebrew-bad-apple/Casks/bad-apple.rb` and `src/platform/apple_desktop/package_homebrew_cask.sh` provide a `brew install --cask bad-apple` path.
3. **Lazy 9B loading + fast 0.5B tier** — `BADAPPLE_LAZY_MAIN_MODEL=1` and `badapple_mlx_server._ensure_main_model()` defer the heavy model until first use. Fast-tier queries no longer require a loaded 9B brain.
4. **Background model manager and dashboard checklist** — `badapple_model_manager.py` tracks `main_9b`, `fast_0.5b`, `vision_2b`, and `flux_4b` with download/load status. `web/models.html` gives users an allow-downloads toggle, download buttons, and progress for each model, and the onboarding wizard now points to it.

### Remaining blockers to 8+

- Signed/notarized `.dmg` and `.zip` as the default release artifact.
- Full FLUX pre-download in the model manager (the current `mflux` path still triggers its own cache download on first image generation).
- A clean-machine VM install + smoke test to verify the DMG and Cask end-to-end.
- Native onboarding/purchase-grade first launch in the Swift menu bar app.

---

## License

LicenseRef-Proprietary. See `LICENSE.txt`.
