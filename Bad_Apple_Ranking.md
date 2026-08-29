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

**Current score: 8.0/10**

| Category | Score | Rationale |
|---|---|---|
| Packaging & distribution | 2.5 / 3 | Unsigned full-release zip, drag-to-Applications DMG with `Install.command`, and a Homebrew Cask formula are in place. A signed/notarized path exists but is not the default artifact. |
| Installation UX | 1.5 / 2 | DMG `Install.command` and `brew install --cask bad-apple` are close to one-click, but both still require administrator approval and a quarantine strip for the unsigned app. |
| First-run experience | 1.75 / 2 | Lazy startup with `BADAPPLE_LAZY_MAIN_MODEL=1` and fast tier means simple queries work instantly. The `badapple_model_manager.py` + `web/models.html` checklist lets users pre-download 9B, 0.5B, and vision models before first use; FLUX is tracked but still downloads on first image generation. |
| QA & reliability | 1.75 / 2 | `cargo fmt`, `cargo build --release`, `cargo clippy`, `cargo audit` (0 vulns), `ruff`, `compileall`, and 171 unit tests all pass. Full air-gap cert suite passes on a live daemon. Smoke tests still require a running daemon; no VM install test. |
| Security & trust posture | 1.5 / 2 | Strong internal controls: SLICKS v2 with Secure Enclave, human-in-the-loop approvals, streaming output firewall, hash-chained audit ledger with SE-signed checkpoints (now correctly verified against the checkpoint's recorded position), 5 real security vulnerabilities found and fixed this pass (scheduler shell bypass, weak nonce, P2P bind-all, 2 symlink-following bugs, 4 cryptography CVEs), `cryptography` upgraded to 50.0.1. Unsigned consumer package still means a Gatekeeper warning for first-time users. |

### What moved the needle from 7.5 → 8.0

1. **Full security audit pass** — 5 real, independently-verified security vulnerabilities found and fixed with regression tests: scheduler `shell=True` bypass that let LLM-invoked scheduled tasks execute arbitrary shell commands; non-cryptographic nonce in SLICKS handshake; P2P mesh binding to `0.0.0.0` in contradiction of its "local only" threat model; two TOCTOU symlink-following bugs (Swift + Python); 4 CVEs in `cryptography` (46.0.7 → 50.0.1).
2. **2 Rust dependency CVEs fixed** — `quick-xml` DoS vulnerabilities (RUSTSEC-2026-0195, RUSTSEC-2026-0194) resolved by upgrading to 0.41.0. `cargo audit` now reports 0 vulnerabilities.
3. **Cert suite false positive fixed** — The Secure Enclave checkpoint verification was broken on any live system: it compared the checkpoint against the *current* ledger tip instead of the chain state at the checkpoint's recorded position. Fixed with `verify_checkpoint_in_ledger()`, 4 new tests.
4. **Real production bug fixed** — The gatekeeper was logging `EINVAL` on ~16% of health-check probes due to a benign socket disconnect race. Now correctly classified.
5. **Graceful shutdown** — MLX daemon now handles SIGTERM/SIGINT properly, eliminating semaphore leaks and orphaned MCP processes on `launchctl unload`.
6. **Code quality** — All clippy findings (default + pedantic) triaged and fixed. Ruff clean across 100 Python files. 171 tests (up from 52), all passing.
7. **Memory hardening** — MLX memory ceiling capped to Apple's recommended working set (11.8 GB) for resilience under memory pressure.

### Remaining blockers to 9+

- Signed/notarized `.dmg` and `.zip` as the default release artifact (eliminates Gatekeeper warning).
- Full FLUX pre-download in the model manager (the current `mflux` path still triggers its own cache download on first image generation).
- A clean-machine VM install + smoke test to verify the DMG and Cask end-to-end.
- Native onboarding/purchase-grade first launch in the Swift menu bar app.
- 3 unmaintained transitive Rust dependencies (`atty`, `fxhash`, `instant`, `paste`) with no safe upgrade path — explicitly triaged in `deny.toml` but worth monitoring.

---

## Competitive Ranking (August 2026)

The on-device macOS AI assistant market is now crowded. Below is an honest,
evidence-based comparison against the direct competitors, based on public
READMEs, feature lists, and install paths as of August 2026.

### The field

| # | Product | Model | Signed? | Native UI | Security controls | Air-gap cert | Open source |
|---|---|---|---|---|---|---|---|
| 1 | **M1K3** | Apple FM + Qwen 3 4B + Gemma 4 12B | Yes (Developer ID) | SwiftUI full app | App Sandbox, on-device only | No | Apache-2.0 |
| 2 | **Ka1zen** | Any MLX/GGUF model, speculative decoding | Yes | SwiftUI chat | On-device only | No | Proprietary (free) |
| 3 | **MLX Studio** | MLX, multi-model, vision, image gen | Yes | SwiftUI all-in-one | On-device only | No | Proprietary (free) |
| 4 | **macMLX** | Swift-native MLX engine, no Python | Yes | SwiftUI + CLI | On-device only | No | Open |
| 5 | **Bad Apple** | 9B Qwen 3.5 4-bit + 0.5B fast tier | No (unsigned) | Menu bar + CLI + web dashboard | SLICKS v2 SE, audit ledger, output firewall, approvals | **Yes** | Proprietary |
| 6 | **Macaw** | 2.7B LFM2.5 fine-tune, 97 tools | No | Menu bar + prompt bar | On-device, explicit consent | No | MIT |
| 7 | **iClaw** | Apple Intelligence or Ollama | Yes (App Store) | SwiftUI | App Sandbox, explicit consent | No | Open |
| 8 | **Ollama** | Any GGUF/MLX, API server | Yes | CLI + GUI | None (it's a server) | No | MIT |
| 9 | **LM Studio** | GGUF + MLX, model browser | Yes | Electron GUI | None (it's a runner) | No | Proprietary (free) |
| 10 | **mlx-serve (Loki)** | MLX, wake word, Telegram, schedules | Yes | Menu bar + launcher | On-device only | No | Proprietary |

### Where Bad Apple wins

1. **Security posture — uncontested.** No competitor has a hash-chained audit
   ledger with Secure Enclave-signed checkpoints, a streaming output firewall
   with real-time secret redaction, SLICKS v2 hardware-bound authentication,
   model provenance with SE signing, or a formal air-gap certification suite.
   This is not a marginal difference — it is a category-defining one. For
   government, enterprise, legal, medical, or anyone who needs *provable*
   privacy and auditability, Bad Apple is the only option in this list.

2. **Human-in-the-loop approvals.** Shared with iClaw and Macaw, but Bad
   Apple's implementation is the most mature: a declarative policy engine
   (`policy.yaml` with 37 tool rules), per-tool approval gates, autopilot
   mode toggle, and a kill switch / safe mode / private mode control surface.

3. **P2P encrypted model transfer.** Unique — no competitor lets you sync
   models between your own machines over a local encrypted mesh without a
   cloud download.

4. **Persona system.** Unique entertainment/branding angle. No competitor
   has switchable, teachable personality packs. This is a legitimate
   consumer differentiator, not just a gimmick — it makes the assistant
   feel like a character, not a tool.

5. **Rust CLI client.** Most competitors are Swift-only or Python-only. Bad
   Apple's Rust CLI talks to a Python daemon through a Rust gatekeeper over
   SLICKS-secured Unix sockets — a more robust, language-separated
   architecture than a monolithic app.

6. **Test coverage and code quality.** 171 tests (including red-team,
   security regression, and path-traversal tests), `cargo audit` clean,
   `ruff` clean, `cargo clippy` clean. Most competitors do not publish
   their test counts or run security audits.

### Where Bad Apple loses

1. **Signing and notarization — the biggest gap.** M1K3, Ka1zen, MLX Studio,
   macMLX, iClaw, and mlx-bun all ship signed/notarized apps. Bad Apple's
   default artifact is unsigned, which means every user sees a Gatekeeper
   warning and must run `strip_quarantine.sh`. This is the single biggest
   consumer-readiness blocker and the reason the security score is 1.5/2
   instead of 2.0/2.

2. **Native chat UI.** M1K3, Ka1zen, MLX Studio, and macMLX all have
   polished SwiftUI chat windows. Bad Apple's primary UI is a menu bar
   icon + CLI + web dashboard. The web dashboard is functional but not a
   first-class native chat experience. This matters for consumer adoption.

3. **Model flexibility.** Ka1zen runs any MLX or GGUF model with
   speculative decoding, vision, and image generation in one app. LM Studio
   and Ollama let you swap models in one command. Bad Apple is built around
   a single 9B Qwen 3.5 model with an optional 0.5B fast tier. The model
   is pinned, not user-selectable.

4. **Model size / hardware floor.** Bad Apple's 9B model needs ~6 GB of
   unified memory, making 16 GB the practical floor. Macaw runs on a 1.5 GB
   model, iClaw uses Apple's built-in Foundation Models (zero download), and
   mlx-bun starts with a sub-GB model. Bad Apple is heavier.

5. **Call transcription.** M1K3 and LokalBot both offer encrypted on-device
   call transcription. Bad Apple does not.

6. **Development velocity.** M1K3 has 1,286 commits; Bad Apple has 302.
   M1K3 is a more actively developed project with a TestFlight beta
   distribution channel. Bad Apple is more mature in its security
   engineering but narrower in feature surface.

7. **No Python-free path.** macMLX ships a ~50 MB pure-Swift app with no
   Python dependency. Bad Apple requires a Python venv with `mlx-lm`,
   `piper-tts`, `cryptography`, and other packages. This is a real
   installation friction point.

### Honest overall placement

**Bad Apple ranks #5 out of 10 in the direct competition**, but this
average hides a bimodal distribution:

- **For security-first / air-gapped / auditable use cases (government,
  enterprise, legal, medical, journalists):** Bad Apple is **#1**. No
  competitor has an audit ledger, SE-signed checkpoints, an air-gap cert
  suite, a streaming output firewall, or hardware-bound authentication.
  This is not a feature checklist — it is a category Bad Apple invented.

- **For general consumer / "I want a nice local ChatGPT replacement":**
  Bad Apple is **#6-7**. M1K3, Ka1zen, MLX Studio, macMLX, and iClaw all
  offer a more polished, signed, native-chat experience with lower
  friction. Bad Apple's menu-bar-plus-CLI UX and unsigned artifact make
  it a harder sell to a non-technical user.

- **For developers who want a local model server / API:** Bad Apple is
  **not competitive**. Ollama and LM Studio are the standard here, with
  100K+ stars and tens of thousands of integrations. Bad Apple is an
  assistant, not a model server.

The path to #1 overall is clear: **ship a signed/notarized DMG as the
default artifact, and build a native SwiftUI chat window.** Those two
changes would close the consumer gap without sacrificing the security
lead. The security architecture is already best-in-class — it just needs
a consumer-grade front door.

---

## License

LicenseRef-Proprietary. See `LICENSE.txt`.
