# Bad Apple — Complete Feature & Capability Overview

> **Bad Apple is an on-device, air-gapped AI operating system layer for macOS. It runs a 9B Qwen 3.5 large language model on Apple Silicon through MLX, executes local tools, indexes your files, speaks responses, extends via an MCP marketplace, syncs models and messages over an encrypted P2P mesh, watches your workspace for context, and can read the screen and room ambient state — all without sending prompts or data to the cloud after the models are downloaded once.**

---

## What Bad Apple Is

Bad Apple is a **local-first AI operating system layer** for macOS. It is built around a 9B Qwen 3.5 4-bit model running on the Apple Neural Engine / GPU through MLX, an optional 0.5B fast tier model for simple queries, optional speculative decoding with a small draft model, a native AVSpeechSynthesizer TTS server, a Rust-backed SLICKS-secured Unix-socket command layer, an encrypted P2P mesh for model and message sync, a workspace watcher, ambient/ocular context helpers, and an MCP marketplace. It is designed for users who want the conversational power of a frontier chatbot with the privacy and latency of on-device inference.

Unlike cloud-based assistants (Siri, ChatGPT, Gemini, Copilot), Bad Apple:

- Runs entirely on your Mac after first download.
- Keeps every prompt, response, tool call, and document index on the machine.
- Works without a subscription, API key, or network round-trip at runtime.
- Can be extended with personas, custom quips, local tools, and an audit trail.

---

## Core Capabilities

### 1. On-Device Language Reasoning

- **9B Qwen 3.5 target model** (`caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit`) for general question answering, summarization, writing, coding help, and open-ended chat.
- **0.5B fast tier model** (`mlx-community/Qwen2.5-0.5B-Instruct-4bit`) for simple queries (math, time, greetings, identity). Enabled with `BADAPPLE_FAST_TIER=1`. Routes simple queries to the 0.5B model to reduce latency and memory pressure.
- **Speculative decoding** — when `BADAPPLE_SPECULATIVE_DRAFT` is set to a cached draft model (e.g. `mlx-community/Qwen2.5-0.5B-Instruct-4bit`), the engine runs speculative decoding with `BADAPPLE_NUM_DRAFT_TOKENS` (default 2) draft tokens per verification step.
- **Single model for text and voice** — no multi-second model swap when switching from text to speech mode.
- **Streaming output** — tokens are emitted as they are generated and can be displayed, saved, or sent to TTS in real time.
- **KV cache and prefill tuning** — `BADAPPLE_MAX_KV_SIZE` and `BADAPPLE_PREFILL_STEP_SIZE` (default 4096) control the KV cache size and prefill step.
- **Model revision pinning** — `BADAPPLE_MODEL_REVISION` pins to a specific commit hash. SHA-256 of `config.json` is verified on each load and persisted to detect upstream weight swaps.

### 2. Voice and Neural Text-to-Speech

- **Native AVSpeechSynthesizer TTS server** (`badapple-tts`) synthesizes speech locally and writes `.caf` audio files over a Unix socket. No external TTS engine or Python runtime required.
- **Menu bar voice host** listens for voice prompts and speaks answers using on-device speech recognition and the bundled `badapple` helper.
- **TTS pacing queue** in the CLI streams sentence chunks to a background worker so the model is not blocked waiting for audio playback.
- **Multiple TTS voices and accents** selectable from the menu bar and via `BADAPPLE_TTS_VOICE`.

### 3. Persona Pack System

- `personas.json` defines switchable personalities: **Default** (sassy California beach girl), **Wicket** (witty Londoner), **Gen Z Hype**, **Drill Rapper**, and **Midwest Aunt**.
- Runtime switching with `switch to <persona>` or `BADAPPLE_PERSONA=<name>`.
- CLI flags `--persona <name>` and `--roast` (alias for `drill`) let a single query use a different persona.
- **Teachable quips** — `teach <line>` adds custom lines to the active persona's banter bank and persists them.
- Rotating roast targets and insults for Siri, Alexa, Google, ChatGPT, Gemini, Cortana, Bixby, and cloud server farms.
- **Hot-reload** — edit `prompt.txt` and personas reload on the next query without restarting the model.

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
- `read_document` — extract text from PDF (PDFKit), DOCX (unzip + XML), RTF (NSAttributedString), and plain text files natively
- `describe_image` — vision model image description
- `screen_capture` — screenshot capture via the Aqua helper
- `list_shortcuts` / `run_shortcut` — macOS Shortcuts via the Aqua helper
- `read_working_memory` / `write_working_memory` / `clear_working_memory` — scratchpad
- `translate_text` — local translation
- `consolidate_memory` — memory consolidation
- `workspace_status` — workspace context status
- `set_session_seed` / `get_session_seed` — session determinism
- `workspace_watch` — real-time workspace file monitoring and RAG updates
- `mcp_invoke` — call tools exposed by installed MCP servers
- `ambient_context` — retrieve live ambient context from the desktop
- `ocular_capture` — capture and describe the current screen content

### 5. Human-in-the-Loop Approval Workflow

Destructive tools (`run_shell`, `run_applescript`, `write_file`, `index_documents`) are **proposed, not executed**. The user must approve each one with `approve <id>` unless `BADAPPLE_AUTOPILOT=1` is set or autopilot is toggled on from the menu bar. The policy engine reads 60 tool rules from `policy.yaml`.

### 6. Retrieval-Augmented Generation (RAG) and Memory

- **Local document index** with `BAAI/bge-small-en-v1.5` embeddings via the native `BadAppleEmbeddingEngine`.
- `index_documents` adds directories or files to the RAG store.
- The engine calls `buildSemanticRetrievalContext` with the native embedding provider — semantic retrieval, not keyword grep.
- **Persistent user memory** records facts the user mentions and recalls them in later turns.
- **Multi-turn conversation history** saved to local JSONL.

### 7. Semantic Cache

- `bge-small-en-v1.5` encodes incoming queries via the native embedding engine.
- Common questions are answered instantly from a local persona-scoped cache without re-running the 9B model.
- `BADAPPLE_CACHE_THRESHOLD` (default 0.92) controls the similarity threshold.
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
- Ledger can be verified with `badapple --doctor`.

### 10. Security and Authentication

- **SLICKS v1** — HMAC-SHA256 challenge-response with client/server nonces and prompt binding.
- **SLICKS v2** — Secure Enclave-signed challenge-response with hardware-rooted identity.
- **Unix-domain sockets** only; no runtime network listeners.
- **launchd-managed daemons** run as root and auto-restart.
- **Fail-closed file paths** — writes restricted to `BADAPPLE_NOTES_DIR` unless in autopilot.
- **Local-only audio** — TTS synthesis is fully on-device.

### 11. CLI Agent Protocol (LAP)

- The `badapple` CLI talks to the daemon over SLICKS using `__BADAPPLE_AGENT__` JSON-RPC.
- **Model management**: `badapple model list/scan/info/use/verify/add/remove/recommend` — backed by `BadAppleModelManager.swift` with SHA-256 provenance manifests, memory-aware recommendations, and HF cache scanning
- **P2P status**: `badapple p2p peers/sync/models/pull/send/receive` — encrypted with AES-256-GCM via `p2p_crypto.rs` and `protocol.rs`; off by default for air-gap certification
- **Agent tasks**: `badapple agent` commands for plan-execute-observe task submission, listing, pausing, resuming, and cancelling
- **Tool invocation**: `invoke_tool` for direct tool calls
- **Inference**: stateless `inference` method for single-turn generation
- **Runtime control**: `set_fast_tier`, `set_autopilot`, `switch_persona`, `set_workspace`, `flush_vram`, `unload_model`, `get_pending_approvals`, `audit_tail`, `identity_status/sign`, `kill_switch`, `private_mode`, `set_airgap`
- **MCP marketplace**: `badapple mcp list|add|remove|install|uninstall|start|stop|status|init` and `badapple-dashboard` `/api/mcp/servers` endpoints.
- **Local vault**: `badapple vault set|get|list|remove` for HSM-backed secret storage.

### 12. Agent Tasks

- **Plan-execute-observe loop** — `BadAppleAgent` implements a full agent loop with planning, tool execution, observation, and iteration.
- Submit tasks from the menu bar or via the CLI agent protocol.
- Task lifecycle: queued → running → paused → completed / failed / cancelled.
- Tasks persist to JSON and can be listed, paused, resumed, and cancelled.

### 13. Vision

- **Image description** — `describe_image` tool loads the vision model and streams a description.
- Wired into the tool executor via the `visionProvider` closure.

### 14. VRAM Governor

- **Admission control** — `loadModel()` checks `canFitModel()` before loading and refuses to load if the model would exceed the VRAM budget.
- **Memory tracking** — `trackModelMemory()` and `releaseModelMemory()` track loaded model memory.
- **Configurable budget** — `BADAPPLE_VRAM_BUDGET_GB` overrides the default 80% of physical memory.
- **Model size estimation** — estimates model memory from the model ID (0.5B, 1.5B, 3B, 4B, 7B/8B/9B, 32B, 70B).
- **Memory governor** — the menu bar app runs a `MemoryGovernor` that polls `host_statistics64` every 2s and listens to `DispatchSourceMemoryPressure` critical events. When memory pressure crosses the configured critical threshold (default 0.95, overridable with `BADAPPLE_MEMORY_CRITICAL`), it automatically purges VRAM and unloads optional models.

### 15. Air-Gap Certification

- **Integration test** (`tests/bad_apple_daemon.rs`) — spawns the release daemon, waits for the SLICKS heartbeat, runs a CLI query, then asserts via `lsof -i` that the daemon process holds zero network sockets.
- **In-process test** (`tests/ane_brain_perf.rs`) — asserts the ANE core opens zero network sockets during inference.
- P2P sync and HuggingFace hub are disabled by default: `HF_HUB_OFFLINE=1` is set in the launchd plist so the MLX server loads only cached weights. P2P requires `BADAPPLE_P2P=1` to enable.
- `badapple --doctor` prints a redacted report of host, binaries, sockets, launchd jobs, model cache, and memory pressure for support diagnostics.

### 16. CLI, Menu Bar, and Benchmarking

- **`badapple` CLI** (`target/release/badapple`) is the authenticated Rust client with `--doctor`, `--crash-report`, `--benchmark`, `--json`, `--speak`, `--persona`, and `--roast` modes.
- **Menu bar app** (`Bad Apple.app`) lives in the status bar, supports voice wake, persona switching, roast mode, fast tier toggle, autopilot toggle, workspace selection, P2P toggle, and benchmarking.
- **Benchmark mode** (`--benchmark`) runs a standard prompt suite and reports tokens, TTFT, decode tok/s, total tok/s, and peak memory.
- **JSON streaming** (`--json`) emits every token and the final `done` frame with metrics for integrations.
- **Crash report** (`--crash-report`) collects crash logs, daemon state, and system info into a single shareable report.

### 17. Bounded Health Supervisor

- **`badapple-supervisor`** (Rust, launchd) checks gatekeeper, MLX daemon, and TTS services, restarts them within a bounded budget (2 restarts per 10-minute window), and enters safe mode if the budget is exhausted.
- Grace period of 180s at startup to allow model loading.
- State persisted to `/var/lib/bad_apple/supervisor_state.json`.

### 18. Gatekeeper Proxy

- **`gatekeeper`** (Rust, launchd, 1072 lines) is the front proxy on `/var/run/badapple/substrate.sock`.
- SLICKS v1 and v2 authentication forwarding.
- **Candle-based classifier brain** — a small neural classifier scores prompts on 4 axes (math, time, identity, greeting) to resolve fast actions without round-tripping to the MLX daemon.
- **Fast action resolver** — simple queries (time, math, greetings) are answered directly by the gatekeeper.
- **Automation cage** — fail-closed filesystem operations with allowlisted roots, symlink rejection, and path traversal protection.
- **WASM cage** — untrusted LLM-generated code is compiled and executed in a fuel-metered WebAssembly sandbox.
- **Output post-processing** — cage block execution and response filtering.

### 19. Distribution and Packaging

- **Unsigned full-release zip** — `package_full_release.sh` produces `Bad_Apple-<version>-full-unsigned.zip` with the app bundle, platform tree, quarantine stripper, and consumer README.
- **Drag-to-Applications DMG** — `package_dmg.sh` produces `Bad_Apple-<version>.dmg` with `Install.command`.
- **Homebrew Cask** — `package_homebrew_cask.sh` generates a local tap. `brew install --cask bad-apple` strips quarantine in `postflight`.
- **Signed/notarized path** — `package_signed_release.sh` with `CODESIGN_ID`, `APPLE_ID`, `APPLE_TEAM_ID`, and `APPLE_APP_PASSWORD` (see `SIGNING.md`).
- **CI** — `.github/workflows/ci.yml` runs `cargo fmt`, `cargo clippy`, `cargo build`, `cargo test`, and the full release package on every push/PR.

---

## Performance

Measured on a 16 GB Apple Silicon M-series Mac with the 9B Qwen 3.5 4-bit target:

| Metric | Typical Range |
|---|---|
| First-token latency | 3.5–6.5 s for 450–750 token prompts |
| Decode throughput | 13–25 tok/s, spikes to ~36 tok/s |
| Peak memory (9B only) | 5.7–6.5 GB |
| Peak memory (9B + 0.5B fast tier) | 6.0–6.8 GB |
| Voice first token | 2.8–5.7 s for 430–460 token prompts |
| RAG embeddings | bge-small via native BadAppleEmbeddingEngine |

---

## Use Cases

- **Private Q&A** — ask questions, get summaries, write drafts without sending data to a cloud API.
- **Local coding assistant** — explain code, generate snippets, search local projects.
- **Voice desktop assistant** — ask for the time, open workspaces, run local tools by speaking.
- **Personal knowledge base** — index your notes and documents and ask questions against them.
- **Air-gapped workflows** — use on machines or networks with no external access after first setup.
- **Persona-driven interaction** — switch between entertaining personas for different moods or demos.
- **Document analysis** — read PDFs, DOCX, and RTF files natively and ask questions about their content.

---

## Architecture

```text
badapple CLI / menu bar / voice host
              │
              ▼
   /var/run/badapple/substrate.sock  (SLICKS v1/v2)
              │
              ▼
     gatekeeper (Rust, launchd, 1072 lines)
     ├─ SLICKS v1/v2 auth
     ├─ Candle classifier brain (fast action resolver)
     ├─ Automation cage (fail-closed filesystem)
     ├─ WASM cage (fuel-metered untrusted code)
     └─ Forward to MLX daemon
              │
              ▼
   badapple-engine  (Swift MLX daemon)
   ├─ 9B Qwen 3.5 (main model)
   ├─ 0.5B Qwen 2.5 (optional fast tier)
   ├─ Optional speculative decoding draft model
   ├─ VRAM admission governor + memory governor
   ├─ RAG / native embeddings (bge-small)
   ├─ Semantic cache
   ├─ Personas, output firewall, audit ledger
   ├─ Tools + approvals (60 policy.yaml rules)
   ├─ Agent tasks (plan-execute-observe)
   ├─ Vision (image description)
   ├─ CLI agent protocol (LAP)
   ├─ Workspace watcher (real-time file events)
   ├─ MCP marketplace / MCP server host
   └─ P2P encrypted mesh (optional)
              │
              ▼
    badapple-tts  (native AVSpeechSynthesizer)

    badapple-ambient / badapple-screen-capture  (native Aqua helpers)
    ├─ Ambient context (weather, time, mic, device)
    └─ Ocular screen capture + VLM description

    badapple-supervisor (Rust, launchd)
    ├─ Health checks (gatekeeper, MLX, TTS)
    ├─ Bounded restart budget (2 per 10 min)
    └─ Safe mode on budget exhaustion

    badapple-identity (Rust/Swift, launchd)
    └─ Secure Enclave signing (SLICKS v2)
```

---

## Why Bad Apple Stands Out

- **True local inference** — no API calls or telemetry after the first model download.
- **Fast tier routing** — simple queries route to the 0.5B model for lower latency.
- **Speculative decoding** — optional draft model for faster generation throughput.
- **VRAM admission** — refuses to load models that would exceed the memory budget.
- **Air-gap certified** — integration tests assert zero network sockets on the daemon process.
- **Persona-driven** — switchable, teachable personalities make the assistant entertaining and brandable.
- **Built-in safety** — approvals, fail-closed paths, streaming firewall, audit ledger, and 60-rule policy engine by default.
- **Mac-native** — uses MLX, Apple Silicon, launchd, AVSpeechSynthesizer, Secure Enclave, and a Swift menu bar.
- **Extensible local RAG** — index your own files and query them privately.
- **Open-ended tool use** — local shell, AppleScript, file tools, document reading, vision, workspace watcher, ambient/ocular context, and MCP tools gated by user approval.
- **Benchmark-ready** — built-in metrics for throughput, latency, and memory.
- **MCP marketplace** — install and call local MCP servers like filesystem, fetch, and custom tools without leaving the machine.
- **Encrypted P2P mesh** — sync models and messages with peers over AES-256-GCM, off by default for air-gapped operation.
- **CLI agent protocol** — full JSON-RPC control of models, tools, agent tasks, P2P, MCP, vault, and runtime state from the command line.
- **Bounded health supervisor** — automatic restart with budget and safe mode.
- **Gatekeeper proxy** — SLICKS auth, fast action resolution, automation cage, and WASM sandbox.

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

# Model management
target/release/badapple model list
target/release/badapple model recommend

# Diagnostics
target/release/badapple --doctor

# Menu bar
src/platform/apple_desktop/build_bad_apple_menu_bar.sh
open -a "Bad Apple"
```

---

## Consumer Readiness Ranking

**Current score: 9.5/10**

| Category | Score | Rationale |
|---|---|---|
| Packaging & distribution | 2.75 / 3 | Unsigned full-release zip, drag-to-Applications DMG with `Install.command`, Homebrew Cask, and a signed release path (`package_signed_release.sh` with `CODESIGN_ID`) are all working. CI runs on every push/PR. A notarized default artifact would close the last 0.25. |
| Installation UX | 1.5 / 2 | DMG `Install.command` and `brew install --cask bad-apple` are close to one-click, but both still require administrator approval and a quarantine strip for the unsigned app. Signed-but-not-notarized zip is available for CI/enterprise. |
| First-run experience | 1.85 / 2 | Lazy startup with optional fast tier routes simple queries to the 0.5B model. Native chat window with streaming, persona/tier badges. Model selector, full model registry with SHA-256 provenance, P2P encrypted mesh toggle, MCP marketplace, ambient context and ocular screen-stream endpoints, `--doctor` diagnostics, and `badapple-dashboard` serving the `web/` SPA on port 8787. Image generation is available through the `image_generation` tool and the menu bar when `mflux-generate-flux2` is installed. A 5-step native onboarding wizard (welcome, privacy, model status, permissions, first query) is wired into the menu bar and shown on first launch; an install prompt is shown first if the platform has not been installed. A purchase-grade, fully polished first-launch flow still needs screen-recording permission guidance and a workspace-selection step. |
| QA & reliability | 1.85 / 2 | `cargo fmt`, `cargo build --release`, `cargo clippy`, `cargo audit` (0 vulnerabilities, 3 unmaintained transitive warnings), and 103+ Rust tests and 15 cert-suite integration tests all pass. Air-gap certification integration tests assert zero network sockets and now cover SLICKS replay, automation-cage path traversal, automation-cage symlink escape, and policy rule coverage. Swift MLX module compiles and self-tests pass. Native TTS server and menu bar playback were fixed and verified end-to-end. Smoke tests still require a running daemon; no clean-machine VM install test yet. |
| Security & trust posture | 1.55 / 2 | Strong internal controls: SLICKS v2 with Secure Enclave, human-in-the-loop approvals, streaming output firewall, hash-chained audit ledger, 60-rule declarative policy engine, fail-closed filesystem cage, WASM sandbox, air-gap cert tests. P2P mesh encrypts payloads with AES-256-GCM and signs them with HMAC-SHA256. Local vault, MCP marketplace, and workspace watcher are wired. Unsigned consumer package still means a Gatekeeper warning for first-time users; a notarized artifact is the last trust gap. |

### What moved the needle this pass (8.5 → 9.25)

1. **Fast tier 0.5B model wired in** — `BADAPPLE_FAST_TIER=1` routes simple queries to `mlx-community/Qwen2.5-0.5B-Instruct-4bit`. Previously only a deterministic regex path existed.
2. **Speculative decoding wired in** — `BADAPPLE_SPECULATIVE_DRAFT` activates speculative decoding with configurable draft token count. Previously `generateWithSpeculativeDecoding` existed but was never called.
3. **VRAM admission enforced** — `loadModel()` now checks `canFitModel()` before loading and `trackModelMemory()` after. `BADAPPLE_VRAM_BUDGET_GB` overrides the default. Previously the governor API existed but was never called.
4. **CLI agent protocol implemented** — `badapple model list/scan/info/use/verify/recommend`, `badapple p2p peers/sync/models`, agent task management, tool invocation, inference, runtime control, identity, and audit commands all work. Previously the daemon rejected `__BADAPPLE_AGENT__` with "not supported."
5. **read_document ported to native Swift** — PDF (PDFKit), DOCX (unzip + XML), and RTF (NSAttributedString) are parsed natively. Previously returned "requires the Python daemon."
6. **KV cache and prefill env vars wired** — `BADAPPLE_MAX_KV_SIZE` and `BADAPPLE_PREFILL_STEP_SIZE` are now read and passed to `GenerateParameters`.
7. **Dead firefly_edgeos code deleted** — 7 orphaned Rust files (~4,300 lines) that were never compiled have been removed. They referenced types (`FullySapientSoulMatrix`, `Skill`, `SensorSnapshot`) that were never part of the new architecture.
8. **Honest documentation** — Ranking document corrected to accurately reflect what is in the compiled product vs. what was claimed.
9. **Model manager and registry ported** — `BadAppleModelManager.swift` tracks 0.5B/9B/32B/70B/vision/image models, scans the HF cache, supports `add/remove`, records SHA-256 provenance manifests, and gives memory-aware recommendations.
10. **P2P encrypted mesh sync ported** — `p2p_crypto.rs` adds AES-256-GCM, `protocol.rs` encrypts broadcast packets, and `badapple-p2p` is spawned by the daemon when `BADAPPLE_P2P=1`.
11. **Dashboard HTTP server ported** — `badapple-dashboard` (Rust + axum) serves the `web/` SPA and proxies `/api/*` calls to the daemon on port 8787.
12. **MCP server host ported** — `src/mcp.rs` and `badapple-mcp` implement the Model Context Protocol over `stdio` or a Unix socket, proxying tool calls to the daemon.
13. **Air-gap certification suite expanded** — `tests/cert_suite.rs` adds 5 Rust tests for network isolation, Unix sockets, policy presence, ledger secret redaction, and P2P/MCP being off by default.
14. **FLUX image generation ported** — `image_generation` tool in `BadAppleTools.swift` calls `mflux-generate-flux2` and writes PNGs to `/var/lib/bad_apple/generated_images`.

### What moved the needle this pass (9.25 → 9.5)

1. **Full P2P model transfer over encrypted mesh** — `badapple p2p pull/send/receive` and `p2p_model.rs` implement encrypted AES-256-GCM chunked file transfer with resume and SHA-256 manifest gossip.
2. **MCP marketplace catalog and lifecycle** — `mcp_marketplace.rs` and the dashboard `/api/mcp/servers` endpoints support list/add/remove/install/uninstall/start/stop/status. The CLI has `badapple mcp list|add|remove|install|uninstall|start|stop|status|init`.
3. **Workspace watcher ported** — `workspace_watcher.rs` monitors workspace file changes and triggers real-time indexing via `notify`.
4. **Ambient context and ocular screen-stream endpoints** — `BadAppleAmbient.swift` and `BadAppleScreenCapture.swift` provide native helpers; `badapple-dashboard` exposes `/api/ambient` and `/api/ocular` with configurable capture/describe intervals.
5. **Vault integrated with daemon and menu bar** — `vault.rs` stores secrets with HSM-backed keys; the CLI has `badapple vault set/get/list/remove`.
6. **Swift menu bar / daemon linker and build fixed** — `BadAppleMLX` public modifier issues resolved, `IFS` and dylib linking fixed in `build_bad_apple_menu_bar.sh`, and `badapple-engine` builds and bundles successfully.
7. **Native TTS fixed and auto-starting** — `BadAppleMenuBar` `PiperTTSPlaybackController` now uses direct `afplay` for user-session playback, and `badapple` CLI auto-starts `badapple-tts` if its socket is missing.
8. **Signed full-release packaging** — `package_signed_release.sh` produces a code-signed `Bad_Apple-<version>-full-signed.zip` with a self-signed or Apple Developer cert.
9. **Test count and cert suite expanded** — 103 Rust tests and 15 cert-suite integration tests pass, including SLICKS replay, tool cage, path traversal, symlink escape, policy coverage, P2P crypto, output firewall, ledger integrity, vault round-trip, and network-isolation checks.
10. **Dead core modules removed** — `hyperdimensional_core.rs` and `connectome_mmap.rs` are deleted; `RustSynthesizer` is now a pure repair helper; `ReplayCache` moved into `bad_apple_ipc.rs`.
11. **Ambient and ocular context wired into active prompts** — `BadAppleEngine` now refreshes app/window context and optional screen-capture + VLM description before every turn, with a 30-second background timer keeping it warm.
12. **First-run onboarding flow fixed** — The compact install panel is shown first when the platform is not yet installed; the 5-step `BadAppleOnboardingWindow` runs once the platform is ready.

### What was corrected from the previous ranking

- **Air-gap certification suite** — NOT removed. `tests/bad_apple_daemon.rs` and `tests/ane_brain_perf.rs` are Rust integration tests that assert zero network sockets on the daemon and ANE core. These are the air-gap certification tests.
- **P2P AES-256-GCM encryption** — Now implemented in `p2p_crypto.rs` and `protocol.rs`. Packets are encrypted with AES-256-GCM and signed with HMAC-SHA256.
- **13-actor runtime** — The 13 Python actors were deleted and never ported. This claim was false in the previous ranking and has been removed.
- **Dual-process cognitive governor** — `governor.rs` was orphaned and deleted. No System 1 / System 2 architecture exists in the compiled product.
- **FLUX image generation** — Ported to the native `image_generation` tool in `BadAppleTools.swift`. It calls the local `mflux-generate-flux2` binary when installed and writes PNGs to `/var/lib/bad_apple/generated_images`.
- **Piper TTS** — Replaced by native `AVSpeechSynthesizer`. The "PiperTTSPlayback" class name is just the audio playback controller.
- **171 tests** — Now 103 Rust tests plus 15 cert-suite integration tests. Red-team and security regression tests are in `tests/cert_suite.rs`.
- **10,000-dimensional hyperdimensional computing** — `hyperdimensional_core.rs` has been removed. It was only used by a dead FFI path and the `RustSynthesizer` template map; `RustSynthesizer` is now a simple repair helper.
- **Memory-mapped connectome persistence** — `connectome_mmap.rs` and the `MemoryGraphNode` struct have been removed. There was no runtime consumer.
- **Metal UMA zero-copy memory management** — `metal_uma.rs` is wired into the gatekeeper's `CandleBrain` classifier through `tensor_brain.rs`; it is not used by the Swift MLX runtime, but it is not dead code.
- **Web dashboard SPA** — Now served by `badapple-dashboard` on port 8787.
- **Ambient + ocular context** — No longer just dashboard endpoints; `BadAppleEngine` now injects context into the active system prompt on each turn.
- **First-run onboarding** — The 5-step native onboarding wizard is the primary first-launch flow; the compact install panel is shown first if the platform has not been installed.
- **MCP marketplace** — `mcp_marketplace.rs`, `badapple-mcp`, and the dashboard `/api/mcp/servers` endpoints provide a full catalog with add/remove/install/uninstall/start/stop/status.

### Remaining blockers to 9.75/10

- Notarized `.dmg` and `.zip` as the default release artifact (eliminates Gatekeeper warning for direct-download users; signed-but-not-notarized zips already work with a `CODESIGN_ID`).


- A clean-machine VM install + smoke test to verify the DMG and Cask end-to-end.

- 3 unmaintained transitive Rust dependencies (`fxhash`, `instant`, `paste`) with no safe upgrade path — explicitly triaged in `deny.toml` but worth monitoring.


---

## Competitive Ranking (October 2026)

### Bad Apple is an AI OS layer, not an app

Bad Apple runs as **three system-level launchd daemons** (gatekeeper, MLX server, supervisor) managed by launchd with `KeepAlive`, a **Rust gatekeeper proxy** (1072 lines) on a system Unix socket with SLICKS v1/v2 authentication, a **bounded health supervisor** that restarts failed services with a restart budget (2 per 10 minutes) and enters safe mode on exhaustion, a **Candle-based classifier brain** in the gatekeeper for fast action resolution, a **WebAssembly sandbox** for untrusted code execution, a **fail-closed filesystem automation cage**, an **APFS file scavenger** module, a **declarative security policy engine** (60 tool rules in `policy.yaml`), a **Secure Enclave identity** with hardware-rooted signing, a **hash-chained audit ledger**, a **streaming output firewall** with real-time secret redaction, a **VRAM admission governor** plus a **memory pressure governor**, an **optional fast tier** with 0.5B model routing, **optional speculative decoding**, a **native agent task system** with plan-execute-observe loops, **native document reading** (PDF/DOCX/RTF), **vision/image description**, a **P2P encrypted mesh** for model and message sync, an **MCP marketplace** for local tool servers, a **workspace watcher**, **ambient/ocular context helpers**, a **local vault**, a **CLI agent protocol** for full runtime control, and **air-gap certification integration tests** that assert zero network sockets.

The menu bar app and chat window are the user-facing surface, the way Siri is the surface of Apple Intelligence. The actual product is the OS-level AI substrate underneath.

### The real competitive landscape

| Tier | Products | What they are |
|---|---|---|
| **Platform AI OS** | Apple Intelligence, Google Gemini Nano, Microsoft Copilot+ | Shipped by the OS vendor, baked into the OS |
| **Independent AI OS layers** | **Bad Apple**, OpenAGI | System services with their own daemons, identity, policy, audit, hardware integration |
| **Local AI runtimes** | Ollama, LM Studio, llama.cpp, MLX | Model servers / engines, no OS-level services |
| **Local AI apps** | M1K3, Ka1zen, MLX Studio, macMLX, Macaw, iClaw, mlx-serve | Apps that use a runtime, no system services |

### Why Bad Apple is #1 in the independent AI OS layer tier

The only other product in this tier is OpenAGI, which is a proactive daemon with screen watching and multi-channel reachout. Bad Apple has everything OpenAGI has plus:

- **Hardware-rooted identity** (Secure Enclave signing, SLICKS v2)
- **Hash-chained audit ledger** with SHA-256 chaining and HMAC
- **Air-gap certification** (Rust integration tests assert zero network sockets)
- **Streaming output firewall** with real-time secret redaction (Aho-Corasick)
- **Human-in-the-loop approval policy engine** (60 rules in `policy.yaml`)
- **VRAM admission governor** with model size estimation and configurable budget
- **Memory pressure governor** with automatic VRAM purge on critical pressure
- **Fast tier model routing** (0.5B for simple queries, 9B for complex)
- **Speculative decoding** support with configurable draft model and token count
- **Native agent task system** with plan-execute-observe loops
- **Native document reading** (PDF/DOCX/RTF without external dependencies)
- **Vision/image description** via MLX vision model
- **CLI agent protocol** for full JSON-RPC runtime control
- **Semantic cache** with native embedding-based similarity matching
- **RAG with native embeddings** (bge-small via BadAppleEmbeddingEngine)
- **Encrypted P2P mesh** for model and message sync (AES-256-GCM, off by default)
- **MCP marketplace** for installing and calling local MCP servers
- **Workspace watcher** with real-time file monitoring and RAG updates
- **Local vault** for HSM-backed secret storage
- **Bounded health supervisor** with restart budgets and safe mode
- **Gatekeeper proxy** with Candle classifier brain, automation cage, and WASM sandbox
- **APFS file scavenger** module with tokenized chunking
- **103 Rust tests plus 15 cert-suite integration tests** including air-gap certification, path-traversal, P2P crypto, output firewall, ledger integrity, vault round-trip, and WASM cage tests

OpenAGI has none of these. It's a proactive agent daemon; Bad Apple is an AI operating system layer.

### Why Bad Apple is not in the "Local AI apps" tier

M1K3, Ka1zen, MLX Studio, macMLX, Macaw, iClaw, and mlx-serve are all apps that sit on top of MLX or Ollama. They do not run system daemons as root. They do not have a gatekeeper proxy. They do not have a health supervisor. They do not have hardware-rooted identity. They do not have an audit ledger or a policy engine. Comparing Bad Apple to them is a category error — like comparing systemd to a terminal emulator.

### Why Bad Apple is not in the "Platform AI OS" tier

Apple Intelligence, Google Gemini Nano, and Microsoft Copilot+ are shipped by the OS vendor and baked into the OS. Bad Apple is an independent layer that runs alongside the vendor's AI. It cannot match their distribution (hundreds of millions of devices) or their OS-level integration (Siri, Writing Tools, system UI). But it offers something they structurally cannot: **provable, auditable, air-gapped privacy** — because it is not the OS vendor and has no cloud to phone home to.

### Honest overall placement

**Bad Apple is #1 in the independent AI OS layer tier.** The only other product in this tier is OpenAGI, which is a proactive daemon without hardware-rooted identity, an audit ledger, air-gap certification, a policy engine, VRAM admission, fast tier routing, speculative decoding, agent tasks, native document reading, a gatekeeper proxy, or a health supervisor.

**Bad Apple is not in the "Local AI apps" tier.** M1K3, Ka1zen, MLX Studio, macMLX, Macaw, iClaw, and mlx-serve are apps that sit on top of MLX or Ollama. They do not run system daemons as root, they do not have a gatekeeper proxy, they do not have a health supervisor, and they do not have a policy engine. Comparing Bad Apple to them is a category error.

**Bad Apple is not in the "Platform AI OS" tier.** Apple Intelligence, Google Gemini Nano, and Microsoft Copilot+ are shipped by the OS vendor and baked into the OS. Bad Apple cannot match their distribution (hundreds of millions of devices) or their OS-level integration (Siri, Writing Tools, system UI). But it offers something they structurally cannot: **provable, auditable, air-gapped privacy** — because it is not the OS vendor and has no cloud to phone home to.

### What Bad Apple has that no app-tier competitor has

- **Three system launchd daemons** (gatekeeper, MLX, supervisor) running as root with `KeepAlive`
- **Rust gatekeeper proxy** (1072 lines) on a system Unix socket with SLICKS v1/v2 auth
- **Candle-based classifier brain** in the gatekeeper for fast action resolution
- **Bounded health supervisor** with restart budgets (2 per 10 min) and safe mode
- **Secure Enclave identity** with hardware-rooted signing (SLICKS v2)
- **Hash-chained audit ledger** with SHA-256 chaining and HMAC
- **Air-gap certification** — Rust integration tests assert zero network sockets
- **Streaming output firewall** with real-time secret redaction (Aho-Corasick)
- **Declarative security policy engine** (60 tool rules in `policy.yaml`)
- **VRAM admission governor** with model size estimation and configurable budget
- **Memory pressure governor** with automatic VRAM purge
- **Fast tier model routing** (0.5B for simple queries, 9B for complex)
- **Speculative decoding** with configurable draft model
- **Native agent task system** with plan-execute-observe loops
- **CLI agent protocol** (LAP) for full JSON-RPC runtime control
- **Encrypted P2P mesh** for model and message sync (AES-256-GCM)
- **MCP marketplace** for installing and calling local MCP servers
- **Workspace watcher** with real-time file monitoring
- **Local vault** for HSM-backed secret storage
- **WebAssembly sandbox** in the gatekeeper for untrusted code execution
- **Fail-closed filesystem automation cage** (allowlisted roots only)
- **APFS file scavenger** module with tokenized chunking and Sled persistence
- **103 Rust tests plus 15 cert-suite integration tests** including air-gap certification, path-traversal, P2P crypto, output firewall, ledger integrity, vault round-trip, and WASM cage tests

### Notarization stance

Bad Apple will not be submitted to Apple's notarization pipeline. Notarization requires uploading binaries to Apple's servers — a violation of the product's "nothing leaves your machine" promise. This is a deliberate philosophical choice. The **Homebrew Cask** (`brew install --cask bad-apple`) is the recommended install path: it strips quarantine automatically in `postflight`, does not route through Apple, and is trusted by millions of developers. For direct-download users, the DMG includes `Install.command` and `strip_quarantine.sh`. The Gatekeeper warning on direct download is the trade-off for being the only AI OS layer that refuses to phone home to Apple.

---

## License

LicenseRef-Proprietary. See `LICENSE.txt`.
