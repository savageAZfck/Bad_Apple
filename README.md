# Bad Apple

> **A bare-metal AI operating system that runs on your Mac — no cloud after the first download.**

Bad Apple is not a chatbot. It is a **bare-metal AI OS for macOS**: a private,
on-device cognitive layer that runs a **Qwen 3.5 9B** brain and a **0.5B fast
tier** on Apple Silicon with [MLX](https://github.com/ml-explore/mlx), owns its
own memory, tools, policy, and audit trail, and never sends prompts, responses,
or data to the cloud after the first model download.

After the model weights are cached once, **nothing leaves your machine**.

---

## What it is

Bad Apple is both a product and a proof-of-concept for a **Bare-Metal AI OS**:

- The model, memory, RAG index, audit ledger, and persona data live on device.
- Every client connection is authenticated with a **SLICKS** HMAC-SHA256
  challenge-response bound to the prompt.
- Tool calls are governed by a declarative `policy.yaml` cage.
- Destructive actions require explicit approval unless the user opts into
  autopilot.
- It can sync encrypted memory with other Bad Apple peers on the same LAN.
- It ships as a consumer **DMG installer** and a **Homebrew Cask**,
  and it can lazy-load the 9B brain so the first query is fast.

Read the public specs in [MANIFESTO.md](MANIFESTO.md) and [STANDARDS.md](STANDARDS.md),
and the roadmap in [ROADMAP.md](ROADMAP.md).

---

## What it does

### Core inference

- **Local 9B reasoning** — `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` on the GPU.
- **Speculative decoding** — DFlash is disabled for the 9B quant; the runtime
  supports `mlx-lm` speculative decoding with a small cached draft model
  (e.g. `Qwen2.5-0.5B-Instruct-4bit`) via `enable draft` or
  `BADAPPLE_SPECULATIVE_DRAFT=auto`.
- **Streaming output** — tokens stream to the terminal, TTS, or the menu bar
  as they are generated.
- **Hot-reloadable prompt** — edit `prompt.txt` without restarting the daemon.
- **Voice + TTS** — local Piper TTS via `badapple_tts_server.py`, played
  through `afplay`; optional menu-bar voice input.

### Memory & retrieval

- **Long-term memory graph** — facts, entities, and relations stored in
  `badapple_extras.py`; supports semantic search.
- **Local RAG** — indexes text/code files with `BAAI/bge-small-en-v1.5` and
  retrieves relevant excerpts.
- **Semantic cache** — caches answers by embedding similarity and serves them
  instantly.

### Personas & behavior

- **Persona packs** — Default persona from `prompt.txt`, plus `personas.json`
  packs for Wicket, Gen Z Hype (`genz`), Drill, and Midwest Aunt; switch at
  runtime with `switch to <persona>`.
- **Teachable quips** — `teach The cloud is just hamsters on a wheel` and the
  line is stored in the active persona's custom banter.

### Agent protocol & tools

- **Local Agent Protocol (LAP)** — JSON-RPC over the authenticated SLICKS
  Unix socket, exposed by `agent_client.py` via `__BADAPPLE_AGENT__`
  sentinel-prefixed JSON payloads.
- **Autonomous agent tasks** — queue multi-step goals with `agent_client.py agent create`,
  or from the dashboard at `/agents`. The OS plans, executes local tools, observes results,
  recovers from errors, and persists tasks across restarts.
- **Tool set** — local file system, shell (approved), AppleScript (approved),
  RAG, time, macOS Shortcuts, Accessibility actions, screen capture, image
  description, P2P peer discovery, LoRA training/inference, local image
  generation, speech-to-text, OCR, PDF/EPUB reading, translation, Git copilot,
  system dashboard, scheduled tasks, Spotlight search, ambient context, and
  Xcode project RAG. No web search or cloud APIs.

### Vision

- **Ocular UI Stream** — live screen capture and optional VLM description on a
  configurable interval. The dashboard shows the latest frame and the most recent
  description at `/ocular`.
- **Screen capture** — capture the main Mac screen to a PNG via `screencapture`
  (macOS screen-recording permission required).
- **Image description** — local MLX-VLM with
  `mlx-community/Qwen2-VL-2B-Instruct-4bit`; no cloud after model cache.

### Personal fine-tuning

- **LoRA training** — `lora_add_example` / `lora_train` to build on-device
  adapters from prompt/completion pairs using `mlx-lm`.
- **LoRA generation** — `lora_generate` to run the saved adapter.
- **P2P adapter handoff** — `p2p_send_adapter` beams a trained adapter to a
  discovered Bad Apple peer like AirDrop for models.
- Adapters and datasets live in `/var/lib/bad_apple/lora_*` by default.

### Multimodal & media

- **Local image generation** — `generate_image` with FLUX.2-klein-4B via `mflux`
  (quantized, on-device, ~4 GB model cache).
- **Local speech-to-text** — `transcribe_audio` with MLX Whisper.
- **OCR / screen text** — `extract_text_from_image` and `capture_and_extract_screen`.
- **PDF/EPUB Q&A** — `read_document` and `index_documents` feed local documents
  into the RAG pipeline.
- **Translation** — `translate_text` with a local `m2m100-418M` model.

### Productivity & control

- **Git copilot** — `git_status`, `git_diff`, `git_log`, and `git_commit` tools
  use only the local repo.
- **Power dashboard** — `system_dashboard` returns CPU, memory, swap, disk,
  battery, thermal pressure, and Bad Apple process stats.
- **Deterministic sessions** — `set_session_seed` pins the MLX random stream;
  same prompt, same output.
- **Scheduler + Shortcuts** — `schedule_task`, `list_scheduled_tasks`, and
  `run_shortcut` run local commands and macOS Shortcuts.
- **Ambient context + memory** — `ambient_start`/`ambient_context` captures active app,
  window title, and screenshots in the background. `badapple_ambient_memory` turns
  every snapshot into facts and episodic memory, so the OS remembers what you were
  doing. Optional VLM screen description with `BADAPPLE_AMBIENT_VLM=1`.
- **Universal Spotlight** — `spotlight_search` queries macOS Notes, Mail,
  files, and the Bad Apple history/ledger locally.
- **Xcode coding assistant** — `xcode_index_project` and `xcode_search` index a
  whole Xcode/Swift project for code Q&A.
- **Safari companion** — scaffolded in `src/platform/safari_extension/`;
  native-messaging bridge to the local socket.

### Security

- **Keychain key storage** — `slicks_keychain_store` puts the SLICKS secret in
  the macOS Keychain instead of a plain file.
- **Per-app / per-file policy** — `allowed_apps` and `allowed_files` in
  `policy.yaml` restrict `accessibility_action`, `read_file`, and other tools.

### Network & sync

- **Encrypted P2P sync** — `badapple_p2p.py` discovers peers on the local
  network via UDP beacons and syncs the memory graph over TCP with
  AES-256-GCM + HMAC-SHA256, keyed from the SLICKS secret.
- **Air-gap certification** — `cert_suite.py` validates network isolation,
  local sockets, policy, secret redaction, local model weights, and absence
  of hard-coded cloud endpoints.

### Safety & audit

- **Declarative policy cage** — `policy.yaml` defines which tools are allowed,
  whether they need approval, and argument constraints.
- **Streaming output firewall** — Aho-Corasick blocklist for secrets, PII,
  and custom patterns.
- **Hash-chained audit ledger** — every query, tool, and response is logged to
  `/var/lib/bad_apple/ledger.jsonl` with SHA-256 chaining and secret redaction.
- **Human-in-the-loop approvals** — destructive tools require approval
  (reply `approve <id>`) by default; set `BADAPPLE_AUTOPILOT=1` to skip.

### Workspace mode

- **Project/workspace** — say `set workspace to <path>` (or call the LAP
  `set_workspace` method) to scope file and RAG operations to a directory;
  `clear workspace` returns to global mode.

---

## Consumer packaging

Bad Apple can be installed like a normal macOS app:

- **DMG installer** — `src/platform/apple_desktop/package_dmg.sh` builds
  `target/release/Bad_Apple-<version>.dmg` with a drag-to-Applications UX
  and an `Install.command` that installs the platform LaunchDaemons.
- **Homebrew Cask** — `src/platform/apple_desktop/package_homebrew_cask.sh`
  builds the tap. The canonical formula is in `homebrew-bad-apple/Casks/bad-apple.rb`.
- **Unsigned zip** — `src/platform/apple_desktop/package_full_release.sh`
  creates `target/release/Bad_Apple-<version>-full-unsigned.zip`.
- **Signed / notarized zip** — `src/platform/apple_desktop/package_signed_release.sh`
  with `CODESIGN_ID`, `APPLE_ID`, `APPLE_TEAM_ID`, and `APPLE_APP_PASSWORD`.

### Lazy loading and the model manager

- Set `BADAPPLE_LAZY_MAIN_MODEL=1` to skip the 9B load at startup. Simple
  queries still hit the fast 0.5B tier; the 9B brain loads on the first deep
  question.
- `badapple_model_manager.py` tracks `main_9b`, `fast_0.5b`, `vision_2b`, and
  `flux_4b` with background download status and progress.
- The dashboard at `http://127.0.0.1:8787/models` lets users opt in to
  downloads and start them before first use. Downloads are disabled by default.

---

## Quick start

### Requirements

- Apple Silicon Mac (M1 or later)
- macOS 26 or later
- Rust toolchain with Cargo
- Python 3.12 with the packages in `.venv` (`mlx`, `mlx-lm`, `mlx-vlm`, etc.)
- Models download on first run (or pre-download from the dashboard):
  - `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit`
  - `BAAI/bge-small-en-v1.5`
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (first vision call)
  - `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (optional, for fast tier / speculative decoding)

### Build

```bash
cargo build --release
```

### Install and start the daemons

See [INSTALL.md](INSTALL.md) for the full consumer instructions. The no-Dev-ID short path is:

```bash
cargo build --release
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
cp -R "target/release/Bad Apple.app" /Applications/
sudo src/platform/apple_desktop/strip_quarantine.sh
osascript -e 'do shell script "cd /path/to/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install --unsigned-install" with administrator privileges'
```

This installs and runs the platform without Apple notarization or a Developer ID. The `--unsigned-install` flag removes the Gatekeeper quarantine flag automatically.

### Signed and notarized release

If you have an Apple Developer ID Application certificate and notarization credentials, build a signed, notarized consumer zip:

```bash
export CODESIGN_ID="Developer ID Application: Your Name (TEAMID)"
export APPLE_ID="you@example.com"
export APPLE_TEAM_ID="TEAMID"
export APPLE_APP_PASSWORD="abcd-1234-abcd-1234"

src/platform/apple_desktop/package_signed_release.sh
```

This produces `target/release/Bad_Apple-<version>-full-signed.zip`. See `SIGNING.md` for details.

### Tests

```bash
cargo fmt --check && cargo build --release
.venv/bin/python -m ruff check .
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Wait ~45 s for the 9B model and embedding model to load. Check the log:

```bash
tail -n 20 /var/log/bad_apple_mlx_server.log
```

Need help? See [SUPPORT.md](SUPPORT.md).

### Run your first prompt

```bash
target/release/badapple "What time is it?"
target/release/badapple -n 240 "Write me a poem about bare metal"
```

### Voice / TTS

```bash
target/release/badapple --speak "What do you think of Siri?"
```

### Personas and roast

```bash
target/release/badapple --persona wicket "Who are you?"
target/release/badapple --roast "What do you think of Siri?"
target/release/badapple "switch to midwest"
```

### Benchmark

```bash
target/release/badapple --benchmark
```

### Agent / tool calls

```bash
# List exposed tools
python agent_client.py discover

# Run a tool
python agent_client.py invoke get_current_time '{}'
python agent_client.py invoke screen_capture '{}'
python agent_client.py invoke lora_adapters '{}'

# Other agent commands
python agent_client.py infer 'What is 2+2?'
python agent_client.py workspace /path/to/bad_apple
python agent_client.py p2p_peers
python agent_client.py p2p_sync
```

### Menu bar

Build and install the status-bar host:

```bash
src/platform/apple_desktop/build_bad_apple_menu_bar.sh
cp -R target/release/Bad\ Apple.app /Applications/
open -a "Bad Apple"
```

---

## Architecture

```text
badapple CLI / menu bar / Siri
              │
              ▼
  /var/run/badapple/substrate.sock  (gatekeeper)
              │
              ▼
     gatekeeper (SLICKS proxy, fast actions)
              │
              ▼
  /var/run/badapple/substrate_mlx.sock
              │
              ▼
   badapple_mlx_server.py (9B Qwen3.5 + DFlash + tools + RAG + memory + vision + LoRA + multimodal)
              │
              ▼
    badapple_tts_server.py (Piper TTS)
```

- **gatekeeper** — authenticated front door; SLICKS challenge-response, prompt
  binding, fast action proxy. It listens on `substrate.sock` and forwards to
  the MLX server on `substrate_mlx.sock`.
- **MLX server** — loads the 9B target + optional draft once; handles
  conversation, memory, RAG, tools, approvals, cache, firewall, audit, P2P,
  screen capture, image description, LoRA, image generation, STT, OCR,
  translation, Git, dashboard, scheduling, Spotlight, ambient context, Xcode
  RAG, and the model manager.
- **TTS server** — Piper on a Unix socket, returns WAV paths.
- **Menu bar app** — Swift status-bar host with voice, persona switching,
  benchmarks, and output.

---

## Tools

All tools are local and policy-governed:

- `get_current_time` — current local time.
- `list_directory` — list a directory.
- `read_file` — read a text file.
- `write_file` — write a note to `~/.bad_apple/notes/` (approved).
- `search_content` — grep under a directory.
- `search_local_files` — Spotlight via `mdfind`.
- `run_shell` — read-only allowlisted shell commands (approved).
- `run_applescript` — safe macOS automation (approved).
- `index_documents` / `read_document` — index or extract PDF/EPUB/plain files.
- `search_notes` — semantic RAG search.
- `run_shortcut` — macOS Shortcuts integration.
- `schedule_task` / `list_scheduled_tasks` — local task scheduler.
- `accessibility_action` — type/click/key/menu via System Events.
- `screen_capture` / `capture_and_extract_screen` — capture the screen.
- `describe_image` / `extract_text_from_image` — VLM and OCR.
- `generate_image` — local FLUX.2-klein-4B image generation.
- `transcribe_audio` — local Whisper speech-to-text.
- `translate_text` — local m2m100 text translation.
- `git_status` / `git_diff` / `git_log` / `git_commit` — local Git copilot.
- `system_dashboard` — power/performance dashboard.
- `set_session_seed` / `get_session_seed` — deterministic sessions.
- `spotlight_search` — local Spotlight across Mail/Notes/files/history.
- `ambient_start` / `ambient_stop` / `ambient_context` — always-on context.
- `lora_add_example` / `lora_train` / `lora_adapters` / `lora_generate` —
  personal LoRA fine-tuning.
- `p2p_peers` / `p2p_list_adapters` / `p2p_send_adapter` — P2P discovery and
  AirDrop-style LoRA handoff.
- `xcode_index_project` / `xcode_search` — Xcode/Swift project RAG.
- `slicks_keychain_store` — store SLICKS secret in macOS Keychain.

---

## Security & privacy

- **Air-gapped at runtime** — after models are cached, no network calls are
  made for inference, actions, TTS, LoRA, or P2P data (P2P is strictly local
  broadcast/TCP).
- **Authenticated** — SLICKS HMAC-SHA256 challenge-response with prompt
  binding; replaying an old challenge does not work.
- **Fail-closed tools** — shell/AppleScript, file writes, indexing, screen
  capture, and LoRA training require approval unless `BADAPPLE_AUTOPILOT=1`.
  The provided `com.badapple.mlx.plist` sets it to `1`, so a
  launchd-installed daemon runs without prompts; remove it if you want
  per-action approval.
- **Declarative cage** — `policy.yaml` controls tool permissions and validates
  arguments.
- **Streaming firewall** — secrets and PII patterns are blocked or redacted.
- **Audit ledger** — append-only, hash-chained JSONL at
  `/var/lib/bad_apple/ledger.jsonl`.
- **Local-only audio** — TTS synthesized on device.

---

## Project structure

```text
badapple_mlx_server.py          # 9B MLX inference + tools daemon
badapple_model_manager.py       # background model download / status manager
badapple_agent_tasks.py         # persistent autonomous agent task manager
badapple_ambient_memory.py      # ambient snapshot → memory graph
badapple_extras.py              # personas, firewall, audit, cache, approvals, memory graph
badapple_vision.py              # screen capture and VLM image description
badapple_lora.py                # on-device LoRA training and generation
badapple_p2p.py                 # encrypted peer-to-peer sync and adapter handoff
badapple_stt.py                 # local Whisper speech-to-text
badapple_image_gen.py           # local FLUX.2 image generation via mflux
badapple_translate.py           # local m2m100 text translation
badapple_documents.py           # PDF/EPUB extraction and RAG
badapple_git.py                 # local Git status/diff/log/commit helper
badapple_dashboard.py           # power and performance dashboard
badapple_scheduler.py           # task scheduler and Shortcuts runner
badapple_ambient.py             # always-on screen/app context capture
badapple_spotlight.py           # local Spotlight search across Notes/Mail/files/history
badapple_xcode.py               # Xcode/Swift project RAG
badapple_keychain.py            # macOS Keychain-backed SLICKS secret storage
badapple_tts_server.py          # Piper TTS daemon
agent_client.py                 # SLICKS/LAP agent client
policy.yaml                     # declarative tool cage
personas.json                   # persona packs
prompt.txt                      # hot-reloadable system prompt
cert_suite.py                   # air-gap certification self-tests
BAD_APPLE.md                    # technical deep-dive
MANIFESTO.md                    # Bare-Metal AI OS principles
STANDARDS.md                    # public interface and protocol specs
src/bin/badapple.rs             # Rust CLI client
src/bin/gatekeeper.rs           # SLICKS proxy + fast action gate
src/bad_apple_ipc.rs            # SLICKS protocol
src/platform/apple_bridge/      # launchd plists and Siri bridge
src/platform/apple_desktop/     # menu bar app source and packaging scripts
src/platform/apple_desktop/package_dmg.sh            # drag-to-Applications DMG
src/platform/apple_desktop/package_homebrew_cask.sh  # Homebrew tap generator
src/platform/apple_desktop/package_full_release.sh   # unsigned zip
src/platform/apple_desktop/package_signed_release.sh # notarized zip
src/platform/safari_extension/  # Safari companion extension scaffold
```

---

## Performance

Measured on a 16 GB Apple Silicon M-series Mac:

- **First token**: ~3.5–6.5 s for 450–750 token prompts.
- **Decode throughput**: ~13–25 tok/s, with spikes to ~36 tok/s on
  high-acceptance turns.
- **Peak memory**: ~5.7–6.5 GB with the 9B model + DFlash draft loaded;
  VLM, image generation, translation, and LoRA generation add transient
  working memory.
- **RAG embeddings**: `bge-small-en-v1.5` on CPU.

---

## Consumer readiness

Bad Apple is currently rated **7.5/10** for consumer readiness.

What works now:
- drag-to-Applications DMG with `Install.command`
- Homebrew Cask formula and tap generator
- lazy 9B loading and a 0.5B fast tier
- background model manager with dashboard checklist
- 52 passing unit tests, `ruff` clean, `cargo build --release`

Remaining blockers to 8+:
- signed and notarized release as the default artifact
- full FLUX pre-download in the model manager
- a clean-machine VM install / smoke test

---

## License

LicenseRef-Proprietary. See `LICENSE.txt`.
