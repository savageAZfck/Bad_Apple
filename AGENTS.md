# AGENTS.md — Bad Apple

This file captures the project-specific commands and conventions learned while working on Bad Apple, so the next agent (or future you) does not have to rediscover them.

## Project layout

- `target/release/badapple-engine` — Native Swift MLX daemon. Loads the 9B Qwen 3.5 model and serves the SLICKS Unix socket.
- `src/bin/badapple.rs` + `src/` — Rust CLI client that talks to the daemon.
- `src/platform/apple_bridge/com.badapple.mlx.plist` — launchd daemon config.
- `prompt.txt` — Hot-reloadable system prompt. Edits take effect on the next query without restarting the model.
- `BAD_APPLE.md` — Technical overview and live performance numbers.
- `BAD_APPLE_BUYERS.md` — Buyer-facing pitch doc.
- `personas.json` — Runtime persona packs (default, wicket, genz, drill, midwest).

## Build

```bash
cargo build --release
```

## Lint

```bash
cargo fmt --check
cargo build --release
```

The platform no longer depends on a venv for core installation. Swift unit tests can be run with `BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh` (which runs `BadAppleMLXSelfTest`).

## Full platform install

The installer replaces `/Applications/Bad Apple.app`, installs system LaunchDaemons, migrates `/var/lib/bad_apple` ownership, and loads the health supervisor. The install is now fully native by default (no `.venv` is created):

```bash
cargo build --release
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
osascript -e 'do shell script "cd /path/to/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install" with administrator privileges'
```

It creates a rollback snapshot under `/var/lib/bad_apple/install_backups/` and restores the previous launchd configuration if readiness does not pass.

To enable the optional native TTS service, install with `BADAPPLE_TTS=1`.

## Unsigned build and install

No Apple Developer ID is required. Build unsigned, strip the quarantine flag locally, and install without `codesign --verify` checks:

```bash
cargo build --release
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
sudo src/platform/apple_desktop/strip_quarantine.sh
osascript -e 'do shell script "cd /path/to/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install --unsigned-install" with administrator privileges'
```

Package an unsigned release zip with a consumer README and the quarantine stripper:

```bash
src/platform/apple_desktop/package_unsigned.sh
```

Produces `target/release/Bad_Apple-<version>-unsigned.zip`. The .app bundle now includes `Contents/Resources/update_bad_apple.sh` and `strip_quarantine.sh`.

## Update Bad Apple

From the menu bar, use **Bad Apple → Check for Updates...**, or run the bundled updater:

```bash
sudo /Applications/Bad\ Apple.app/Contents/Resources/update_bad_apple.sh
```

The updater compares the installed `CFBundleShortVersionString` against the latest GitHub release, downloads `Bad_Apple-<version>-unsigned.zip`, backs up the old app, and replaces it. It then strips quarantine and restarts the menu bar.

## Start / restart the daemon

The live setup uses the Rust `gatekeeper` on `/var/run/badapple/substrate.sock` as the front proxy. Restart both after a CLI/Rust change; only the MLX daemon needs a restart after a Swift engine change.

On this machine `launchctl bootstrap` returns `Input/output error` for these plists; use `load -w` / `unload` instead:

```bash
osascript -e 'do shell script "launchctl unload /Library/LaunchDaemons/com.badapple.mlx.plist 2>/dev/null; launchctl unload /Library/LaunchDaemons/com.badapple.gatekeeper.plist 2>/dev/null; launchctl unload /Library/LaunchDaemons/com.badapple.supervisor.plist 2>/dev/null; sleep 2; launchctl load -w /Library/LaunchDaemons/com.badapple.gatekeeper.plist; launchctl load -w /Library/LaunchDaemons/com.badapple.mlx.plist; launchctl load -w /Library/LaunchDaemons/com.badapple.supervisor.plist" with administrator privileges'
```

Wait ~45 s for the model bundle and embedding model to load. Check the tail of the log:

```bash
tail -n 20 /var/log/bad_apple_mlx_server.log
```

## Keep the menu bar running

Install the user LaunchAgent after `/Applications/Bad Apple.app` exists. It starts the icon at login and restarts it after a crash:

```bash
src/platform/apple_desktop/install_menu_bar_agent.sh
launchctl print gui/$(id -u)/com.badapple.menubar
```

## Test a query

```bash
# text
target/release/badapple "What is 2+2?"

# voice (text output only, no audio)
BADAPPLE_VOICE=1 target/release/badapple "What do you think of Siri?"

# voice with TTS (Piper server must be running)
target/release/badapple --speak "What do you think of Siri?"

# set max tokens
target/release/badapple -n 120 "Write me a poem about bare metal"

# benchmark the default prompt suite (skips fast tier, cache, and tool loops)
target/release/badapple --benchmark

# benchmark a single prompt
target/release/badapple --benchmark -n 120 "What is the capital of France?"

# switch persona just for this query
badapple --persona wicket "Who are you?"
badapple --roast "What do you think of Siri?"  # alias for --persona drill

# stream to TTS (background queue keeps generation uncoupled from afplay)
badapple --speak "What do you think of Siri?"
```

## Edit the persona

Edit `prompt.txt`. The daemon hot-reloads it on the next query. The default voice prompt lives in `prompt.txt` itself; persona-specific voice prompts are in `personas.json` under `voice_system_prompt`. Voice prompt changes require a daemon restart.

## Persona packs

```bash
# Switch persona at runtime
target/release/badapple "switch to wicket"
target/release/badapple "switch to drill"
target/release/badapple "switch to midwest"
target/release/badapple "switch to genz"

# Teach a custom line
target/release/badapple "teach The cloud is just hamsters on a wheel"
```

Personas live in `personas.json` (or `BADAPPLE_PERSONAS_FILE`).  The default
pack falls back to `prompt.txt` and learns `~/.bad_apple/custom_banter.json`.

## Output firewall

Add patterns (one per line) to `/var/lib/bad_apple/blocklist.txt` or set
`BADAPPLE_BLOCKLIST`.  Patterns are tokenized and matched with a streaming
Aho-Corasick automaton.  When the model is about to emit a match, the response
is replaced with `[Output firewall: ...]`.

## Hash-chained audit ledger

Every query, tool call, cache hit, and response is appended to
`/var/lib/bad_apple/ledger.jsonl` with SHA-256 chaining.  Verify the chain with
`target/release/badapple --doctor` or the standalone `verify-ledger` tool when
available.

Secrets, emails, SSNs, phones, API keys, and long random tokens are redacted
before writing.

## Semantic cache

The first response to a question is embedded with `BAAI/bge-small-en-v1.5` and
stored.  Repeated semantically similar queries return the cached answer
instantly, scoped by active persona.  Set `BADAPPLE_CACHE_THRESHOLD` (default
0.92).  Cache file: `/var/lib/bad_apple/semantic_cache.json`.

## Human-in-the-loop approvals

Destructive tools (`run_shell`, `run_applescript`, `write_file`, `index_documents`)
require approval by default.  The API returns a proposal ID; reply with the CLI:

```bash
# Approve with the CLI
target/release/badapple "approve <id>"
# Or skip approval for the session
BADAPPLE_AUTOPILOT=1 target/release/badapple "run shell ls /tmp"
```

## Kill switch, safe mode, and private mode

Voice/text commands:

```bash
target/release/badapple "kill switch"           # stop generation and actions
target/release/badapple "resume bad apple"      # reset kill switch and leave safe mode
target/release/badapple "leave safe mode"       # exit supervisor-induced safe mode
target/release/badapple "enable private mode"   # pause persistence
target/release/badapple "disable private mode"
```

## Air-gap certification

```bash
# The air-gap certification suite is being ported to Rust and is not currently
# available from the command line. Run `target/release/badapple --doctor` for a
# socket/process health report in the meantime.
```

P2P sync and HuggingFace hub are disabled by default for certification:

- P2P: set `BADAPPLE_P2P=1` in `com.badapple.mlx.plist` to enable link-local peer discovery.
- HF Hub: `HF_HUB_OFFLINE=1` is set in `com.badapple.mlx.plist` so the MLX server loads only cached weights.

## Check performance

```bash
# live log
tail -n 20 /var/log/bad_apple_mlx_server.log

# memory pressure on macOS
memory_pressure

## SLICKS 2.0 identity

The identity agent (`badapple-identity`) owns the Secure Enclave signing
context and runs as a user LaunchAgent. The Rust CLI and menu bar default to v2
when the agent socket is present.

```bash
# Socket and key paths
/var/run/badapple/identity.sock            # identity agent socket
/var/run/badapple/substrate_mlx.sock       # MLX daemon socket
/var/run/badapple/substrate.sock           # gatekeeper socket (default for CLI)

# Force SLICKS version
BADAPPLE_SLICKS2=1  target/release/badapple "prompt"   # v2 (Secure Enclave)
BADAPPLE_SLICKS2=0  target/release/badapple "prompt"   # v1 (HMAC)

# Restart the identity agent after changing the identity agent
rm -f /var/run/badapple/identity.sock
launchctl unload ~/Library/LaunchAgents/com.badapple.identity_agent.plist 2>/dev/null
src/platform/apple_bridge/install_identity_agent.sh
```

Look for log lines like:

```
[perf] 31 tokens @ 20.1 decode t/s (5.4 total t/s), draft_accept_ratio=58%, block_tokens=6, peak_memory=5.72 GB
```

## Optional native TTS

TTS is now provided by the native `badapple-tts` binary / menu-bar TTS service,
not a venv. Enable it at install time with `BADAPPLE_TTS=1`.

Install with TTS enabled:

```bash
BADAPPLE_TTS=1 osascript -e 'do shell script "cd /path/to/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install" with administrator privileges'
```

The platform plists are templates using `__BADAPPLE_ROOT__`, `__CONSOLE_USER__`, `__CONSOLE_HOME__`, and `__CONSOLE_GROUP__` placeholders. `install_badapple_platform.sh` renders them at install time, so the same release can be installed from any path and any user.

## Current generation settings

Set in `src/platform/apple_bridge/com.badapple.mlx.plist`:

- `BADAPPLE_DFLASH=0` — DFlash is off for this quant. DFlash's 9B draft model is too heavy to beat the verification overhead, so plain `mlx-lm` is faster overall.
- `BADAPPLE_SPECULATIVE_DRAFT=auto` — set to a cached MLX-LM draft model (e.g. `mlx-community/Qwen2.5-0.5B-Instruct-4bit`) or `auto` to scan the HF cache. Loaded at startup as the main model's draft.
- `BADAPPLE_NUM_DRAFT_TOKENS=2` — number of tokens the draft model generates per verification step. Runtime command: `set draft tokens to 4`.

In `src/platform/apple_desktop/BadAppleEngineDaemon.swift` (the native `badapple-engine` daemon):

- `prefill_step_size=4096` and `max_kv_size=4096` keep prefill in one shot and bound the KV cache.
- System prompt is hot-reloaded from `prompt.txt`; keep it compact to minimize TTFT.

## Common gotchas

- `BAD_APPLE.md` is the live technical doc; keep it in sync with the architecture and the latest benchmark numbers.
- `BAD_APPLE_BUYERS.md` is the buyer-facing doc; commit it when the numbers change.
- The 4B voice bundle has been removed in favor of the unified 9B brain; do not reintroduce it.
- `cargo fmt --check` is now clean; run `cargo fmt` and `cargo build --release` after Rust changes.
- Natural-language tool invocation (e.g. "run shell ls /tmp") is driven by the explicit `<tool_call>` XML instructions and few-shot examples in `prompt.txt`. The model emits the block, the CLI prompts for approval, and the user replies `approve <id>` to execute.

## Phase 3 Resource Governor

- `target/release/badapple "flush vram"` clears the Metal allocation cache.
- `target/release/badapple "unload all models"` drops optional models.
- `badapple "flush vram"` and `badapple "unload all models"` are the CLI control phrases that the menu bar sends.
- The menu bar app (`Bad Apple.app`) runs a `MemoryGovernor` that polls `host_statistics64` every 2 s and listens to `DispatchSourceMemoryPressure` critical events. When memory pressure crosses the configured critical threshold (default 0.95, overridable with `BADAPPLE_MEMORY_CRITICAL`), it automatically purges VRAM and unloads optional models.
- Build the menu bar app with `src/platform/apple_desktop/build_bad_apple_menu_bar.sh` and install the persistent agent with `src/platform/apple_desktop/install_menu_bar_agent.sh`.

## Phase 4 Agent OS Control Center

- The local dashboard, working-memory scratchpad, and runtime `tier` toggle are
  being ported to Rust/Swift and are not currently available from the command
  line.
- Fast tiering is controlled by the `BADAPPLE_FAST_TIER` setting in
  `com.badapple.mlx.plist` and the menu bar `Fast Tier Only` toggle. When on,
  simple math, identity, time, and greeting queries route to the 0.5B fast model.
- macOS Shortcuts can be listed/run through `list_shortcuts` and `run_shortcut`
  tools and the menu bar `Tools > Run Shortcut...`/`List Shortcuts`.

## Lint

```bash
cargo fmt --check
cargo clippy --release
```

## Phase 5 Local Intelligence Mesh

- Autopilot: the menu bar `Tools > Autopilot` toggle. When on, destructive tools
  run without approval prompts.
- Workspace / project mode: the menu bar `Mesh > Set Workspace...`/`Open Workspace`.
  Adds workspace context to prompts.
- P2P encrypted sync: `target/release/badapple p2p <peers|sync|models|pull <peer_id> <model_id>|send <peer_id> <model_id>|receive [peer_id model_id]>`. Link-local UDP/TCP, AES-256-GCM, off by default for air-gap certification.
- macOS Shortcuts: the menu-bar Aqua helper runs in the Aqua session and serves
  `/var/run/badapple/aqua_helper.sock`. `list_shortcuts` and `run_shortcut` proxy
  through the helper, so they work even though the daemon is not in a GUI session.

## Model registry and P2P (latest)

- Self-hosting model registry with first-class Rust CLI:
  - `badapple model list`, `badapple model scan`, `badapple model info <id>`
  - `badapple model use <id>`, `badapple model verify [id]`
  - `badapple model add <path> [id]`, `badapple model remove <id>`
  - `badapple model recommend`
- The model registry records SHA-256 fingerprints, signs manifests with the Secure Enclave, and verifies on demand.
- The MLX server reads `BADAPPLE_MAX_KV_SIZE` and `BADAPPLE_PREFILL_STEP_SIZE` (default 4096). Lower `BADAPPLE_MAX_KV_SIZE` to 1024-2048 before loading 32B+ models to stay within unified memory.
- The model registry includes 9B, 32B, 70B, and 1.5B options with memory guidance.
- P2P model manifest gossip and chunked file transfer with Rust CLI:
  - `badapple p2p peers`, `badapple p2p sync`, `badapple p2p models`, `badapple p2p pull <peer_id> <model_id>`
  - `badapple p2p send <peer_id> <model_id>`, `badapple p2p receive [peer_id] [model_id]`
- P2P is off by default. Enable with the menu bar `Mesh > P2P Sync`.
- Consumer install: double-click `Install Bad Apple` from the release zip, or run `src/platform/apple_desktop/install_badapple.sh`.
- First-run onboarding and plain-English Status window in `BadAppleMenuBar.swift`.

## Latest features (new)

- Streaming chat: POST /api/chat with `{"prompt": "...", "stream": true}` returns Server-Sent Events (tokens, tool calls, done, error).
- Automatic fact extraction, workspace file watching, ocular/screen-stream, and MCP marketplace features are being ported to Rust/Swift and are not currently available.
- Persona editor, dashboard, and control-center UI are being ported to the Swift menu bar and are not currently available from the web.

## New web UI (SPA) and native splash

- The dashboard is now a single-page app served from `web/index.html` with a unified sidebar, dark design system, and responsive layout.
- Routes: `/` (Dashboard), `/chat`, `/persona`, `/models`, `/agents`, `/ambient`, `/control`, `/mcp`, `/settings`, `/logs`, `/splash`.
- Static assets live in `web/static/` (styles.css, app.js) and are served by `DashboardHandler`.
- Chat view: streaming markdown, code blocks, tool-call cards, generated-image preview, scroll-to-bottom, auto-resize textarea.
- Settings view: workspace setter, MCP server list/add/remove, active models, runtime toggles (autopilot, fast tier, P2P).
- Dashboard view: status cards, P2P peers, latest perf, log tail.
- Control center (`/control`): kill/resume, autopilot, fast tier, ambient, P2P, VRAM flush, CLI override.
- MCP marketplace (`/mcp`): install from catalog, register custom servers, list tools, invoke tools.
- Onboarding modal shown once for new browsers.
- Native boot splash: `BadAppleSplashWindow` in `BadAppleMenuBar.swift` shows a progress bar on macOS app launch and auto-closes.

## Dashboard/chat UX (latest)

- Chat: conversation history saved in localStorage, new-chat, edit/retry/delete per message, copy code-block button, prompt suggestion chips, drag-and-drop file/image upload.
- Dashboard: live canvas chart for memory % and decode tokens/s, MCP server status, recent tool calls from ledger.
- Settings: model selector (list/switch), MCP tool invocation UI, theme toggle, runtime toggles, workspace, MCP marketplace.
- Global: light/dark theme, keyboard shortcuts (`?` help, Cmd/Ctrl 1-5 views, Cmd/Ctrl N new chat, Cmd/Ctrl Enter send, Esc close modals), onboarding + guided tour.

## Production readiness

- No hardcoded `/Users/savag3` or dev paths remain in source. LaunchAgent plists use `__REPO_ROOT__` and `__HOME__` placeholders that installers substitute at install time.
- `package_full_release.sh` excludes dev artifacts (`.cargo`, `.DS_Store`, `state.*`, `state-backup*`, `sapient_agi_soul*`, `test_*.wasm`, `test_cage`, `wild_workspace`, `strategy_db`, `data`, `voices`, `curriculum`, `com.badapple.substrate*` legacy plists, and `install_daemon.sh`).
- CI-style suite:
  ```bash
  cargo fmt --check && cargo build --release && cargo test --release
  ```
- Full release package:
  ```bash
  src/platform/apple_desktop/package_full_release.sh
  ```
  Produces `target/release/Bad_Apple-<version>-full-unsigned.zip`.
- Build dependencies are managed by `Cargo.lock` and the Swift package
  dependencies. No additional language runtimes are required.
- Support diagnostics:
  ```bash
  target/release/badapple --doctor
  ```
  Prints a redacted report of host, binaries, sockets, launchd jobs, and model cache.
- Signed / notarized release:
  ```bash
  CODESIGN_ID="Developer ID Application: ..." \
  APPLE_ID="..." \
  APPLE_TEAM_ID="..." \
  APPLE_APP_PASSWORD="..." \
  src/platform/apple_desktop/package_signed_release.sh
  ```
  See `SIGNING.md` for the full code-signing and notarization path.
- CI: `.github/workflows/ci.yml` runs `cargo fmt`, `cargo clippy`, `cargo build`, `cargo test`, and the full release package on every push/PR.

## Distribution packaging

- Drag-to-Applications DMG:
  ```bash
  src/platform/apple_desktop/package_dmg.sh
  ```
  Produces `target/release/Bad_Apple-<version>.dmg` with `Bad Apple.app`,
  the platform tree, an `Applications` alias, and an `Install.command` that
  copies the app to `/Applications` and installs the system LaunchDaemons.

- Homebrew Cask tap:
  ```bash
  src/platform/apple_desktop/package_homebrew_cask.sh
  ```
  Generates a local tap in `target/release/homebrew-bad-apple` from the
  canonical `homebrew-bad-apple/Casks/bad-apple.rb`. The canonical tap is
  configured for a GitHub release; the local tap points at the freshly built
  unsigned zip for testing:
  ```bash
  brew tap local/bad-apple /Users/savag3/bad_apple/target/release/homebrew-bad-apple
  brew install --cask bad-apple
  ```

## Lazy main-model loading

- Set `BADAPPLE_LAZY_MAIN_MODEL=1` in `com.badapple.mlx.plist` to skip loading
  the 9B brain at daemon startup. The first non-fast-tier request calls
  `_ensure_main_model()` and loads the brain on demand.
- `BADAPPLE_FAST_TIER=1` keeps simple queries (math, time, identity, greetings,
  `ping`, jokes, thanks) on the fast tier so the UI is responsive while the
  9B model is still absent.
- When lazy mode is on and `BADAPPLE_FAST_MODEL` is unset, the fast tier falls
  back to the cached `mlx-community/Qwen2.5-0.5B-Instruct-4bit` model.
- The `runtime_status` dashboard payload now includes `main_model_loaded` and
  the health `main_model` readiness reflects the lazy state.

## Background model manager

- The background model manager tracks four models: `main_9b`, `fast_0.5b`,
  `vision_2b`, and `flux_4b`. Each has download status (`missing`, `queued`,
  `downloading`, `cached`, `loaded`, `error`) and progress.
- Downloads are disabled by default. Set `BADAPPLE_ALLOW_DOWNLOADS=1` to enable
  them.
- The manager temporarily overrides `HF_HUB_OFFLINE=1` only inside the download
  thread, so the rest of the daemon stays air-gap certifiable by default.
- Use `badapple model list|scan|info|use|verify|add|remove|recommend` from the
  CLI, or the menu-bar model selector, to manage and pre-download models.

## `tools/` and generated artifacts

- `tools/` was gitignored for a long time under a stale "Generated tool sandbox
  files" rule that no longer matched reality -- if you add real, reusable
  scripts there, check `git ls-files tools/` actually tracks them, not just
  that they exist on disk.

## MLX daemon lifecycle and memory

- The Swift `badapple-engine` daemon installs `SIGTERM`/`SIGINT` handlers that
  cancel the run loop gracefully. Do not remove this: without it,
  `launchctl unload`'s SIGTERM kills the process at the OS level and can leak
  Metal resources or leave stale sockets.
- The daemon sets the MLX memory/cache limit at startup to
  `device_info().maxRecommendedWorkingSetSize` (Apple's own guidance for this
  GPU) rather than MLX's default of 1.5x that value. On a 16 GB Mac the default
  lets the daemon claim ~15.2 GB, leaving under 1 GB guaranteed for the OS and
  everything else. Don't remove this, and don't hardcode a GB value if you
  touch it -- `device_info()` scales correctly across different Macs.
- `draft_accept_ratio=0%` in the `[perf]` log line does **not** by itself
  mean speculative decoding is active and failing -- it reads exactly 0%
  whenever the draft model is `None` too (the default: neither
  plist sets `BADAPPLE_SPECULATIVE_DRAFT`). Grep the log for
  `[speculate] draft model loaded` to confirm whether a draft model
  actually loaded before concluding anything about acceptance rates.
- If a benchmark shows Bad Apple slower than expected, check `top`'s
  `PhysMem` line (compressor size, "unused" figure) and `vm_stat`'s
  cumulative Swapins/Swapouts before assuming it's a code bug -- this dev
  Mac has 16 GB total and routinely runs low on genuinely free memory with
  an IDE/agent session open alongside the model, which alone is enough to
  explain a real, reproducible slowdown that isn't Bad Apple's fault.
- The menu bar `PiperTTSClient` streams chunked TTS through `PiperTTSPlaybackController`, which queues WAVs and crossfades consecutive chunks with a 50 ms volume ramp. If `AVAudioPlayer` fails, it falls back to the previous `afplay` path. Apple TTS fallback for a full Piper failure remains in `BadAppleVoiceHost`.

## Cognitive architecture validation

- The cognitive architecture benchmark (connectome, hyperdimensional core,
  dual-process governor) is being ported to Rust/Swift and is not currently
  available from the command line.
- When it is available, it will run the same set of prompts (simple / medium /
  complex) in three modes: `cognitive_full` (full cognitive stack),
  `fast_tier_only` (0.5B model, no cognitive layer), and `9b_only` (9B brain,
  no cognitive layer, no fast tier).
- Each mode is selected with env vars: `BADAPPLE_COGNITIVE` toggles the
  cognitive layer and `BADAPPLE_FAST_TIER` toggles the fast 0.5B tier.
- Metrics captured per query: latency, token count, decode tok/s, and tier.
  Results will print as a comparison table and can be saved as JSON.
- It will use the CLI's `--json` stream, so the `badapple` binary must be built
  (`target/release/badapple`) or on `PATH`. Each query has a 120 s timeout.

## Model versioning

The main brain is loaded by repo name (`BADAPPLE_MAIN_MODEL`, e.g.
`caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit`) in `badapple-engine`. Loading
by repo name alone resolves to whatever commit HuggingFace currently serves as
`main`, so an upstream update or yank can swap the weights without notice. That
silently breaks the semantic-cache embeddings, token vectors, and ANE shard
manifest, which are all pinned to a specific snapshot.

To prevent this, the Swift MLX daemon pins to a revision when loading the model:

- `DEFAULT_MODEL_REVISION` (constant in the Swift MLX daemon) is the
  hardcoded fallback. It currently defaults to `"main"`; **replace it with an
  actual commit hash** for production pinning.
- `BADAPPLE_MODEL_REVISION` (env var, exposed in
  `src/platform/apple_bridge/com.badapple.mlx.plist`) overrides the constant at
  runtime. An empty/unset value falls back to `DEFAULT_MODEL_REVISION`. Set it to
  a branch, tag, or 40-char commit hash to pin a specific snapshot.

After every load, `_verify_model_integrity()` computes a SHA-256 of the loaded
model's `config.json` and logs it. The hash is persisted to
`/var/lib/bad_apple/model_config_hash.json` (next to the semantic cache). On the
next load, if the hash no longer matches, the daemon logs a warning that the
semantic cache, KV cache, and ANE shard manifest may be stale and should be
cleared. The check is best-effort: a missing or unreadable `config.json` is
logged and skipped, never blocks startup.

To pin a model after first download, capture the current commit hash from the HF
cache snapshot dir (`~/.cache/huggingface/hub/models--<org>--<model>/snapshots/`)
and set `BADAPPLE_MODEL_REVISION` to it, then restart the daemon.

## Swift-native MLX runtime

- The native inference package is `src/platform/apple_desktop/MLXInference`.
  Build its checks with `swift run -c release BadAppleMLXSelfTest` from that
  directory.
- `mlx-swift` 0.31.6 embeds MLX core 0.31.1 and requires the matching
  `mlx-metal==0.31.1` `mlx.metallib`. The menu build caches the verified shader
  at `~/.cache/badapple/mlx-metal-0.31.1/mlx.metallib` and packages it beside
  `libBadAppleMLX.dylib` under `Contents/Libraries/`.
- Do not put the bare MLX dylib in `Contents/Frameworks/`; AppKit treats entries
  there as framework bundles during Accessibility loading. Keep both the dylib
  and metallib in `Contents/Libraries/`.
- Local builds use a deep ad-hoc signature even when `BADAPPLE_NO_SIGN=1` so
  macOS validates all nested MLX code consistently.
- Before replacing `/Applications/Bad Apple.app`, unload the menu LaunchAgent;
  reload it only after replacement. macOS launchd records a launch constraint
  for the prior code hash and otherwise enters a restart/rejection loop.
- Do not launch permission-sensitive builds through an IDE automation process
  when testing TCC prompts. macOS may attribute the permission request to the
  responsible IDE process instead of Bad Apple. Use the registered LaunchAgent.
- A successful native smoke test logs `Native Swift MLX engine loaded` and a
  Swift audit-ledger response with a nonzero `tps` value.
