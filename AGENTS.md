# AGENTS.md — Bad Apple

This file captures the project-specific commands and conventions learned while working on Bad Apple, so the next agent (or future you) does not have to rediscover them.

## Project layout

- `badapple_mlx_server.py` — MLX daemon (Python). Loads the 9B Qwen 3.5 model, DFlash draft, and serves the SLICKS Unix socket.
- `src/bin/badapple.rs` + `src/` — Rust CLI client that talks to the daemon.
- `src/platform/apple_bridge/com.badapple.mlx.plist` — launchd daemon config.
- `prompt.txt` — Hot-reloadable system prompt. Edits take effect on the next query without restarting the model.
- `BAD_APPLE.md` — Technical overview and live performance numbers.
- `BAD_APPLE_BUYERS.md` — Buyer-facing pitch doc.
- `badapple_extras.py` — OS extras: persona packs, output firewall, audit ledger, semantic cache, approvals.
- `personas.json` — Runtime persona packs (default, wicket, genz, drill, midwest).

## Build

```bash
cargo build --release
```

## Lint

```bash
cargo fmt --check
cargo build --release
.venv/bin/python -m ruff check .
.venv/bin/python tests/test_smoke.py
```

## Full platform install

The installer replaces `/Applications/Bad Apple.app`, installs system LaunchDaemons, migrates `/var/lib/bad_apple` ownership, and loads the health supervisor:

```bash
cargo build --release
src/platform/apple_desktop/build_bad_apple_menu_bar.sh
osascript -e 'do shell script "cd /Users/savag3/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install" with administrator privileges'
```

It creates a rollback snapshot under `/var/lib/bad_apple/install_backups/` and restores the previous launchd configuration if readiness does not pass.

## Unsigned build and install

If no Apple Developer ID is available, build the menu-bar app unsigned and install without `codesign --verify` checks:

```bash
cargo build --release
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
osascript -e 'do shell script "cd /Users/savag3/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install --unsigned-install" with administrator privileges'
```

Package an unsigned release zip with a consumer README:

```bash
src/platform/apple_desktop/package_unsigned.sh
```

Produces `target/release/Bad_Apple-<version>-unsigned.zip`.

## Start / restart the daemon

The live setup uses the Rust `gatekeeper` on `/var/run/badapple/substrate.sock` as the front proxy. Restart both after a CLI/Rust change; only the MLX daemon needs a restart after a Python change.

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

Edit `prompt.txt`. The daemon hot-reloads it on the next query. Voice mode uses `VOICE_SYSTEM_PROMPT` inside `badapple_mlx_server.py`, which requires a daemon restart to change.

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
`/var/lib/bad_apple/ledger.jsonl` with SHA-256 chaining.  Verify it from Python:

```python
from badapple_extras import AuditLedger
AuditLedger(Path('/var/lib/bad_apple')).verify()
```

Secrets, emails, SSNs, phones, API keys, and long random tokens are redacted
before writing.

## Semantic cache

The first response to a question is embedded with `BAAI/bge-small-en-v1.5` and
stored.  Repeated semantically similar queries return the cached answer
instantly, scoped by active persona.  Set `BADAPPLE_CACHE_THRESHOLD` (default
0.92).  Cache file: `/var/lib/bad_apple/semantic_cache.json`.

## Human-in-the-loop approvals

Destructive tools (`run_shell`, `run_applescript`, `write_file`, `index_documents`)
require approval by default.  The API returns a proposal ID; reply with the CLI or `agent_client.py`:

```bash
# Use the agent protocol directly (no model parsing needed)
.venv/bin/python agent_client.py invoke run_shell '{"command":"ls /tmp"}'
# Approve with the CLI
target/release/badapple "approve <id>"
# Or skip approval for the session
BADAPPLE_AUTOPILOT=1 .venv/bin/python agent_client.py invoke run_shell '{"command":"ls /tmp"}'
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
# Run as root for full socket/process visibility
osascript -e 'do shell script "cd /Users/savag3/bad_apple && /Users/savag3/bad_apple/.venv/bin/python cert_suite.py" with administrator privileges'
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
```

Look for log lines like:

```
[perf] 31 tokens @ 20.1 decode t/s (5.4 total t/s), draft_accept_ratio=58%, block_tokens=6, peak_memory=5.72 GB
```

## Python venv

The venv must live on persistent storage. Do not place it on a RAM disk because macOS updates and reboots clear the disk and leave both MLX and TTS launch jobs failing with `EX_CONFIG`.

The live setup uses:

```bash
/opt/homebrew/bin/python3.12 -m venv ~/.local/share/badapple/venv
ln -s ~/.local/share/badapple/venv .venv
source .venv/bin/activate
```

Core packages include `mlx-lm`, `dflash-mlx`, `mlx-vlm`, `mlx-audio`, `mflux`, `piper-tts`, `langdetect`, `cryptography`, `psutil`, `PyYAML`, `pdfplumber`, and `EbookLib`.

## Current generation settings

Set in `src/platform/apple_bridge/com.badapple.mlx.plist`:

- `BADAPPLE_DFLASH=0` — DFlash is off for this quant. The 9B Qwen 3.5 uses a hybrid linear/full attention state cache (`ArraysCache`) that does not provide a trimmable KV cache, so the `mlx-lm` draft path cannot load a small draft. DFlash's own 9B draft model is too heavy to beat the verification overhead on the benchmark suite, so plain `mlx-lm` is faster overall.
- `BADAPPLE_DRAFT_MODEL=` (empty) — no speculative draft.
- `BADAPPLE_NUM_DRAFT_TOKENS=2` (unused when draft is empty).

In `badapple_mlx_server.py`:

- `prefill_step_size=4096` and `max_kv_size=4096` keep prefill in one shot and bound the KV cache.
- System prompt is hot-reloaded from `prompt.txt`; keep it compact to minimize TTFT.

## Common gotchas

- `BAD_APPLE.md` is the live technical doc; keep it in sync with the architecture and the latest benchmark numbers.
- `BAD_APPLE_BUYERS.md` is the buyer-facing doc; commit it when the numbers change.
- The 4B voice bundle has been removed in favor of the unified 9B brain; do not reintroduce it.
- `cargo fmt --check` is now clean; run `cargo fmt` and `cargo build --release` after Rust changes.
- Natural-language tool invocation (e.g. "run shell ls /tmp") is driven by the explicit `<tool_call>` XML instructions and few-shot examples in `prompt.txt`. The model emits the block, the CLI prompts for approval, and the user replies `approve <id>` to execute.

## Phase 3 Resource Governor

- `agent_client.py flush` clears the Metal allocation cache.
- `agent_client.py unload [vision|image|all]` drops optional models and kills stray `mflux` processes.
- `badapple "flush vram"` and `badapple "unload all models"` are the CLI control phrases that the menu bar sends.
- The menu bar app (`Bad Apple.app`) runs a `MemoryGovernor` that polls `host_statistics64` every 2 s and listens to `DispatchSourceMemoryPressure` critical events. When memory pressure crosses the configured critical threshold (default 0.95, overridable with `BADAPPLE_MEMORY_CRITICAL`), it automatically purges VRAM and unloads optional models.
- Build the menu bar app with `src/platform/apple_desktop/build_bad_apple_menu_bar.sh` and install the persistent agent with `src/platform/apple_desktop/install_menu_bar_agent.sh`.

## Phase 4 Agent OS Control Center

- `agent_client.py dashboard [--open]` prints and optionally opens the local dashboard at `http://127.0.0.1:8787`.
- `agent_client.py work <read|write|clear> [content]` manages the working memory scratchpad.
- `agent_client.py tier <on|off>` toggles fast tiering.
- `badapple_dashboard.py` serves a dark, auto-refreshing dashboard on `127.0.0.1:8787` with runtime, health, memory, active models, breakers, log tail, and audit ledger tail.
- Working memory tools: `read_working_memory`, `write_working_memory`, `clear_working_memory` are first-class tools in `prompt.txt` and `policy.yaml`.
- Fast tiering in `badapple_tier.py` routes simple math, identity, time, and greeting queries to the 0.5B fast model, bypassing the 9B brain. It is enabled by default via `BADAPPLE_FAST_TIER=1` in `com.badapple.mlx.plist` and can be toggled at runtime with `agent_client.py tier <on|off>` or the menu bar `Fast Tier Only` toggle.
- macOS Shortcuts can be listed/run through `list_shortcuts` and `run_shortcut` tools and the menu bar `Tools > Run Shortcut...`/`List Shortcuts`.
- `cert_suite.py` treats `127.0.0.1`/`::1` TCP listeners as local-only, so the dashboard does not fail the air-gap test.

## Lint

```bash
.venv/bin/python -m ruff check --select E4,E7,E9,F,B,UP,PLW1510,F821,DTZ005,DTZ006,F841,RUF013 . --exclude .venv --exclude target --exclude build --exclude __pycache__ --exclude web --exclude tests/ane_brain_perf
.venv/bin/python -m ruff check --select BLE001,S110 . --exclude .venv --exclude target --exclude build --exclude __pycache__ --exclude web --exclude tests/ane_brain_perf
.venv/bin/python -m compileall -q .
```

## Phase 5 Local Intelligence Mesh

- Autopilot: `agent_client.py autopilot <on|off>` or the menu bar `Tools > Autopilot` toggle. When on, destructive tools run without approval prompts.
- Ambient context: `agent_client.py ambient <on|off>` or menu bar `Mesh > Start/Stop Ambient`. Captures active app/window and (when GUI allows) screenshots locally. Dashboard and `runtime_status` show the latest context.
- Workspace / project mode: `agent_client.py workspace <path>` or menu bar `Mesh > Set Workspace...`/`Open Workspace`. Adds workspace context to prompts and the dashboard.
- P2P encrypted sync: `agent_client.py p2p <on|off|peers|sync>` or menu bar `Mesh > P2P Sync`. Link-local UDP/TCP, AES-256-GCM, off by default for air-gap certification.
- Local MCP server: `badapple_mcp_server.py` runs on Unix socket `/var/run/badapple/mcp.sock` and stdio, exposing tools/resources to MCP clients. Started automatically by the MLX daemon.
- Tiny fast model: `badapple_fast_model.py` loads a 0.5B Qwen MLX model for chitchat and short queries when `fast_tier` is enabled; deterministic handlers still handle math, identity, and time.
- macOS Shortcuts: `badapple_aqua_helper.py` runs in the menu bar's Aqua session and serves `/var/run/badapple/aqua_helper.sock`. `list_shortcuts` and `run_shortcut` now proxy through the helper, so they work even though the daemon is not in a GUI session.
- The dashboard (`http://127.0.0.1:8787`) displays ambient, workspace, P2P, MCP, fast model, autopilot, and fast-tier state.

## Model registry, tool router, chat dashboard, P2P (latest)

- Model registry in `badapple_model_registry.py` with CLI commands `list models`, `scan models`, `model info <id>`, and `use model <id>` for runtime LLM switching.
- Natural-language tool router in `badapple_tool_router.py` routes common requests directly to tools without waiting for the 9B.
- Web chat UI at `http://127.0.0.1:8787/chat`; REST endpoint `POST /api/chat`.
- P2P supports manual `p2p add peer <host>:<port>` and the sync beacon now carries the TCP sync port.

## Latest features (new)

- Streaming chat: POST /api/chat with `{"prompt": "...", "stream": true}` returns Server-Sent Events (tokens, tool calls, done, error).
- Automatic fact extraction: `badapple_fact_extractor.py` pulls name/like/location/work/etc. from every user message and stores normalized facts.
- Workspace file watcher: `badapple_workspace_watcher.py` scans `workspace` every 10s and auto-indexes changed .txt/.md/.py/.rs/.swift/etc. files into the RAG index.
- Image generation: say `generate an image of ...` or `draw a picture of ...` to use the cached FLUX.2-klein-4B model. The dashboard displays generated images via `/api/image/<filename>`.
- Persona editor: `http://127.0.0.1:8787/persona` lets you switch between personas and edit the active system prompt live. `prompt.txt` hot-reloads; `personas.json` changes are reloaded on the next `get_system_prompt`.
- MCP marketplace: `badapple_mcp_marketplace.py` is a minimal stdio MCP client for registering and invoking local MCP servers. Try:
  - `add mcp server <name> <command>`
  - `list mcp servers`
  - `list mcp tools <server>`
  - `invoke mcp tool <tool> on server <name> with text <arg>`

## New web UI (SPA) and native splash

- The dashboard is now a single-page app served from `web/index.html` with a unified sidebar, dark design system, and responsive layout.
- Routes: `/` (Dashboard), `/chat`, `/persona`, `/settings`, `/logs`, `/splash`.
- Static assets live in `web/static/` (styles.css, app.js) and are served by `DashboardHandler`.
- Chat view: streaming markdown, code blocks, tool-call cards, generated-image preview, scroll-to-bottom, auto-resize textarea.
- Settings view: workspace setter, MCP server list/add/remove, active models, runtime toggles (autopilot, fast tier, P2P).
- Dashboard view: status cards, P2P peers, latest perf, log tail.
- Onboarding modal shown once for new browsers.
- Native boot splash: `BadAppleSplashWindow` in `BadAppleMenuBar.swift` shows a progress bar on macOS app launch and auto-closes.

## Dashboard/chat UX (latest)

- Chat: conversation history saved in localStorage, new-chat, edit/retry/delete per message, copy code-block button, prompt suggestion chips, drag-and-drop file/image upload.
- Dashboard: live canvas chart for memory % and decode tokens/s, MCP server status, recent tool calls from ledger.
- Settings: model selector (list/switch), MCP tool invocation UI, theme toggle, runtime toggles, workspace, MCP marketplace.
- Global: light/dark theme, keyboard shortcuts (`?` help, Cmd/Ctrl 1-5 views, Cmd/Ctrl N new chat, Cmd/Ctrl Enter send, Esc close modals), onboarding + guided tour.
