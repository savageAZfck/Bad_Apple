# Changelog

All notable changes to Bad Apple are documented in this file.

## [0.3.0] — 2026-09-19

### Added
- **Council of Minds** (`BadAppleCouncil.swift`): a 14-seat deterministic
  deliberation layer that votes on every gated action before it runs.
  Seats: four financial minds (Buffett, Dalio, Musk, Jobs) and ten
  strategists (Sun Tzu, Clausewitz, Musashi, Machiavelli, Napoleon,
  Hannibal, Aurelius, Boyd, Genghis Khan, Patton). Each proposed action is
  encoded into an 8-feature vector — destructiveness, irreversibility,
  blast radius, sensitivity, privilege, scope, cost, novelty — every seat
  votes with an in-voice rationale, and weighted consensus yields
  approve/deny/abstain plus a dissent score.
- **Autopilot gating**: under autopilot, a passed council vote executes
  and a failed or contested vote escalates to a human approval proposal
  instead of running. In manual mode the verdict rides on the approval
  prompt as counsel. The council votes — the human stays the final
  authority on anything contested.
- **`council <question>`** command: semantic mode voices all fourteen
  seats through the local model, then returns a tally verdict. `council`
  alone prints the roster.
- **Journaled deliberation**: every council vote lands on the audit
  ledger as `council_deliberation` (tool, args, all fourteen votes with
  rationales, consensus mean, dissent, mode) and failed votes under
  autopilot also emit `council_escalated`. The record shows not only what
  the AI did, but what fourteen minds concluded first — including what
  they refused.

## [0.2.3] — 2026-09-17

### Security
- **Tool-jail hardening.** `runShell` path-like arguments are now confined
  through `jailPath` with the working directory pinned to the user home —
  previously an approved `cat`/`find`/`grep` call could read files outside
  the jail (including the v1 SLICKS key). `runAppleScript` now also denies
  native file verbs (`open for access`, `read file`, `POSIX file`,
  `load/store script`) that bypassed the shell filter entirely.
- **Replay cache.** The cache now FIFO-evicts the oldest entry at capacity
  instead of clearing all entries — a flood inside the skew window could
  previously make captured frames replayable.
- **Resource bounds.** MCP request lines are read through a bounded
  `Read::take` (an unterminated line could exhaust memory before the size
  check ran), and both the gatekeeper and MCP server cap concurrent
  connections with a panic-safe decrement guard.
- **Installer plist escaping.** Values substituted into LaunchDaemon
  plists are XML- and sed-escaped before rendering (`plutil -lint`
  validation retained).

### Fixed
- **Boot-time socket lockout.** When the gatekeeper LaunchDaemon started
  before login, `/dev/console` was still root-owned, console-user
  resolution failed, and a freshly created `/var/run/badapple` stayed
  `root:daemon` — locking out the identity agent, MLX daemon, and CLI
  until restart. Console-user resolution now falls back to scutil's
  `State:/Users/ConsoleUser`, the last resort hands the directory to the
  `staff` group, and a periodic repair pass converges ownership once a
  session exists.

### Added
- **`badapple status`** — a three-line "is it working?" check for humans
  (daemon reachability, identity agent, version), distinct from the
  machine-readable `cert` suite and the full `--doctor` report.
- **Startup wait.** `badapple` now waits up to ~90s for the daemon socket
  and the model to come up instead of failing instantly after boot/login
  (`BADAPPLE_NO_WAIT=1` restores fail-fast behavior).
- **Plain-language errors.** Connection failures now say what happened and
  what to do (app not running vs. socket permissions vs. still starting),
  and the VRAM admission error tells the user to close apps and retry —
  the model loads automatically on the next query.

## [0.2.2] — 2026-09-13

### Added
- **IFY watchdog** (`badapple-ify`): a behavioral daemon that tails the
  audit ledger, verifies each new line against the hash chain as it
  arrives, and learns deterministic baselines (per-type hourly rates,
  tool frequencies, approval outcomes). Anomalies become findings on a
  severity ladder — findings log only, then approval-gated proposals and
  notifications, then (in autopilot phase) a kill-switch brake plus
  SAFE_MODE runtime state for critical events like a chain break or
  ledger truncation. Detection is statistical, not model-based;
  optional plain-English narration renders findings through the fast
  tier. See `IFY.md`.
- Phases: 14-day `gestation` (silent learning) → 14-day `secondary`
  (proposals + notifications) → `autopilot` (brake unlocked). Tunable
  via `BADAPPLE_IFY_GESTATION_DAYS`/`BADAPPLE_IFY_SECONDARY_DAYS`;
  `BADAPPLE_IFY_PHASE` forces a phase; `BADAPPLE_IFY=0` disables.
- `com.badapple.ify` LaunchAgent installed by
  `install_badapple_platform.sh`; state in `~/.bad_apple/ify/`.
- `badapple ify <status|once|findings|proposals>` CLI verb.

## [0.2.1] — 2026-09-13

### Added
- **Sovereign ledger layer** (`badapple-sovereign`): an independent, hardened
  copy of the audit ledger built on the public `sovereign_ledger` crate. Each
  run re-verifies the primary ledger (all three historical formats) and
  rewrites `/var/lib/bad_apple/ledger.sovereign.jsonl` as an HMAC-SHA256
  hash chain with Merkle roots, then signs checkpoints for both chains via
  the identity agent (Secure Enclave).
- `com.badapple.checkpoint` LaunchAgent: runs the sovereign hardening pass
  daily; installed automatically by `install_badapple_platform.sh`.
- `badapple cert` now fails when the sovereign checkpoint is missing, stale
  (>36 h), or future-dated, so a stopped checkpoint agent is loud.

### Changed
- `sovereign_ledger` dependency is pinned to a git rev on the public repo
  instead of a local path, so packaged and CI builds resolve identically.

## [0.2.0] — 2026-09-10

### Security
- `mcp_marketplace.rs` now rejects `npx`, `npm`, `pip`, `curl`, `wget`, `git`, `ssh`, `scp`, `ftp`, `telnet`, and any command containing shell metacharacters or `..`/`~` in arguments. This prevents the marketplace from being used to install remote-download or overly broad filesystem MCP servers without explicit local validation.
- Removed the built-in `npx` filesystem/fetch MCP catalog defaults entirely.

### Fixed
- `BadAppleEngine.toolRequiresApproval` now respects the `autopilot` level, so full autopilot correctly skips approval prompts in the UI path as well.
- Removed the dead `/var/lib/bad_apple/autopilot` override file. Autopilot is now derived from `~/.bad_apple/autopilot_level` consistently.
- `badapple --doctor` now annotates the missing `mcp.sock` as "(MCP off by default)" instead of reporting a bare failure.
- Curious `readWorkingMemory`, `listDirectory`, `recordCuriousFeedback`, `appendCuriousBuildLog`, and patch backup/parent-directory creation now surface errors instead of failing silently.
- Curious rollback paths in `applyProposedPatch` now report rollback failures explicitly instead of swallowing them with `try?`.

### Hardened
- Replaced all `try!` and `fatalError` cases in `BadAppleMenuBar.swift` with safe optional regex compilation and `return nil` from unavailable `init(coder:)` paths.
- Replaced production `print()` calls in `BadAppleEngine.swift`, `BadAppleModelManager.swift`, and `BadAppleMenuBarUIResponder.swift` with `NSLog` so logs are captured by the system instead of leaking to stdout.
- Replaced hardcoded `Regex::new(...).unwrap()` chain in `sanitize_for_tts` with a single `tts_regex` helper that falls back to a non-matching regex if a static pattern ever fails to compile.
- Hardened `badapple-dashboard` response builders and `badapple-supervisor` JSON serialization against panic on unexpected builder/serialization failures.

## [0.1.9] — 2026-09-10

### Security
- `badapple mcp init` no longer pre-populates the catalog with `npx`-based MCP
  servers (`@modelcontextprotocol/server-filesystem` with root `/` and
  `@modelcontextprotocol/server-fetch`). The product is air-gapped by default;
  remote-download MCP servers should be added explicitly only after local
  validation.

## [0.1.8] — 2026-09-10

### Fixed
- Menu bar `autopilot` toggle now sets `full` (not `safe-apply`), so it matches the documented behavior of skipping approval prompts for all destructive tools. Use the dashboard for `suggest`/`safe-apply` levels.
- `mcp.rs` server version now reports `0.1.8`.
- `deriveRuntimeRepairs` no longer suggests a malformed `cp` command for a missing app install.
- README and BAD_APPLE_BUYERS.md updated to reflect current features.

## [0.1.7] — 2026-09-10

### Fixed
- Menu bar `autopilot` toggle now sets `full` (not `safe-apply`), so it matches the documented behavior of skipping approval prompts for all destructive tools. Use the dashboard for `suggest`/`safe-apply` levels.
- `mcp.rs` server version now reports `0.1.7`.
- `deriveRuntimeRepairs` no longer suggests a malformed `cp` command for missing app installs.
- README and BAD_APPLE_BUYERS.md updated to reflect v0.1.7 features.

### Added
- **Real-world runtime repairs** — `self_audit` now detects common install/runtime issues (missing data dir, missing output firewall blocklist, unloaded identity/dashboard/TTS/menu bar agents, missing app install) and emits a ranked `repairs` list.
- **`repair_runtime_issue` tool** — a bounded, allowlisted tool that can create the data dir, create the blocklist, or `launchctl bootstrap` a user LaunchAgent. Unsafe repairs (like app install) are surfaced for human approval.
- **Curious now repairs before it patches** — in `safe-apply` and `full` levels, `curious_self_improve` attempts every safe runtime repair first, records the outcome, and only then proposes a source patch for the remaining issue.
- **Event-driven Curious triggers** — the autopilot loop now wakes on `BadAppleCuriousTrigger` notifications instead of only a timer. Triggers fire on engine startup, model load failure, missing output firewall blocklist, and `self_audit` detecting runtime repairs.
- **Autopilot loop sync** — calling `curious_self_improve` or toggling autopilot now keeps the background Curious loop in the right state so dashboard-level changes take effect without a restart.

### Changed
- `BadApple.app` version reported by `badapple --doctor` now reflects `0.1.7`.

## [0.1.6] — 2026-09-10

### Added
- **Curious autopilot is now a product, not a feature**:
  - Patches are verified with `cargo fmt`, `cargo clippy`, `cargo build --release`, and `cargo test --release` before they are allowed to stay applied.
  - Any patch that fails verification is automatically rolled back to its backup.
  - The dashboard and the daemon both run this verification pipeline, so autopilot and human apply are held to the same standard.
- **Autopilot levels**: `off`, `suggest`, `safe-apply`, and `full`.
  - `off` disables the loop.
  - `suggest` writes proposals for human review.
  - `safe-apply` auto-applies only non-control, verifiable patches.
  - `full` auto-applies all verifiable patches.
- **Curious feedback and few-shot learning**: accepted, rejected, and failed patches are recorded in `~/.bad_apple/curious_feedback.json`. New prompts include the last accepted examples and recent failure reasons so the model learns the project style and safety bounds.
- **Control Center Curious tab refresh**: shows autopilot level, patch status badges, inline diff, and Rollback/Archive/Apply actions.
- **Runtime self-repair signals**: `self_audit` now reports whether `/Applications/Bad Apple.app`, the identity agent, the dashboard agent, and the background engine are present and loaded.
- **Manual Curious trigger**: `POST /api/curious_trigger` and a `Run Curious now` button let the user force a self-improvement cycle from the dashboard.
- **Curious build log**: every applied/failed patch is appended to `~/.bad_apple/CURIOUS.md` for an auditable history.

### Changed
- Protected-file list for autopilot patches expanded to include `BadAppleConversation.swift`, `BadAppleTTS.swift`, `badapple-dashboard.rs`, and `lib.rs`.

## [0.1.5] — 2026-09-09

### Added
- **Custom tools are first-class native tools**: the engine loads workshop custom
  tools from `~/.bad_apple/custom_tools.json`, exposes
  them in tool selection and model tool schemas, and executes shell, AppleScript,
  and macOS Shortcut custom tools through the existing policy cage.
- **MCP page and API compatibility**: `/api/mcp_servers` now supports direct
  REST list/add/remove, and `web/mcp.html` has been rewritten to match.
- **Curious autopilot proposal UI**: the Control Center now lists pending patch
  proposals from `~/.bad_apple/notes/proposed_patches/`, shows affected file,
  old/new text, reason, creation time, and status, and provides human-in-the-loop
  Approve/Reject/Dismiss/Refresh actions with safe old-string, backup, and path
  root checks.
- **Control Center is the single cockpit**: model listing/switching and MCP
  server add/remove are now available from `/control` alongside Overview, Memory,
  Curious, and Workshop.
- **Persona voice and roast preview**: the Workshop persona editor adds a Preview
  panel for local system-prompt inference, roast-bank testing, and local TTS
  preview when TTS is available.
- **Dashboard ships in the consumer package**: `badapple-dashboard` and `web/`
  assets are packaged with the unsigned release and installed/launched for
  normal users without a repository checkout.
- **CSRF protection for the web dashboard**: all mutating `/api/*` routes require
  a valid `X-CSRF-Token` header; `GET /api/csrf` supplies the token and the
  bundled `csrf.js` refreshes it automatically for every page.

### Changed
- Custom tool definitions are read from the same path the dashboard writes them
  to, and they are hot-reloaded by the engine using mtime checks.

## [0.1.4] — 2026-09-09

### Added
- **Control Center** (`/control`) is now a real web dashboard with three tabs:
  Overview, Memory, and Workshop.
- New dashboard routes: `/api/snapshot`, `/api/tail`, `/api/ledger`, `/api/audit`,
  `/api/cert`, `/api/doctor`, `/api/voice`, `/api/capabilities`, `/api/control`,
  `/api/workspace`, `/api/working_memory`, `/api/memory/facts`, and
  `/api/workshop/personas` / `/api/workshop/custom_tools`.
- **Automatic local fact extraction**: the dashboard reads text or workspace files,
  asks the local 7B Qwen model for JSON `subject-predicate-object` triples, and
  writes them atomically to `/var/lib/bad_apple/memory_graph/facts.json`.
- **Memory tab**: working memory editor, searchable fact bank, manual fact
  addition, single-file extraction, and bounded recursive directory indexing.
- **Workshop tab**: create, edit, delete, and switch to local personas stored in
  `~/.bad_apple/personas.json`; create and run declarative shell, AppleScript, or
  macOS Shortcut tools stored in `~/.bad_apple/custom_tools.json`.
- Dashboard status adapter now exposes `autopilot`, `fast_tier`, `private_mode`,
  `airgap`, `killed`, and `workspace` at the top level for the UI.
- New `generateRaw(prompt:systemPrompt:maxTokens:temperature:)` engine method for
  deterministic, persona-free structured generation, used by fact extraction.
- `inference` agent call accepts `system_prompt` and `temperature` parameters.
- `/api/mcp_servers` and `/api/mcp_registry` aliases for the existing MCP routes.

### Changed
- `BadAppleEngine.switchPersona` now reloads persona files before switching so
  workshop-created personas are active immediately.

## [0.1.3] — 2026-09-08

### Added
- `curious_self_improvement` now reasons with the local 7B model. It analyzes cert,
  doctor, output firewall, git status, and TODO/FIXME/HACK/XXX source markers and
  outputs either a concrete `{"patch":{"file","old","new","why"}}` or
  `{"no_patch":true}`.
- Autopilot can apply its own bounded patches: jail-checks the path, requires the
  exact `old` string, backs up the original, writes the replacement, and verifies
  the result. New files are supported with `"old":""`.
- Core control files (`BadAppleEngine.swift`, `BadAppleTools.swift`,
  `BadAppleEngineDaemon.swift`, etc.) are protected from autopilot edits.
- Output firewall automatically creates a missing `/var/lib/bad_apple/blocklist.txt`
  so the cert suite no longer reports it absent.

### Changed
- `curious_self_improvement` now runs with a fixed engineering system prompt and
  a structured JSON output format instead of a hand-wavy audit dump.
- Self-improvement proposals are now logged with the model's actual patch
  proposal and the apply result (applied, refused, or error).

### Fixed
- Empty-file verification in `applyProposedPatch` now handles `"new":""` correctly.
- `curious_self_improvement` finds the project root from the running binary
  instead of defaulting to `~`.

## [0.1.2] — 2026-09-07

### Fixed
- Deterministic control-phrase handling for `kill switch`, `stop everything`, `resume bad apple`,
  `resume`, `emergency stop`, and safe-mode exit phrases. These now bypass the model and
  engage the engine kill-switch directly.
- Deterministic natural-language tool invocation for `run shell ...`, `run command ...`,
  `write a note ...`, and `generate an image of ...`. Requests are routed directly to the
  tool cage and require approval when autopilot is off.
- Tool-call prompt examples now produce valid `<tool_call>` XML without duplicated tool names.
- Moved the tools and control phrases instruction to the top of `prompt.txt`.
- Certification self-tests locate sibling binaries relative to the current executable, so
  `badapple cert` passes from the installed CLI as well as from `cargo test`.
- Replaced model-name-based memory cutoffs with estimates derived from the resolved
  cached weights, model dimensions, configured KV cache, and prefill workspace.
- Preserved live memory admission checks and released the original reservation
  after unloading a model or a failed load.
- Voice helper failures now preserve stderr and wait for both output streams to
  drain; partial output no longer hides a failed request.
- Model-loading failures report the underlying cause, and failed model switches
  are no longer marked as loaded. Load/unload operations are serialized.
- Manual installs keep runtime files in a persistent versioned location, stop
  the menu bar before app replacement, and retain rollback backups.
- Updates install the complete app and engine together and require SHA-256
  verification. A compatibility archive supports the older updater layout.
- Packaging rejects missing/stale runtime artifacts and private Swift/Rust source.

## [0.1.1] — 2026-09-06

### Changed
- Curious self-improvement autopilot is now wired to the same toggle as autopilot.
  Enabling autopilot in the menu bar or settings starts periodic `curious_self_improve`
  checks; it no longer requires a separate `curious` persona.
- Menu bar and settings labels renamed from "Auto-Run Commands" to "Autopilot".

## [0.1.0] — 2026-08-29

### Added

#### Core AI Runtime
- 9B Qwen 3.5 4-bit model on Apple Neural Engine / GPU via MLX
- Optional speculative decoding with DFlash and MTP draft models
- 0.5B fast tier for simple queries (math, identity, time, greetings)
- Dual-process cognitive governor (576-D CandleBrain + 10,000-D hyperdimensional VSA)
- Memory-mapped connectome persistence (2048-D embeddings, zero-copy)
- Semantic cache with BAAI/bge-small-en-v1.5 embeddings
- Streaming output with token-by-token generation
- Persona pack system (Default, Wicket, Gen Z, Drill, Midwest)
- Voice mode with Piper TTS and on-device speech recognition
- Siri integration via BadAppleIntent
- macOS Shortcuts integration via aqua helper

#### Security Architecture
- SLICKS v1 (HMAC-SHA256) and v2 (Secure Enclave ECDSA) IPC authentication
- Fail-closed filesystem automation cage with openat-based path operations
- WebAssembly sandbox with fuel metering, StoreLimits, and output caps
- Hash-chained audit ledger with Secure Enclave-signed checkpoints
- Streaming output firewall with Aho-Corasick secret redaction
- Air-gap certification suite (12 tests, zero network listeners)
- P2P encrypted mesh sync (AES-256-GCM, link-local, off by default)
- Human-in-the-loop approval policy engine (37 rules in policy.yaml)
- APFS file scavenger with tokenized chunking and Sled persistence
- Bounded health supervisor with restart budgets and safe mode
- Server key pinning (TOFU trust store at /var/lib/bad_apple/keys/daemon.pub)
- Replay cache in gatekeeper (nonce deduplication within freshness window)

#### System Infrastructure
- Three launchd daemons: gatekeeper, MLX server, health supervisor
- Rust gatekeeper proxy on SLICKS-authenticated Unix socket
- 13-actor runtime (audit, breakers, cache, health, MCP, metrics, model, P2P, persona, resources, task, workspace)
- Apple Neural Engine bridge via multi-shard FFI with KV cache scatter
- Apple Intelligence Foundation Models bridge via dlopen
- Metal UMA zero-copy memory management
- MCP server on Unix socket (/var/run/badapple/mcp.sock)
- Local web dashboard at http://127.0.0.1:8787
- Model version pinning with commit hash and integrity verification

#### UI / UX
- Menu bar app with persona switching, voice, benchmarking, and diagnostics
- Onboarding wizard (5-step: welcome, privacy, model status, permissions, first query)
- Chat window with message bubbles, markdown rendering, code blocks, streaming cursor
- Multi-line input with Enter to send, Shift+Enter for newline
- Dark theme matching macOS visual effect views
- Diagnostics command (badapple --doctor / --diagnostics)

#### Tooling
- 4 fuzzing targets (IPC frame, WASM cage, protocol frame, scavenger path)
- Cognitive architecture A/B benchmark script
- Clean-machine install test (32 automated checks)
- Homebrew Cask formula for distribution
- Unsigned DMG with Install.command and quarantine stripper
- Updater with GitHub release comparison and rollback

### Security Audit

Two full security audit passes were conducted. 69 vulnerabilities were found and fixed:

#### Pass 1 (30 bugs)
- Gatekeeper WASM path bypass and symlink bypass
- P2P predictable secret and replay attacks
- Non-constant-time HMAC comparison
- WASM cage memory policy bypass and string ABI issues
- Connectome mmap race condition and corrupt header panic
- NaN panics in tensor brain and divide-by-zero in conscience oracle
- Aqua helper unauthenticated socket with world-writable permissions
- Shell allowlist bypass via absolute path
- SLICKS secret file permission gaps
- P2P bound to 0.0.0.0 (should be 127.0.0.1)
- World-writable /var/run/badapple directory
- Path traversal in IPC response writing
- dlopen hijacking via CWD and parent directories
- dlsym null → unsafeBitCast crash
- Pipe deadlock in process management
- Timer on background queue (never fires)
- MLMultiArray use-after-free in ANE inference
- Hardcoded version in Info.plist
- Mixed codesign --verify and --sign flags
- eval on user input in installer
- Hardcoded staff group in installer
- launchctl bootstrap on incompatible machine
- Root-context launchctl for console user app
- AppleScript injection in Mail/Calendar/Reminders
- CSRF timing leak in dashboard
- Notarization workflow staples without checking status
- Scavenger TOCTOU in file indexing

#### Pass 2 (39 bugs)
- SLICKS v2 authentication bypass (server accepts any client_pubkey from Execute frame)
- policy.yaml ships with autopilot: true by default
- AppleScript injection in accessibility_action (type/click/menu)
- AppleScript injection in aqua_helper (backslash not escaped before quote)
- Path traversal in read_file, list_directory, search_content (no path jailing)
- MLMultiArray argmax out-of-bounds read (byte stride used as element index)
- SLICKS secret loaded from environment variable (injection vector)
- Non-constant-time HMAC comparison in Swift UI responder
- WASM cage no StoreLimits (memory.grow bypasses 1 MiB policy)
- P2P adapter zip slip (peer-supplied adapters_dir + unsanitized name)
- CopyFile source TOCTOU (fs::copy follows symlinks after validation)
- Scavenger follows symlinked directories (collect_files uses is_dir)
- WebSocket frame size unbounded (no MAX_FRAME_BYTES check)
- MAX_ENGRAM_TEXT_CHARS defined but never enforced
- Hyperdimensional encoder O(n × 10,000) DoS (unbounded input)
- Tensor brain no input validation (NaN/Inf and wrong-length inputs)
- Connectome HEADER_SIZE + record_bytes unchecked overflow
- Hypervector.values public (attacker can create wrong-length vectors)
- Peer spec SSRF to 169.254.169.254 (cloud metadata)
- CSRF token public with no Origin/Referer check
- Audit ledger empty HMAC secret by default
- Swift full environment inheritance to child processes
- Aqua helper searched from ~/.bad_apple (user-writable)
- Gatekeeper socket 0o666 (world-writable)
- Gatekeeper doesn't validate client_nonce in Hello
- WASM output vector unbounded (no total cap)
- WASM alloc bump pointer corruption (validates after mutating)
- Browser action allows file:// and smb:// schemes
- Screen capture writes to user-supplied path
- Shell allowlist includes interpreters (swift, cargo, rustc, git)
- P2P retry backoff powi sign flip (u32 to i32 cast)
- Aqua helper no replay/nonce protection
- /tmp debug log pre-creation with world-readable permissions
- Non-ASCII confusables bypass reject_lexical_path
- P2P replay after restart (nonce set lost on restart)
- P2P v1 HMAC allows origin spoofing with v2 public key
- Dashboard /api/control can toggle autopilot without identity check
- Output firewall check_full stateless (misses patterns split across chunks)
- Ledger verify doesn't validate Secure Enclave checkpoint signature

### Fuzzing

4 fuzzing targets were created and run for 2 hours each (127+ million total iterations) with zero crashes:
- fuzz_ipc_frame: SLICKS frame parsing, nonce validation, timestamp freshness (423K iterations)
- fuzz_wasm_cage: WASM compilation and execution with arbitrary bytes (638K iterations)
- fuzz_protocol_frame: P2P signed packet parsing, HMAC verification (694K iterations)
- fuzz_scavenger_path: Path handling, canonicalization, NUL bytes (760K iterations)

### Test Coverage

- 78 Rust unit tests (including 25 security regression tests)
- 32 clean-machine install checks
- 4 fuzzing targets with corpus
- Cognitive architecture A/B benchmark

### Performance

| Metric | Value |
|---|---|
| First-token latency | 3.5–6.5s for 450–750 token prompts |
| Decode throughput | 13–25 tok/s, spikes to ~36 tok/s |
| Peak memory | 5.7–6.5 GB with 9B + optional draft |
| Voice first token | 2.8–5.7s for 430–460 token prompts |
| Fast tier latency | <1s for simple queries |
| Cognitive routing overhead | ~0.7s classification per query |

### Known Limitations

- Unsigned by philosophical choice (no Apple notarization, no cloud upload)
- macOS / Apple Silicon only (total platform lock-in for ANE/Metal/SE access)
- Single model (9B Qwen 3.5 pinned to commit 5ae9734)
- No proactive reachout (reactive only, unlike Hermes/OpenAGI)
- No cross-platform support
- Solo project, no external contributors yet
