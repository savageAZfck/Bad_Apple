# Bad Apple

> **A sovereign, local AI operating system layer for macOS.**
>
> On-device inference, hardware-rooted identity, fail-closed security, and a
> native Swift menu bar — with zero cloud round-trips after the models are
> downloaded once.

---

## Download the public beta

The latest public beta is **v0.1.2**:

- **[Download the consumer zip](https://github.com/savageAZfck/bad-apple-releases/releases/download/v0.1.2/Bad_Apple-0.1.2-unsigned.zip)** — unzip and run `sudo ./install.sh`
- **[View the release page](https://github.com/savageAZfck/bad-apple-releases/releases/tag/v0.1.2)**
- Or install via Homebrew:
  ```bash
  brew tap savageAZfck/bad-apple https://github.com/savageAZfck/homebrew-bad-apple
  brew install --cask bad-apple
  ```

## What Bad Apple Is

Bad Apple is a **local-first AI operating-system layer for macOS**. It is not a
chat app and it is not a cloud assistant. It is a set of `launchd` daemons,
native Swift/Rust services, and a menu bar that turn an Apple Silicon Mac into a
private, air-gapped assistant with real OS-level "hands":

- Read and write files.
- Run shell, AppleScript, and Shortcuts.
- Watch your workspace and re-index it for RAG.
- Talk to local MCP servers over stdio, Unix socket, or HTTP+SSE.
- Sync models and messages over an encrypted P2P mesh.
- Speak responses with native TTS.
- Enforce a declarative security policy and keep a hash-chained audit ledger.

All inference, tool execution, memory, and audit state stay on the machine.

## Consumer Readiness

**Current score: 9.85 / 10**

Bad Apple is **#1 in the independent AI OS layer tier**. The only other product
in this tier is OpenAGI, and it lacks Bad Apple's hardware-rooted identity,
hash-chained audit ledger, air-gap certification, policy engine, VRAM governor,
fast-tier routing, speculative-decoding telemetry, native document reading,
MCP marketplace, or health supervisor.

It is **below the platform-vendor AI tier** (Apple Intelligence, Copilot+,
Gemini Nano) on distribution and OS integration, and it is **behind frontier
cloud models** (Claude 4, GPT-4o, Gemini 2.5) on raw reasoning and difficult
coding. Its value is the integrated, auditable, air-gapped system architecture —
not the raw model alone.

The remaining blockers to 10/10 are the deliberate lack of Apple notarization,
a clean-machine VM install/smoke test run on a real fresh Mac, and one
remaining `paste` transitive dependency after the recent `sled` → `redb` and
`bincode` → `ciborium` work.

Recent work closed the first-run preflight, memory-adaptive install,
menubar-plist rendering, source-free public release
(`savageAZfck/bad-apple-releases`), and public Homebrew Cask
(`savageAZfck/homebrew-bad-apple`).

The **v0.1.2 corrective beta** fixes deterministic control-phrase handling
(`kill switch`, `resume bad apple`) and natural-language tool invocation
(`run shell ...`, `write a note ...`, `generate an image of ...`). It also
replaces model-name memory cutoffs with measured weight sizes and
configuration-based context/workspace estimates, fixes voice helper error
reporting, and makes the air-gap certification suite pass from the installed
CLI. Curious self-improvement remains wired to the autopilot toggle.

**Hardware:** Apple Silicon Mac (M1 or newer). 8 GB unified memory is the practical
floor and 16 GB is recommended. Runtime requirements vary with the selected
weights, context settings, and other applications; installation does not reserve
memory for Bad Apple.

See [Bad_Apple_Ranking.md](Bad_Apple_Ranking.md) for the full ranking,
benchmarks, and competitive placement.

## Models

Bad Apple ships with a switchable, multi-tier model registry:

| Model | Role | How to use |
|---|---|---|
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | **Default main model** | Loaded by default |
| `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` | **Switchable general model** | `badapple model use <id>` or `BADAPPLE_MAIN_MODEL` |
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | **Optional fast tier** | `BADAPPLE_FAST_TIER=1` |
| any cached draft model | **Optional speculative decoding** | `BADAPPLE_SPECULATIVE_DRAFT=<id>` |

The 7B Coder is now the default because it scored **6/7 (~86%)** on Bad Apple's
live seven-problem coding suite, while the 9B model scored **4/7 (~57%)** on the
same suite. The 7B Qwen2.5 Coder is best-in-class for the 7B tier on public
coding benchmarks.

## Key Capabilities

- **Native Swift MLX inference** on Apple Silicon, with streaming, KV cache,
  VRAM admission, and optional fast-tier routing.
- **Real speculative-decoding telemetry** (`draft_accept_pct` from MLX
  `GenerateCompletionInfo`).
- **SLICKS v1/v2 authenticated IPC** over Unix domain sockets: HMAC-SHA256 and
  Secure Enclave ECDSA P-256.
- **Hardware-rooted identity** with Secure Enclave key storage, signing, key
  pinning, and model-provenance verification.
- **Hash-chained audit ledger** — SHA-256 chained, HMAC'd, secret-redacted logs.
- **Declarative policy engine** — 60+ rules in `policy.yaml` with per-tool
  argument enforcement, path allowlists, and human-in-the-loop approvals.
- **Fail-closed automation cage** — `openat`-based file operations with
  `O_NOFOLLOW`, allowlisted roots, and symlink/hardlink rejection.
- **WASM sandbox** — fuel-metered, store-limited, output-capped execution.
- **Streaming output firewall** — Aho-Corasick pattern matching with real-time
  secret redaction.
- **Native FSEvents workspace watcher** with automatic re-indexing.
- **RAG + semantic cache** — `bge-small-en-v1.5` embeddings, cosine-similarity
  lookup, scoped by persona.
- **MCP marketplace** — local tool-server catalog with stdio, Unix socket, and
  HTTP+SSE transports.
- **Encrypted P2P mesh** — AES-256-GCM model and message sync, symmetric
  push/pull model transfer, off by default for air-gap certification.
- **Native TTS** via `AVSpeechSynthesizer` and `badapple-tts`.
- **CLI agent protocol** — full JSON-RPC control of runtime, models, tools, P2P,
  MCP, vault, and audits.
- **Persona system** — hot-reloadable `prompt.txt`, `personas.json`, and
  voice-specific prompts.
- **Bounded Curious self-improvement autopilot** — wired to the autopilot toggle.
  When autopilot is on, the engine runs a local self-check (cert, doctor, output
  firewall, git status, source TODO/FIXME scan) and writes a proposal note to
  `~/.bad_apple/notes/proposed_patches/`. Trigger manually with `badapple
  "curious check"`.
- **Air-gap certification** — `badapple cert` runs 15 runtime checks; Rust
  integration tests assert zero network sockets.

## Architecture

```text
badapple CLI / menu bar / voice host / dashboard
              │
              ▼
   /var/run/badapple/substrate.sock  (SLICKS v1/v2)
              │
              ▼
     gatekeeper (Rust, launchd)
     ├─ CandleBrain semantic router
     ├─ Fast action resolver
     ├─ Replay cache (nonce dedup)
     ├─ Automation cage (openat, O_NOFOLLOW)
     └─ WASM sandbox
              │
              ▼
   badapple-engine  (Swift MLX)
   ├─ 7B Qwen2.5 Coder (default) + 9B switchable + 0.5B fast tier
   ├─ Optional speculative decoding with live acceptance telemetry
   ├─ Embeddings / semantic cache
   ├─ Audit ledger (CryptoKit checkpoints)
   ├─ Output firewall (Aho-Corasick)
   ├─ Tool router + policy engine
   ├─ FSEvents workspace watcher
   ├─ Curious autopilot self-improvement loop (policy-gated)
   ├─ MCP server (stdio / socket / SSE)
   └─ P2P encrypted mesh
              │
              ▼
    badapple-tts  (native AVSpeechSynthesizer)

    BadAppleAmbient / screen capture (opt-in)
```

## Build

```bash
cargo build --release
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
```

## Test

```bash
# Rust unit tests and integration tests
cargo test --release

# Format and lint
cargo fmt --check
cargo clippy --release -- -D warnings

# Air-gap certification suite
cargo test --release --test cert_suite
badapple cert
```

## Install

### Consumer install (Homebrew Cask — one command)

This is the smoothest path. Homebrew removes the Gatekeeper quarantine flag and
runs the native platform installer for you:

```bash
brew tap savageAZfck/bad-apple https://github.com/savageAZfck/homebrew-bad-apple
brew install --cask bad-apple
```

The 7B model is downloaded on first use. If you prefer to seed the cache
offline, set `MODEL_CACHE_SRC` before installing; see `tests/vm_smoke_test.sh`.

### Developer / manual install (unsigned)

No Apple Developer ID required:

```bash
cargo build --release
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
sudo src/platform/apple_desktop/strip_quarantine.sh
osascript -e 'do shell script "cd /path/to/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install --unsigned-install" with administrator privileges'
```

For release packaging and signing see `package_full_release.sh`,
`package_homebrew_cask.sh`, `package_signed_release.sh`, and
`package_unsigned.sh`.

## Quick Use

```bash
# Text query
target/release/badapple "What is 2+2?"

# Voice (text output)
BADAPPLE_VOICE=1 target/release/badapple "What do you think of Siri?"

# Voice with TTS
target/release/badapple --speak "What do you think of Siri?"

# Benchmark
target/release/badapple --benchmark

# Diagnostics and air-gap cert
target/release/badapple --doctor
target/release/badapple cert

# Curious self-improvement check (manual trigger)
target/release/badapple "curious check"

# Persona switch
target/release/badapple "switch to wicket"
target/release/badapple --roast "Tell me about cloud AI"
```

## Security & Privacy

Bad Apple is designed around a fail-closed, local-first security model. The
runtime is air-gap certifiable: with P2P and MCP disabled, the daemon process
holds zero network sockets.

| Property | Implementation | Tests |
|---|---|---|
| IPC authentication | SLICKS v1 HMAC-SHA256, v2 Secure Enclave P-256 | `bad_apple_ipc.rs` + `tests/cert_suite.rs` |
| Replay protection | Nonce-pair replay cache | `replay_cache_rejects_replayed_slicks_proofs` |
| Filesystem isolation | `openat` + `O_NOFOLLOW` automation cage | `tool_cage_rejects_path_traversal`, `tool_cage_rejects_symlink_escape` |
| Untrusted code | WASM sandbox with fuel, memory, and output caps | `wasm_cage::tests::*` |
| Output safety | Streaming Aho-Corasick firewall | `output_firewall_patterns_present` |
| Audit integrity | SHA-256 chained, secret-redacted ledger | `ledger_hash_chain_is_valid`, `ledger_redacts_secrets` |
| Network posture | Air-gap cert suite proves zero external sockets | `no_external_network_sockets` |
| Policy coverage | Declarative `policy.yaml` covering 60+ tool rules | `policy_yaml_covers_dangerous_tools` |

See [PRIVACY.md](PRIVACY.md) for the data posture.

## Notarization Stance

Bad Apple is intentionally **not notarized**. Notarization requires uploading
binaries to Apple's servers, which conflicts with the product's "nothing leaves
your machine" promise. The recommended install path is the **Homebrew Cask**
(`brew install --cask bad-apple`), which strips quarantine locally without
routing through Apple. Direct-download users can run the included
`strip_quarantine.sh`.

## Documentation

- [Bad_Apple_Ranking.md](Bad_Apple_Ranking.md) — Consumer readiness score,
  live benchmarks, and competitive placement.
- [BAD_APPLE.md](BAD_APPLE.md) — Technical deep dive.
- [BAD_APPLE_BUYERS.md](BAD_APPLE_BUYERS.md) — Buyer-facing overview.
- [AGENTS.md](AGENTS.md) — Build commands, project conventions, and architecture
  notes.
- [CHANGELOG.md](CHANGELOG.md) — Development history.
- [PRIVACY.md](PRIVACY.md) — Privacy and data posture.

## Stats

| Metric | Value |
|---|---|
| Consumer-readiness score | 9.85 / 10 |
| Rust tests | 103 passing |
| Cert suite | 15 checks passing |
| Mesh-sync tests | 4 passing |
| Red-team tests | 6 passing |
| Default model | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` |
| Optional fast tier | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| Switchable general model | `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` |

## License

LicenseRef-Proprietary. See `LICENSE.txt`.
