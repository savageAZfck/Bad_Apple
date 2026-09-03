# Bad Apple

> **A baremetal AI OS layer for Apple Silicon — runs directly on the Neural Engine, Secure Enclave, and Metal GPU. Provably air-gapped. Audited. Fuzzed. Now with native Swift MLX inference.**

Bad Apple is not an app. It's not a model wrapper. It's a system-level AI runtime that manages hardware, security, and IPC for on-device AI workloads on macOS. It runs as three launchd daemons with root privileges, authenticates every interaction with a custom protocol backed by the Secure Enclave, and can prove — with a 12-test certification suite — that zero network listeners are active.

## What It Does

- **Runs a 9B Qwen 3.5 model via native Swift MLX inference** — the menu bar app loads the model directly via mlx-swift-lm, with no subprocess or daemon overhead.
- **Provable air-gap privacy** — a toggle that turns off all network access, verified by a 12-test certification suite. Not a trust claim. A proof.
- **Hash-chained audit ledger** with SHA-256 chaining — every query, tool call, and response is logged and tamper-evident (now in Swift via CryptoKit)
- **Fail-closed filesystem cage** using openat-based fd operations with O_NOFOLLOW — structurally eliminates TOCTOU race conditions
- **WASM sandbox** with fuel metering, StoreLimits, and output caps for untrusted tool synthesis
- **Streaming output firewall** with real-time secret redaction (now in Swift)
- **Semantic cache** with cosine similarity lookup — repeated questions return instantly (now in Swift)
- **RAG context builder** — retrieves from memory graph and workspace documents (now in Swift)
- **Tool router + policy engine** — 6 tools with path jailing and approval gates (now in Swift)
- **Persona system** with hot-reload, custom banter, and roast bank (now in Swift)
- **Fast tier** — simple queries get fewer tokens for faster response
- **10,000-dimensional hyperdimensional computing substrate** (vector symbolic architecture) for script profiling
- **Memory-mapped connectome** with 2048-D embeddings, zero-copy persistence
- **P2P encrypted mesh sync** (AES-256-GCM, link-local, off by default)
- **Voice mode** with Piper TTS, on-device speech recognition, wake word detection, and streaming TTS during generation
- **Swift menu bar app** with onboarding wizard, chat window (markdown rendering, message bubbles, code blocks), settings UI, model selector, and image drag-and-drop

## Architecture

```
badapple CLI / menu bar / voice host
              │
              ▼
   /var/run/badapple/substrate.sock  (SLICKS)
              │
              ▼
     gatekeeper (Rust, launchd, root)
     ├─ Semantic router (576-D CandleBrain)
     ├─ Fast action resolver
     ├─ Replay cache (nonce dedup)
     └─ Automation cage (openat, O_NOFOLLOW)
              │
              ▼
   badapple-engine  (Swift MLX)
   ├─ 9B Qwen 3.5 + 0.5B fast tier
   ├─ Speculative decoding (DFlash + MTP)
   ├─ RAG / embeddings / semantic cache
   ├─ Audit ledger (SE-signed checkpoints)
   ├─ Output firewall (Aho-Corasick)
   ├─ Tool router + approval policy
   └─ MCP server
              │
              ▼
    badapple-tts  (Piper TTS)
    
    ANE Bridge (Swift FFI)
    ├─ Multi-shard CoreML execution
    ├─ KV cache scatter
    ├─ Placement measurement
    └─ QoS elevation
```

## Security

Bad Apple has been through **two full security audit passes**. 69 vulnerabilities were found and fixed across the Rust, Swift, Shell, and Metal code layers. Four fuzzing targets were built with cargo-fuzz and run for 2 hours each — **127 million iterations, zero crashes**.

| Attack Surface | Fuzzer | Iterations | Crashes |
|---|---|---|---|
| SLICKS IPC frame parsing | fuzz_ipc_frame | 423K | 0 |
| WASM cage compilation + execution | fuzz_wasm_cage | 638K | 0 |
| P2P protocol frame parsing | fuzz_protocol_frame | 694K | 0 |
| Scavenger path handling | fuzz_scavenger_path | 760K | 0 |

**Security architecture:**
- SLICKS v1 (HMAC-SHA256) and v2 (Secure Enclave ECDSA P-256) IPC authentication
- Server key pinning (TOFU trust store)
- Replay cache with nonce deduplication
- Fail-closed filesystem cage with openat operations
- WASM sandbox with StoreLimits, fuel metering, output caps
- Hash-chained audit ledger with SE-signed checkpoints
- Air-gap certification suite (12 tests, zero network listeners)
- Streaming output firewall (Aho-Corasick secret redaction)
- Human-in-the-loop approval policy engine (37 rules)
- P2P encrypted mesh (AES-256-GCM, off by default)

See [THREAT_MODEL.md](THREAT_MODEL.md) for the full threat model and [docs/SLICKS_PROTOCOL.md](docs/SLICKS_PROTOCOL.md) for the IPC protocol specification.

## Build

```bash
cargo build --release
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
```

## Test

```bash
# Rust tests (78 tests)
cargo test --release

# Clean install test (32 checks)
tests/test_clean_install.sh

# Fuzzing (requires nightly)
cargo +nightly fuzz run fuzz_ipc_frame -- -max_total_time=300
```

## Install

```bash
# Unsigned build (no Apple Developer ID required)
cargo build --release
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
sudo src/platform/apple_desktop/strip_quarantine.sh
osascript -e 'do shell script "cd /path/to/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install --unsigned-install" with administrator privileges'
```

## Use

```bash
# Text query
target/release/badapple "What is 2+2?"

# Voice (text output)
BADAPPLE_VOICE=1 target/release/badapple "What do you think of Siri?"

# Voice with TTS
target/release/badapple --speak "What do you think of Siri?"

# Benchmark
target/release/badapple --benchmark

# Diagnostics
target/release/badapple --doctor

# Crash report
target/release/badapple --crash-report

# Persona switch
target/release/badapple "switch to wicket"
target/release/badapple --roast "Tell me about cloud AI"
```

## Stats

| Metric | Value |
|---|---|
| Lines of code | ~72,000 (Rust 18,864, Swift 15,480, Shell 2,315) |
| Swift logic modules | BadAppleEngine, BadAppleMLX, BadAppleSecurity, BadAppleTools, BadAppleConversation, BadAppleRAG |
| Build time | 35 days + Swift migration |
| Security audits | 2 passes, 69 bugs found and fixed |
| Fuzzer iterations | 127 million, zero crashes |
| Tests | 78 Rust + 32 install + 4 fuzz targets |
| Model | 9B Qwen 3.5 4-bit (pinned to commit hash) |
| Native inference | mlx-swift-lm (streaming, speculative decoding, KV cache) |
| Security | SLICKS v1+v2, openat cage, WASM sandbox, audit ledger, air-gap cert |

## Documentation

- [CHANGELOG.md](CHANGELOG.md) — Full changelog with all 69 security fixes
- [THREAT_MODEL.md](THREAT_MODEL.md) — Trust boundaries, threat agents, security properties
- [docs/SLICKS_PROTOCOL.md](docs/SLICKS_PROTOCOL.md) — IPC protocol specification
- [AGENTS.md](AGENTS.md) — Project conventions, build commands, architecture notes
- [fuzz/README.md](fuzz/README.md) — Fuzzing guide and target descriptions

## License

LicenseRef-Proprietary. See `LICENSE.txt`.

## Notarization Stance

Bad Apple will not be submitted to Apple's notarization pipeline. Notarization requires uploading binaries to Apple's servers — a violation of the product's "nothing leaves your machine" promise. This is a deliberate philosophical choice. The Homebrew Cask is the recommended install path.
