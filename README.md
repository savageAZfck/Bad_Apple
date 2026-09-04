# Bad Apple

> **A local-first AI operating system layer for macOS. On-device inference, hardware-rooted identity, fail-closed security, and a native Swift menu bar.**

Bad Apple is not a chat app. It is a system-level AI runtime for Apple Silicon that keeps prompts, memory, and tools on the machine. It runs as three launchd daemons, authenticates every interaction through a custom IPC protocol, and includes an air-gap certification suite that proves the runtime opens zero non-loopback network sockets when P2P and MCP are disabled.

## What It Does

- **On-device inference with native Swift MLX** — 9B Qwen 3.5 4-bit, optional 0.5B fast tier, and optional speculative decoding via `mlx-lm`.
- **SLICKS authenticated IPC** — HMAC-SHA256 (v1) and Secure Enclave ECDSA P-256 (v2) challenge-response over Unix domain sockets.
- **Hardware-rooted identity** — Secure Enclave key storage for signing, key pinning, and model provenance.
- **Hash-chained audit ledger** — SHA-256 chained, secret-redacted logs of every query, tool call, and response.
- **Fail-closed automation cage** — `openat`-based file operations with `O_NOFOLLOW`, path allowlisting, and symlink/hardlink rejection.
- **WASM sandbox** — fuel-metered, store-limited, output-capped execution for untrusted code synthesis.
- **Streaming output firewall** — Aho-Corasick pattern matching with real-time secret redaction.
- **Semantic cache** — `bge-small-en-v1.5` embeddings with cosine-similarity lookup, scoped by persona.
- **RAG context** — workspace file watching, ambient context, and optional ocular screen-stream summarization.
- **Tool router + policy engine** — declarative policy in `policy.yaml` with tool-specific rules, path allowlists, denied patterns, and human-in-the-loop approvals for destructive tools.
- **P2P encrypted mesh** — AES-256-GCM link-local peer sync for models and messages, off by default for air-gap certification.
- **MCP marketplace** — local tool-server catalog with lifecycle management.
- **Persona system** — hot-reloadable `prompt.txt`, runtime persona packs, and voice-specific prompts.
- **Voice mode** — on-device speech recognition, wake phrase, and native `AVSpeechSynthesizer` TTS.
- **Native menu bar app** — onboarding wizard, streaming chat window, dashboard, model selector, settings, and image drag-and-drop.

## Architecture

```
badapple CLI / menu bar / voice host
              │
              ▼
   /var/run/badapple/substrate.sock  (SLICKS v1/v2)
              │
              ▼
     gatekeeper (Rust, launchd)
     ├─ CandleBrain semantic router
     ├─ Fast action resolver
     ├─ Replay cache (nonce dedup)
     └─ Automation cage (openat, O_NOFOLLOW)
              │
              ▼
   badapple-engine  (Swift MLX)
   ├─ 9B Qwen 3.5 + 0.5B fast tier
   ├─ Optional speculative decoding
   ├─ Embeddings / semantic cache
   ├─ Audit ledger (CryptoKit checkpoints)
   ├─ Output firewall (Aho-Corasick)
   ├─ Tool router + approval policy
   └─ MCP server
              │
              ▼
    badapple-tts  (native AVSpeechSynthesizer)

    BadAppleAmbient / screen capture (opt-in)
```

## Security

Bad Apple is designed around a fail-closed, local-first security model. The current verification surface includes:

| Property | Implementation | Tests |
|---|---|---|
| IPC authentication | SLICKS v1 HMAC-SHA256, v2 Secure Enclave P-256 | `bad_apple_ipc.rs` + `tests/cert_suite.rs` |
| Replay protection | Nonce-pair replay cache | `replay_cache_rejects_replayed_slicks_proofs` |
| Filesystem isolation | `openat` + `O_NOFOLLOW` automation cage | `tool_cage_rejects_path_traversal`, `tool_cage_rejects_symlink_escape` |
| Untrusted code | WASM sandbox with fuel, memory, and output caps | `wasm_cage::tests::*` |
| Output safety | Streaming Aho-Corasick firewall | `output_firewall_patterns_present` |
| Audit integrity | SHA-256 chained, secret-redacted ledger | `ledger_hash_chain_is_valid`, `ledger_redacts_secrets` |
| Network posture | Air-gap certification suite proves zero external sockets; P2P/MCP are off by default | `no_external_network_sockets`, `p2p_and_mcp_off_by_default` |
| Policy coverage | Declarative policy in `policy.yaml` covering destructive tools, path traversal, and shell allowlists | `policy_yaml_present`, `policy_yaml_covers_dangerous_tools` |

See `AGENTS.md` for build and verification conventions, and `Bad_Apple_Ranking.md` for the current consumer-readiness score and remaining blockers.

## Build

```bash
cargo build --release
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
```

## Test

```bash
# Rust unit tests and integration tests
cargo test --release

# Cert suite (network isolation, tool cage, SLICKS replay, ledger, policy)
cargo test --release --test cert_suite
```

## Install

```bash
# Unsigned build (no Apple Developer ID required)
cargo build --release
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
sudo src/platform/apple_desktop/strip_quarantine.sh
osascript -e 'do shell script "cd /path/to/bad_apple && src/platform/apple_bridge/install_badapple_platform.sh --install --unsigned-install" with administrator privileges'
```

For release packaging and signing see `package_signed_release.sh` and `package_unsigned.sh`.

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

# Persona switch
target/release/badapple "switch to wicket"
target/release/badapple --roast "Tell me about cloud AI"
```

## Stats

| Metric | Value |
|---|---|
| Lines of code | ~56,000 (Rust ~20,000, Swift ~25,300, Shell ~2,500, plus web/docs) |
| Rust tests | 103 passing |
| Cert suite | 15 integration tests passing |
| Model | 9B Qwen 3.5 4-bit, optional 0.5B fast tier |
| Native inference | `mlx-swift-lm` with streaming, optional speculative decoding, KV cache |
| Security | SLICKS v1+v2, openat cage, WASM sandbox, audit ledger, air-gap certification suite |

## Documentation

- [Bad_Apple_Ranking.md](Bad_Apple_Ranking.md) — Consumer readiness score, what is done, what remains.
- [AGENTS.md](AGENTS.md) — Build commands, project conventions, architecture notes.
- [CHANGELOG.md](CHANGELOG.md) — Development history.
- [PRIVACY.md](PRIVACY.md) — Privacy and data posture.

## Current Status

Bad Apple is **working, tested, and running 24/7 on the author's Mac**. It is intended for technical early adopters and researchers. The remaining blockers to a general consumer release are:

- Apple notarization for the release artifact.
- A clean-machine VM install / smoke test.
- Final visual polish of the first-launch onboarding.

## License

LicenseRef-Proprietary. See `LICENSE.txt`.
