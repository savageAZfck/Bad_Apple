# Bad Apple

**An air-gapped, hardware-fused cognitive substrate for Apple Silicon.**

Bad Apple runs a 4-billion-parameter Qwen3-class language model locally and in-process on macOS. Its production daemon keeps a 36-layer stateful FP16 CoreML pipeline, four INT8 language-model heads, and a memory-mapped FP16 embedding table warm for low-latency requests. No cloud model, remote inference service, or external model daemon is required.

Text and voice requests share one path. The `badapple` terminal client and the native `BadAppleIntent` AppIntent authenticate with the same SLICKS challenge-response protocol, connect through a local Unix-domain socket, and stream generation from the one ANE engine owned by `badappled`.

> Bad Apple is an experimental edge-AI runtime, not a claim of AGI or sentience.

## Recent fixes

- **ANE prediction / context-window bug fixed.** The generation budget in `ane_core.rs` now preserves the prompt tokens and caps output to the positions that actually fit in the compiled sequence length. This eliminates the `ANE prediction failed` errors and the nonsensical Java/Spring output that appeared for short prompts.
- **Arbitrary prompts now work.** With the budget fix, the daemon correctly answers open-ended questions ("What is the capital of France?") and emits action blocks ("open my bad_apple workspace").
- **Menu-bar fallback improved.** `BadAppleMenuBar` now parses both fenced ` ```badapple-action` blocks and bare JSON action objects, and the bundled helper was synchronized with the fixed daemon.
- **Daemon entitlements corrected.** `badappled` is now signed with entitlements that let it load the existing `libBadAppleBridge.dylib` under Hardened Runtime on macOS.

## What is Bad Apple?

Bad Apple is an air-gapped, Apple-Silicon-native cognitive runtime. It runs a 4-billion-parameter Qwen3-class language model entirely on-device, without cloud inference, telemetry, or external model services. At its core is a `launchd` daemon, `badappled`, that owns a 36-layer stateful FP16 CoreML pipeline and four LM-head shards over a unified-memory ANE backend. Requests arrive through either the `badapple` terminal client or a native Siri `BadAppleIntent`; both authenticate to the daemon through the SLICKS HMAC-SHA256 protocol before any token is generated.

The repository also contains the broader experimental Bad Apple research runtime: a 576-D Candle transformer brain, a multi-agent signed engram fabric, a self-improving strategy library, a WASM sandbox for untrusted tools, and a telemetry dashboard for local inspection.

## Key features

- **On-device 4B Qwen3 language model.** A Qwen3-4B model compiled into 36 FP16 CoreML layer shards, four INT8 LM-head shards, and a ~742 MB mmap'd FP16 embedding table.
- **Air-gapped by default.** Daemon mode uses only a Unix-domain socket; the daemon integration test asserts zero Internet sockets.
- **Authenticated local ingress.** `badapple` CLI and `BadAppleIntent` Siri shortcut use the same SLICKS handshake with HMAC-SHA256 mutual proof, nonces, and prompt binding.
- **Streaming local generation.** Token deltas stream to the client as they are decoded from the warm ANE pipeline.
- **Stateful CoreML backend.** `MLState` KV caches persist across calls; the daemon stays warm for low-latency inference.
- **Multi-modal cognitive runtime.** 576-D transformer, hyperdimensional memory, connectome, conscience oracle, strategy library, and wild-workspace watcher (non-daemon research mode).
- **Signed multi-agent fabric.** TCP/UDP/WebSocket engrams with HMAC signatures and a 2048-D cosine-similarity firewall.
- **WASM + Python sandboxes.** Untrusted Rust/WASM and Python tools run in isolated sandboxes.
- **Durable sovereign state.** Memory, identity, strategies, and weights survive restarts through background incremental saves.
- **Reproducible, optimized build.** `lto`, single codegen unit, `panic = "abort"`, and a deterministic `Cargo.lock`.

## Architecture

```text
badapple CLI ─────────────┐
                          │  Unix-domain IPC + SLICKS HMAC-SHA256
Siri / BadAppleIntent ────┤  mutual proof, nonce, timestamp, prompt binding
                          ▼
                  badappled launch daemon
                          │
                          ▼
                  in-process Rust/Swift FFI
                          │
            ┌─────────────┴─────────────┐
            │ 36 stateful FP16 layers  │
            │ mmap FP16 embeddings     │
            │ 4 sharded INT8 LM heads  │
            │ CoreML MLState KV cache  │
            └─────────────┬─────────────┘
                          ▼
                  streamed local text
```

The CLI does not attempt to attach to a process ID or call FFI across a process boundary. Process IDs change after every restart, and FFI is in-process only. Instead, both ingress clients connect to the stable socket at `/var/run/badapple/substrate.sock`; the daemon alone owns and invokes the ANE FFI handle.

## Bad Apple model substrate

- **36 stateful CoreML layer shards.** Each transformer layer is independently compiled as FP16 and retains KV cache through `MLState`.
- **Four INT8 LM-head shards.** The heads jointly cover the model's 151,936-token vocabulary.
- **Memory-mapped embeddings.** The approximately 742 MB FP16 embedding table is mapped rather than copied into a second host buffer.
- **Unified-memory handoff.** Hidden states move between CoreML shards as `MLMultiArray` values over Apple unified memory.
- **Compute-unit selection.** Startup measures candidate CoreML configurations and rejects compiler failures or pathological prewarm latency.
- **Warm daemon lifetime.** The ANE model handle, shard graph, and mapped artifacts remain owned by the long-running daemon.
- **Observable inference.** Placement ratio, prewarm time, first-token latency, decode latency, throughput, call count, and failure count are exposed in local logs and tests.

A reference run with the current Qwen3-4B FP16 layer shards measured 41.09% CoreML ANE operation placement, approximately 174 ms/token decode latency, and approximately 5.7 tokens/second. These are measurements from one machine and model conversion, not guaranteed performance figures.

## Authenticated local ingress

### SLICKS handshake

Every request must complete a fail-closed local handshake before inference begins:

1. The client sends a protocol version, timestamp, and 256-bit client nonce.
2. The daemon returns a fresh 256-bit server nonce and an HMAC-SHA256 server proof.
3. The client verifies the daemon and sends a client proof bound to both nonces, the timestamp, token limit, and SHA-256 hash of the prompt.
4. The daemon rejects stale timestamps, malformed frames, invalid proofs, empty prompts, prompts over 64 KiB, and generation limits outside `1..=4096`.
5. After authentication, the daemon emits `accepted`, `token`, and `done` frames.

The production key is generated during installation at `/var/lib/bad_apple/slicks.key` with `root:staff` ownership and mode `0640`. The socket is created inside a setgid `root:staff` directory. No secret is compiled into either binary.

### Terminal client

```bash
badapple "Explain quantum error correction"
badapple --max-tokens 128 "Summarize the thermodynamic arrow of time"
```

`badapple` writes decoded token deltas to stdout as the warm daemon generates them. Configuration overrides:

```text
BADAPPLE_SOCKET_PATH
BADAPPLE_SLICKS_KEY_PATH
BADAPPLE_SLICKS_SECRET   # intended for tests and controlled development only
```

### Siri and AppIntents

`BadAppleIntent` accepts a spoken-text array, joins it into one prompt, completes the same SLICKS handshake, and routes the request to the same daemon generation queue. `BadAppleShortcuts` publishes the shortcut title **Execute Bad Apple** and the Siri phrases **Execute Bad Apple** and **Ask Bad Apple** through the application-name token.

Build the native bridge and AppIntent host:

```bash
src/platform/apple_bridge/build_apple_bridge.sh
src/platform/apple_desktop/build_bad_apple_menu_bar.sh
```

The second command creates `target/release/Bad Apple.app`. Install it in `/Applications` and launch it once in a logged-in user session so macOS can index its AppIntents metadata:

```bash
sudo ditto "target/release/Bad Apple.app" "/Applications/Bad Apple.app"
open "/Applications/Bad Apple.app"
```

Siri cannot be registered by a pre-login LaunchDaemon alone: AppIntents must be hosted by a signed application in a user session. Inference still occurs in the system daemon; the application is only the Siri ingress host. Siri returns the completed response as a dialog, while the underlying daemon protocol remains token-streamed.

## Build

Requirements:

- Apple Silicon Mac running macOS 26 or later
- Rust toolchain with Cargo
- macOS 26 SDK containing CoreML, FoundationModels, AppIntents, CryptoKit, and Security
- Converted model artifacts under `tests/ane_brain_perf/artifacts/qwen3b_ane_shards/`, or equivalent paths supplied through the environment

```bash
cargo build --release
src/platform/apple_bridge/build_apple_bridge.sh
```

Release outputs:

```text
target/release/badappled
target/release/badapple
target/release/libbad_apple.dylib
target/release/libBadAppleBridge.dylib
target/release/bad_apple_core.h
```

## Install the launch daemon

The installation script validates the release binaries and model artifacts, installs immutable executable copies under `/usr/local`, creates the SLICKS key and restricted runtime directories, installs the launchd plist, and bootstraps the system job.

```bash
cargo build --release
src/platform/apple_bridge/build_apple_bridge.sh
sudo src/platform/apple_bridge/install_daemon.sh
```

Default artifacts can be overridden while installing:

```bash
sudo BADAPPLE_ANE_MODEL="/absolute/path/conversion_manifest.json" \
     BADAPPLE_ANE_TOKENIZER="/absolute/path/tokenizer.json" \
     src/platform/apple_bridge/install_daemon.sh
```

Inspect the service and logs:

```bash
sudo launchctl print system/com.badapple.substrate
tail -f /var/log/bad_apple_daemon.log
badapple "Report substrate status"
```

Installed paths:

```text
/usr/local/libexec/badapple/badappled
/usr/local/libexec/badapple/libBadAppleBridge.dylib
/usr/local/bin/badapple
/Library/LaunchDaemons/com.badapple.substrate.plist
/var/lib/bad_apple/
/var/run/badapple/substrate.sock
/var/log/bad_apple_daemon.log
```

The plist uses `RunAtLoad=true` and `KeepAlive=true`. `launchd` restarts crashes and unexpected exits, but an administrator can intentionally stop the service with `launchctl bootout`; no correctly administered macOS process is literally unkillable.

## Air-gap boundary

`badappled --daemon` returns into the dedicated SLICKS Unix-socket loop before the research runtime initializes its HTTP, TCP, UDP, or WebSocket services. The daemon integration test verifies `lsof -i -a -p <pid>` reports zero network sockets after model initialization and after an authenticated CLI generation.

A Unix-domain socket is still local IPC. It is not an Internet socket and is not included by `lsof -i`. The daemon does not expose a TCP listener, contact a cloud endpoint, or start the repository's optional swarm/telemetry stack.

Running `badappled` without `--daemon` starts the broader experimental cognitive runtime, which includes local HTTP telemetry and optional peer transports. That mode is intentionally separate and must not be described as having zero sockets.

## One-shot benchmark mode

```bash
BADAPPLE_ANE_MODEL="/absolute/path/conversion_manifest.json" \
BADAPPLE_ANE_TOKENIZER="/absolute/path/tokenizer.json" \
BADAPPLE_ANE_BOOT_PROMPT="What is quantum computing?" \
BADAPPLE_ANE_BOOT_TOKENS=20 \
BADAPPLE_ANE_BOOT_ONESHOT=1 \
./target/release/badappled --oneshot
```

The process prints the Bad Apple banner, ANE placement, prewarm time, decode latency, throughput, output text, and failure counters, then exits before any network service starts.

## Model conversion

The resumable conversion pipeline is documented in `tests/ane_brain_perf/README.md`:

```bash
python3 tests/ane_brain_perf/convert_ane_coreml.py --help
```

The generated `conversion_manifest.json` describes the Qwen3-4B embedding table, 36 stateful FP16 layer shards, INT8 LM-head ranges, context length, RoPE settings, and model metadata consumed by the Swift bridge.

## Main components

| Path | Responsibility |
|---|---|
| `src/main.rs` | Daemon lifecycle, authenticated IPC server, one-shot telemetry, and research runtime. |
| `src/bin/badapple.rs` | Lightweight streaming terminal client. |
| `src/bad_apple_ipc.rs` | Shared SLICKS frames, proofs, key loading, limits, and blocking client. |
| `src/ane_core.rs` | Tokenization, streaming generation, ANE bridge loading, counters, and governor policy. |
| `src/platform/apple_bridge/BadAppleBridge.swift` | CoreML monolithic/sharded backends and C ABI exports. |
| `src/platform/apple_bridge/BadAppleIntent.swift` | Native SLICKS client, `BadAppleIntent`, and `BadAppleShortcuts`. |
| `src/platform/apple_desktop/BadAppleMenuBar.swift` | Logged-in AppIntent host and local menu-bar application. |
| `src/platform/apple_bridge/com.badapple.substrate.plist` | System LaunchDaemon definition. |
| `src/platform/apple_bridge/install_daemon.sh` | Root installation and launchd bootstrap. |
| `src/metal_uma.rs` | Shared/private Metal residency, zero-copy buffers, and safetensors persistence. |
| `src/lib.rs` + `build.rs` | `libbad_apple.dylib`, renamed C ABI, and `bad_apple_core.h`. |

## Verification

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --release
src/platform/apple_bridge/build_apple_bridge.sh
src/platform/apple_desktop/build_bad_apple_menu_bar.sh
bash -n src/platform/apple_bridge/install_daemon.sh
plutil -lint src/platform/apple_bridge/com.badapple.substrate.plist
```

The daemon integration test starts `badappled` with an isolated socket and SLICKS secret, invokes the actual `badapple` binary, confirms generation succeeds, and asserts the daemon has zero Internet sockets.

## Security and privacy

- Inference prompts and model outputs remain local.
- SLICKS uses HMAC-SHA256 mutual authentication and constant-time proof verification.
- Request proofs are bound to nonces, timestamp, token budget, and prompt digest.
- The daemon refuses to replace a non-socket filesystem object at its IPC path.
- Model and tokenizer loading fail closed in daemon mode.
- Secrets are never written to logs or committed to the repository.
- See `SECURITY.md` and `PRIVACY.md` for the wider research-runtime threat model.

## License

See `LICENSE.txt`. Proprietary and confidential. No public distribution or commercial use without written permission from Adam Clark.
