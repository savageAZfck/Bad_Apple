# Bad Apple Architecture

This document describes the current structure, data flow, and invariants of the Bad Apple runtime. It is intended for acquisition due diligence and for engineers extending the system.

## Design principles

- **Cloudless and bare-metal.** No remote inference APIs, no telemetry exfiltration. Models, embeddings, and reasoning stay on the local machine after the first model weight download.
- **Sovereign state.** Identity, audit logs, semantic cache, learned facts, and model weights survive restarts through local persistence in `~/.bad_apple` and `/var/lib/bad_apple`.
- **Bounded resources.** The Swift `MemoryGovernor` polls macOS memory pressure and unloads optional models. The Rust `wasm_cage` enforces fuel, output, and memory limits. The policy engine enforces per-tool timeouts, output limits, and approval requirements.
- **Signed IPC and mesh.** All CLI traffic is authenticated through SLICKS v1 (HMAC-SHA256) or SLICKS v2 (Secure Enclave). P2P mesh payloads are AES-256-GCM encrypted and HMAC-SHA256 authenticated.
- **Reproducible builds.** Release profile uses `lto`, `codegen-units = 1`, and `panic = "abort"`.

## Component topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  User interfaces                                                              │
│  ├── `badapple` CLI                                                         │
│  ├── `Bad Apple.app` native menu bar                                        │
│  └── `badapple-dashboard` at http://127.0.0.1:8787                          │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Rust gatekeeper + binaries                                                   │
│  ├── `badapple` — CLI, SLICKS client, persona/tier/voice flags              │
│  ├── `gatekeeper` — prompt classification, fast-action resolver, proxy       │
│  ├── `badapple-mcp` — Model Context Protocol server (off by default)        │
│  ├── `badapple-p2p` — encrypted P2P model/message sync (off by default)     │
│  ├── `badapple-supervisor` — health and restart budget monitor               │
│  ├── `badapple-dashboard` — web Control Center (off by default)              │
│  └── `badapple-identity` + `badapple-identity-agent` — SLICKS v2 identity    │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │ Unix socket: /var/run/badapple/substrate_mlx.sock
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Swift MLX daemon (`badapple-engine`)                                         │
│  ├── `BadAppleMLX` — MLX-LM model loading and speculative decoding          │
│  ├── `BadAppleEngine` — prompt handling, tool routing, RAG, cache           │
│  ├── `BadAppleModelManager` — model registry and memory admission           │
│  ├── `BadAppleTools` — policy enforcement and native tool invocation        │
│  ├── `BadAppleNativeRuntime` — UMA reservations and health checks           │
│  ├── `BadAppleSecurity` — SLICKS v2 Secure Enclave identity client          │
│  ├── `BadAppleConversation` — chat history and streaming                    │
│  ├── `BadAppleWorkspaceWatcher` — FSEvents-based workspace indexing         │
│  └── `BadAppleMenuBar` — native menu bar and TTS playback                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Rust source modules

Core runtime (`src/`):

- `lib.rs` — C FFI bridge, public API, and the 103 Rust unit tests.
- `bad_apple_ipc.rs` — SLICKS frame protocol, replay cache, hash-chained audit ledger.
- `cert.rs` — Air-gap certification and runtime checks.
- `config.rs` — Centralized `BADAPPLE_*` environment configuration.
- `automation_cage.rs` / `automation_cage_impl.rs` — Declarative filesystem tool policy and `openat`/`O_NOFOLLOW` sandbox.
- `wasm_cage.rs` — Fuel-metered WebAssembly sandbox for untrusted tool synthesis.
- `mcp.rs` / `mcp_marketplace.rs` — Model Context Protocol server and local marketplace.
- `mesh_sync.rs` / `p2p_crypto.rs` / `p2p_model.rs` / `protocol.rs` — Encrypted P2P mesh, model transfer, and signed engram fabric.
- `metal_uma.rs` — Metal UMA buffer and safetensors utilities.
- `tensor_brain.rs` — BPE tokenization and embedding utilities.
- `strategy_library.rs` — Sled-backed durable strategy cache.
- `vault.rs` — Secure key-value store.
- `workspace_watcher.rs` — FSEvents workspace watcher.
- `red_team/` — Adversarial self-test probes for SLICKS, cage, P2P, policy, and WASM.
- `metrics.rs` / `benchmark.rs` / `ane_core.rs` / `scavenger.rs` / `arena.rs` — telemetry, benchmarks, ANE/CoreML artifacts, file scavenger, and allocator utilities.

Binaries (`src/bin/`):

- `badapple.rs` — main CLI.
- `gatekeeper.rs` — gatekeeper daemon with prompt classification and tool routing.
- `badapple-mcp.rs` — MCP server.
- `badapple-p2p.rs` — P2P daemon.
- `badapple-supervisor.rs` — health supervisor.
- `badapple-dashboard.rs` — web dashboard.
- `badapple-fetch.rs` / `badapple_fetch_metallib.rs` — artifact and Metal cache fetch helpers.
- `train_gatekeeper.rs` — gatekeeper classifier training.

## Data flow

1. The user sends a prompt through the `badapple` CLI or the menu bar.
2. The Rust CLI SLICKS-authenticates to the `gatekeeper` and forwards to `badapple-engine` over the Unix socket.
3. The Swift daemon loads the active model, merges `prompt.txt`, persona, and workspace context, then streams a response.
4. If the model emits a `<tool_call>`, `BadAppleTools` enforces `policy.yaml` limits and asks for approval for destructive tools unless autopilot is enabled.
5. Filesystem and shell tools run through the `automation_cage` (`openat`/`O_NOFOLLOW`).
6. Untrusted tool synthesis runs in the `wasm_cage` with bounded fuel, memory, and output.
7. Every query, tool call, cache hit, and response is appended to the hash-chained audit ledger (`/var/lib/bad_apple/ledger.jsonl`).
8. Optional P2P sync can sync personas, prompts, and model manifests over the encrypted mesh when enabled.

## Verified quality gates

```bash
cargo fmt --check
cargo clippy --release --tests
cargo test --release
```

The Swift menu-bar build is verified with:

```bash
BADAPPLE_NO_SIGN=1 src/platform/apple_desktop/build_bad_apple_menu_bar.sh
```
