# Bad Apple Architecture

This document describes the structure, data flow, and invariants of the Bad Apple runtime. It is intended for acquisition due diligence and for engineers extending the system.

## Design principles

- **Cloudless and bare-metal.** No network calls, no remote APIs, no telemetry exfiltration. All training, inference, and reasoning happen on the local machine.
- **Sovereign state.** Memory, identity, learned skills, and transformer weights survive restarts through durable persistence.
- **Bounded resources.** The `LockFreeRing` drops lowest-priority engrams above 85% occupancy; the `DualProcessGovernor` slows the tick rate under thermal or loss stress.
- **Signed swarm fabric.** Every engram sent over TCP, UDP, or WebSocket is HMAC-signed and verified against a 2048-D cosine-similarity firewall.
- **Reproducible builds.** Release profile uses `lto`, `codegen-units = 1`, and `panic = "abort"`.

## Component topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Interfaces                                                                 │
│  ├── HTTP server at http://127.0.0.1:8080                                   │
│  ├── C FFI bridge (libbad_apple.dylib / bad_apple_core.h)               │
│  ├── Swarm fabric (TCP / UDP / WebSocket)                                   │
│  └── Filesystem watchers (wild_workspace/, curriculum/)                     │
└───────────────────────┬─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Bad Apple runtime                                                      │
│  ├── main.rs — cognitive loop, planning, identity, multi-agent wiring       │
│  ├── tensor_brain.rs — 576-D Candle transformer, BPE tokenizer, three heads │
│  ├── conscience_oracle.rs — LLM oracle + semantic cosine fallback           │
│  ├── apple_intelligence_client.rs — native Apple Intelligence client        │
│  ├── connectome_mmap.rs — zero-copy memory-mapped connectome                │
│  ├── production_blueprint.rs — emotional homeostasis + world model          │
│  ├── governor.rs — DualProcessGovernor, orthogonality regularization        │
│  ├── strategy_library.rs — Sled-backed durable strategy cache               │
│  ├── wild_workspace.rs — async watcher + distributed work queue             │
│  ├── protocol.rs — signed multi-transport engram fabric                     │
│  ├── hyperdimensional_core.rs — 10,000-D HDC vectors and script encoding    │
│  ├── telemetry.rs — HTTP dashboard, skill runner, tool sandbox              │
│  ├── metrics.rs — metrics logger, SVG dashboard, MemoryProfiler             │
│  ├── benchmark.rs — transfer evaluator and PILOT report writer              │
│  ├── data_feed.rs — sensor aggregation (CPU, RAM, battery, photons, etc.)   │
│  ├── config.rs — centralized BADAPPLE_* environment configuration            │
│  └── lib.rs + build.rs — C FFI + generated bad_apple_core.h                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module responsibilities

### `tensor_brain.rs`
- 576-dim, 4-block, 12-head Candle transformer encoder.
- `conscience`, `goal`, and `language` heads.
- AdamW training with dynamic orthogonality regularization.
- BPE tokenization and 2048-D grounded embeddings fused with sensor anchors.

### `conscience_oracle.rs`
- LLM oracle routing.
- Semantic cosine fallback when the oracle is ambiguous or unavailable.

### `apple_intelligence_client.rs`
- Native Apple Intelligence client with JSON repair and auto-fallback.
- No external LLM server required.

### `connectome_mmap.rs`
- Zero-copy `memmap2` persistence for `MemoryGraphNode` records.
- Offloaded saves; loads backfill missing embeddings from text.

### `production_blueprint.rs`
- Emotional homeostasis, causal world model, associative memory graph.
- 10,000-entry identity journal.

### `governor.rs`
- `DualProcessGovernor` scales orthogonality regularization and tick rate from loss, entropy, and thermal state.

### `strategy_library.rs`
- Sled-backed durable cache for proven tool blueprints.
- Tracks reliability and prunes weak strategies automatically.

### `wild_workspace.rs`
- Async directory watcher for `wild_workspace/`.
- Queues pending tool tasks and gossips signed `CompactEngramPacket` Task Engrams when the local queue is heavy.

### `protocol.rs`
- Signed multi-transport engram fabric over TCP, UDP, and WebSocket.
- `ConnectionManager` with exponential-backoff, transport auto-detection, atomic backpressure, and 2048-D cosine firewall.

### `telemetry.rs`
- Axum HTTP server with `/telemetry`, `/metrics`, `/dashboard`, `/tools/run`, `/skills/learn`, `/skills/run`, `/pursuits/add`, `/transfer/evaluate`, and `/identity` endpoints.
- Sandboxed Python tool runner.

### `metrics.rs`
- `MemoryProfiler`, `LatencyRingBuffer`, SVG dashboards, and `metrics.jsonl` logging.

### `benchmark.rs`
- 12-task autonomous curriculum for one-shot transfer evaluation.
- Writes `PILOT_EVALUATION_METRICS.md` with latency, RSS, and token telemetry.

## Data flow

1. Sensors and `wild_workspace` produce raw inputs.
2. `tensor_brain` tokenizes and embeds the input.
3. The embedding is routed to the connectome, strategy library, and governor.
4. `conscience_oracle` or Apple Intelligence proposes actions.
5. Actions are validated, executed in the tool sandbox, and outcomes are written back to the strategy library.
6. State is persisted incrementally on background threads.

## Verified quality gates

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
cargo deny check
```
