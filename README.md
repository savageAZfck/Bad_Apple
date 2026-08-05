# Firefly EdgeOS

**Sovereign, local-first AGI research runtime.**

Firefly EdgeOS is a self-contained, self-training cognitive OS written in Rust. It runs an async runtime with a native Candle Transformer, a local LLM oracle, an associative memory graph, a Sled-backed strategy library, a live telemetry server, a multi-transport swarm fabric, a zero-copy connectome persistence layer, and a C FFI bridge — all on your own hardware, with no cloud required.

This is private, early-access research code. **Do not share or distribute.**

Contact: savagetism@icloud.com

---

## What it does

`firefly_edgeos` is a continuously running agent that:

- **Senses the host**: CPU, RAM, battery, photons/audio/mass proxies.
- **Encodes experience**: BPE tokenization + 2048-D grounded embeddings fused with real sensor anchors.
- **Runs a native Transformer**: 576-dim, 4-block, 12-head Candle encoder with `conscience`, `goal`, and `language` heads, trained with AdamW and dynamic orthogonality regularization.
- **Thinks in a graph**: associative memory, causal world model, emotional homeostasis, and long-horizon planning.
- **Learns skills from one example**: `/skills/learn` generates, validates, and stores sandboxed Python tools.
- **Plans and replans**: decomposes active pursuits into multi-step plans, recalls skills and Sled strategies, and regenerates when steps fail.
- **Transfers across domains**: 12-task autonomous curriculum evaluates one-shot generalization, and writes a `PILOT_EVALUATION_METRICS.md` report with latency, RSS, and token telemetry.
- **Remembers who it is**: durable 10,000-entry identity journal, persisted across restarts, steering tool selection and emotional state.
- **Improves its own policy**: caches successful tool blueprints in Sled, tracks reliability, and prunes weak strategies automatically.
- **Operates in a wild sandbox**: watches `wild_workspace/`, ingests new files, and synthesizes read-only Python cleaners without touching the network. Demo scripts live in `wild_workspace/demo_scripts/`.
- **Distributes tool synthesis**: when the local work queue is heavy, `wild_workspace` offloads signed Task Engram Packets over the swarm; peers execute and return the finalized state vector.
- **Forms a wide-area swarm grid**: signed engrams over TCP, UDP, and WebSocket via `ConnectionManager` with exponential-backoff retries, transport auto-detection, an atomic backpressure guard, and a 2048-D cosine-similarity gate on every inbound frame.
- **Exposes everything on `http://127.0.0.1:8080`**: live dashboards, metrics, skill runner, transfer evaluator, and identity endpoints.
- **Exposes a C FFI bridge**: `build.rs` generates `firefly_core.h` and `cargo build --release` produces `libfirefly_edgeos.dylib` for native macOS interop.
- **Uses structured `tracing` logging**, `anyhow` error handling, and a centralized `Config` loaded from `FIREFLY_*` environment variables.
- **Routes all LLM generation through the native Apple Intelligence bridge**; no external LLM server is required.

---

## Quick start

Requires Rust, macOS with the native Apple Intelligence bridge built, and a `curriculum/` directory with `.txt` files.

```bash
git clone <private repo>
cd Firefly-EdgeOS
cargo build --release
./target/release/firefly_edgeos
```

The system starts training immediately, opens the telemetry server, and watches `wild_workspace/`.

To build the macOS C bridge:

```bash
cargo build --release
# generates firefly_core.h and target/release/libfirefly_edgeos.dylib
```

---

## HTTP endpoints

| Endpoint | Description |
|----------|-------------|
| `/telemetry` | Live telemetry, sensors, and state-save timing. |
| `/metrics` | Training metrics and summary JSON. |
| `/dashboard` | HTML dashboard with SVG sparklines and live efficiency metrics. |
| `/live` | Live streaming dashboard. |
| `/tools/run` | Run a sandboxed Python tool. |
| `/skills/learn` | Learn a Python skill from one example. |
| `/skills/run` | Execute a learned skill. |
| `/pursuits/add` | Inject or merge a new active pursuit. |
| `/transfer/evaluate` | Evaluate one-shot domain transfer. |
| `/identity` | Return the persisted narrative identity and journal. |

---

## Architecture at a glance

- `src/main.rs` — cognitive loop, planning, identity, memory, multi-agent wiring.
- `src/tensor_brain.rs` — 576-D Candle Transformer, BPE tokenizer, three heads, AdamW training, and dynamic orthogonality regularization.
- `src/conscience_oracle.rs` — LLM oracle + semantic cosine fallback.
- `src/apple_intelligence_client.rs` — Native Apple Intelligence oracle client with JSON repair and auto-fallback.
- `src/connectome_mmap.rs` — Zero-copy memory-mapped connectome persistence with `memmap2` and `bytemuck`.
- `src/strategy_library.rs` — Sled-backed durable cache for proven tool blueprints.
- `src/wild_workspace.rs` — Async directory watcher, payload processor, and distributed work-stealing queue with signed Task Engram Packets.
- `src/benchmark.rs` — Transfer and puzzle benchmark suites, plus `PilotReport` metrics.
- `src/telemetry.rs` — HTTP server, sandboxed tool runner, skill learner, metrics, and SVG dashboards.
- `src/metrics.rs` — Metrics logger, HTML dashboard, and `MemoryProfiler`.
- `src/protocol.rs` — Signed multi-transport engram fabric (TCP / UDP / WebSocket), `ConnectionManager`, lock-free ring with atomic backpressure, vector cosine firewall, and exponential-backoff retry state machine.
- `src/hyperdimensional_core.rs` — 10,000-D HDC vectors, script encoding, overhead analysis.
- `src/production_blueprint.rs` — Emotional homeostasis, memory graph, and world model.
- `src/governor.rs` — Dual-process metacognitive governor that scales orthogonality regularization and tick rate from loss, entropy, and thermal state.
- `src/config.rs` — Central runtime configuration.
- `src/lib.rs` + `build.rs` — C FFI bridge and generated `firefly_core.h`.

---

## Phase 4 production-kernel highlights

- **Zero-copy connectome persistence**: `ConnectomeMmap` maps `MemoryGraphNode` records into a fixed-size file; saves are offloaded and loads backfill missing embeddings from text.
- **Atomic backpressure guard**: `LockFreeRing` drops lowest-priority engrams when occupancy exceeds 85%, keeping the hot path non-blocking under load.
- **Vector cosine firewall**: every inbound UDP, TCP, and WebSocket engram is compared against the active-goal embedding matrix; packets below the threshold are rejected before entering the connectome.
- **Dynamic orthogonality regularization**: the `DualProcessGovernor` raises the `ortho_lambda_factor` when loss plateaus and entropy is low, nudging the 12 attention heads into distinct subspaces.
- **Distributed work-stealing**: `wild_workspace` queues pending tool tasks and, above a heavy-execution threshold, gossips signed `CompactEngramPacket` Task Engrams; peers execute them via `process_wild_source` / `process_script` and broadcast the finalized result back.
- **UDP packet-size safety**: `CompactEngramPacket` truncates `experiential_text` to 1024 characters and the UDP sender skips any signed payload that would exceed 60,000 bytes, preventing `Message too long` errors.

---

## Verified quality gates

```bash
cargo fmt --check   # pass
cargo clippy --all-targets -- -D warnings   # pass, zero warnings
cargo test          # 49 tests passed (23 lib + 26 bin)
cargo build --release   # pass, libfirefly_edgeos.dylib + firefly_core.h generated
```

- `protocol::tests::wan_tcp_roundtrip` — single-packet TCP roundtrip through `ConnectionManager`.
- `benchmark::tests::wan_tcp_load_and_pilot_report` — 25-engram load test with `PILOT_EVALUATION_METRICS.md` output.
- `connectome_mmap::tests::round_trip_records` — memory-mapped connectome save/load roundtrip.

---

## Key design constraints

- **No network access for tools**: sandboxed Python runs with a restricted module list.
- **No source-code self-modification**: the agent improves its cached strategies and reliability model, not its own Rust source.
- **Thread safety**: async code uses `tokio::sync::Mutex`; `spawn_blocking` calls use `.blocking_lock()`. `std::sync::Mutex` has been removed from the async hot path.
- **State survives restarts**: memory graph, identity journal, learned skills, Transformer weights, and connectome mmap are persisted on background threads.
- **Lock-free hot path**: the engram ring uses a `crossbeam` lock-free `SegQueue` with a priority-aware backpressure guard.

## Honest caveats

This is a research runtime and a scaffold, not a finished product.

- It is **not enterprise-grade line-rate infrastructure**.
- It is **not a real AGI** or a sentient system.
- The HDC-based code synthesizer is a **pattern-matching template engine**, not a full compiler from hypervectors.
- Latency, throughput, and thermodynamic numbers are approximations or derived from telemetry, not lab-benchmarked.

---

## License

See `LICENSE.txt`. Proprietary and confidential. No public distribution or commercial use without written permission from Adam Clark.
