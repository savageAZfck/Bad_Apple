# Firefly EdgeOS

**A sovereign, single-core-saturated, self-training cognitive operating system.**

Firefly EdgeOS is not a distributed system, a microservice mesh, or a cloud API wrapper. It is a single-threaded-at-the-core, hardware-aware cognitive runtime that keeps one Apple Silicon performance core packed and hands every heavy, non-deterministic, or I/O-bound operation to lock-free background lanes. It trains its own 576-dimensional Candle transformer on your machine, persists state to disk without blocking the cognitive clock, defends its swarm ports with a 2048-D vector firewall, and learns, plans, and reasons inside a self-contained process.

This is a research runtime and a scaffold. It is private, early-access code.

---

## What makes this different

Most machine-learning systems treat the CPU as a scheduler for a cloud of workers, GPUs, and network calls. Firefly EdgeOS inverts that model.

- **Single-core saturation by design.** One performance thread runs the transformer, the memory graph, the planner, and the learner back-to-back, at >90% core utilization, with no yield to cross-core coordination.
- **Zero-copy state persistence.** Tensor weights are snapshotted as lightweight `Arc` references and passed to a dedicated background thread. The foreground loop pays only the handoff, not the deep copy.
- **Lock-free task offloading.** Wild-workspace and swarm packets are posted to `crossbeam`-backed `LockFreeRing`s with atomic priority-based backpressure. No mutexes, no thread-pool dispatch, no stalls.
- **Hardware-fused network firewall.** The 2048-D cosine-similarity gate is implemented with ARM64 NEON intrinsics, streaming four `f32` lanes per 128-bit vector and zeroing NaN/Inf with `vbslq_f32`.
- **Local-first, cloud-zero.** All training, inference, state, and swarm traffic stays on the host. No remote model endpoints, no telemetry exfiltration, no network calls for tool execution.

This is a deliberate shift in the mental model of how a cognitive engine is built: one core, one stream of execution, one persistent self, with the rest of the machine reduced to an I/O and transport support plane.

---

## Core capabilities

- **Native Transformer training.** 576-dim, 4-block, 12-head Candle encoder with `conscience`, `goal`, and `language` heads, AdamW, layer-wise learning-rate decay, and dynamic orthogonality regularization.
- **Self-supervised curriculum.** Trains on local text from `curriculum/`, continuously, from process start, using a self-generated BPE tokenizer.
- **Conscience head with hard LR floor.** If the cross-entropy loss stays above `ln(100)` past cycle 50, the optimizer clamps to a 0.005 floor so the head breaks symmetry and converges.
- **Xavier/Glorot head initialization.** The three transformer heads are initialized with scaled, clamped weights so gradients flow within the 0.01 stability band.
- **Associative connectome.** A `HashMap`-based memory graph with 2048-D grounded embeddings, cosine-similarity edges, and a 10,000-entry identity journal.
- **Emotional homeostasis and world model.** `production_blueprint.rs` and `governor.rs` modulate valence, arousal, planning mode, and orthogonality regularization from live loss and entropy.
- **One-shot skill learning.** `/skills/learn` generates sandboxed Python functions from one example, validates them, and stores them in Sled.
- **Wild workspace.** Watches a local directory and either runs read-only cleaners locally or offloads signed `CompactEngramPacket`s to peers when the queue saturates.
- **Signed multi-transport swarm fabric.** TCP, UDP, and WebSocket gossip through `ConnectionManager`, with HMAC signing, exponential-backoff retry, transport auto-detection, and a 2048-D cosine firewall on every inbound frame.
- **Lock-free rings everywhere.** `protocol.rs` already used `LockFreeRing` for incoming engrams. The new outbound ring makes wild-workspace task offloading completely non-blocking.
- **Candle tensor snapshots without blocking.** `StateSaveWorker` flushes state, safetensors, connectome mmap, and legacy weight files on a background `std::thread` while the cognitive loop resumes immediately.
- **Live telemetry and HTTP API.** Axum server on `http://127.0.0.1:8080` with `/telemetry`, `/metrics`, `/dashboard`, `/tools/run`, `/skills/learn`, `/skills/run`, `/pursuits/add`, `/transfer/evaluate`, and `/identity`.
- **C FFI bridge.** `cargo build --release` produces `libfirefly_edgeos.dylib` and a generated `firefly_core.h` for macOS interop.

---

## Measured performance

These numbers come from the live telemetry endpoint on an Apple M4 Max:

| Metric | Value | Note |
|---|---|---|
| State-save foreground handoff | **369 µs** | From main-loop start to `StateSaveWorker` queue, after removing the Tensor deep copy from the hot path. |
| State-save background flush | **27-56 ms** | Time for the worker to deep-copy tensors and write JSON/safetensors/mmap. The foreground loop is unblocked. |
| Conscience loss convergence | below 4.0 by cycle 6 | On a fresh state file with the new Xavier heads and LR floor. |
| Single-core operation | 90%+ sustained | Cognitive tick, training, and inference saturate one core; all persistence and network work is backgrounded. |

The handoff is the figure that matters. A full save used to block the main thread for hundreds of milliseconds. It now completes in under half a millisecond, letting the transformer training pipeline maintain mechanical sympathy with the CPU pipeline.

---

## Architecture at a glance

| Module | Responsibility |
|---|---|
| `src/main.rs` | Cognitive loop, planning, identity, memory, multi-agent wiring, and state-save orchestration. |
| `src/tensor_brain.rs` | 576-D Candle Transformer, BPE tokenizer, three heads, AdamW, LLRD, orthogonality regularization, and the zero-copy `snapshot_weights` path. |
| `src/conscience_oracle.rs` | LLM oracle routing and semantic cosine fallback. |
| `src/apple_intelligence_client.rs` | Native Apple Intelligence client with JSON repair and fallback. |
| `src/connectome_mmap.rs` | Zero-copy memory-mapped connectome persistence. |
| `src/strategy_library.rs` | Sled-backed durable cache for proven tool blueprints with reliability tracking. |
| `src/wild_workspace.rs` | Lock-free wild-workspace watcher, task queue, and distributed offloading via `push_outgoing`. |
| `src/protocol.rs` | Signed multi-transport engram fabric, `ConnectionManager`, inbound and outbound `LockFreeRing`s, and the NEON 2048-D cosine firewall. |
| `src/state_saver.rs` | Background `StateSaveWorker` that flushes double-buffered state snapshots. |
| `src/telemetry.rs` | HTTP server, sandboxed tool runner, skill learner, metrics, and dashboard. |
| `src/metrics.rs` | `MemoryProfiler`, `LatencyRingBuffer`, SVG dashboards. |
| `src/governor.rs` | `DualProcessGovernor` scales orthogonality and tick rate from loss, entropy, and thermal state. |
| `src/hyperdimensional_core.rs` | 10,000-D HDC vectors, script encoding, overhead analysis. |
| `src/production_blueprint.rs` | Emotional homeostasis, memory graph, and causal world model. |
| `src/config.rs` | Centralized `FIREFLY_*` environment configuration. |
| `src/lib.rs` + `build.rs` | C FFI bridge and generated `firefly_core.h`. |

---

## Quick start

Requires Rust, macOS with Apple Intelligence support, and a `curriculum/` directory of `.txt` files.

```bash
git clone https://github.com/savageAZfck/Firefly-EdgeOS.git
cd Firefly-EdgeOS
cargo build --release
./target/release/firefly_edgeos
```

The runtime starts training immediately, opens the telemetry server at `http://127.0.0.1:8080/telemetry`, and watches `wild_workspace/`.

To build the C bridge:

```bash
cargo build --release
# generates firefly_core.h and target/release/libfirefly_edgeos.dylib
```

---

## HTTP endpoints

| Endpoint | Description |
|---|---|
| `/telemetry` | Live telemetry, sensors, and state-save timing. |
| `/metrics` | Training metrics and summary JSON. |
| `/dashboard` | HTML dashboard with SVG sparklines. |
| `/live` | Live streaming dashboard. |
| `/tools/run` | Run a sandboxed Python tool. |
| `/skills/learn` | Learn a Python skill from one example. |
| `/skills/run` | Execute a learned skill. |
| `/pursuits/add` | Inject or merge a new active pursuit. |
| `/transfer/evaluate` | Evaluate one-shot domain transfer. |
| `/identity` | Return the persisted narrative identity and journal. |

---

## Verified quality gates

```bash
cargo fmt --check
cargo clippy --all-targets --all-features --release -- -D warnings
cargo test --release
cargo build --release
cargo deny check
```

Results on the reference M4 Max:

- `cargo clippy` — zero warnings, both `Firefly-EdgeOS` and `firefly_inferno`.
- `cargo test --release` (EdgeOS) — 49 tests passed (23 lib + 26 bin).
- `cargo test --release` (firefly_inferno) — 26 tests passed across agent, compiler, coprocessor, integration, and SLICKS suites.
- `cargo deny check` — pass; only pre-existing duplicate-dependency warnings.

---

## Key design constraints

- **No network access for tools.** Sandboxed Python runs with a restricted module list.
- **No source-code self-modification.** The agent improves its cached strategies, not its own Rust source.
- **Single-core cognitive hot path.** Background `std::thread` and `tokio` tasks handle persistence and transport only.
- **Lock-free hot path.** The engram and task rings are `crossbeam` `ArrayQueue`s; priority-aware backpressure drops low-priority items above 85% occupancy.
- **FFI safety.** Apple Intelligence calls are serialized by the Swift `NSLock` inside the Siri bridge; Rust paths stay lock-free.
- **State survives restarts.** Memory graph, identity journal, learned skills, transformer weights, and connectome mmap are persisted on background threads.

---

## Honest caveats

This is a research runtime and a scaffold, not a shipping product.

- It is **not enterprise-grade line-rate infrastructure**.
- It is **not a real AGI** or a sentient system.
- The HDC-based code synthesizer is a **pattern-matching template engine**, not a full compiler from hypervectors.
- Throughput, latency, and thermodynamic figures are telemetry-derived measurements on one machine, not lab-benchmarked guarantees.

---

## License

See `LICENSE.txt`. Proprietary and confidential. No public distribution or commercial use without written permission from Adam Clark.
