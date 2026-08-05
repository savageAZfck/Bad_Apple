# Firefly AGI Core

**Sovereign, local-first AGI research runtime.**

Firefly is a self-contained, self-training cognitive OS written in Rust. It runs a 14,224-line async runtime (8,047 lines in `src/main.rs`) with a native Transformer, a local LLM oracle, an associative memory graph, a Sled-backed strategy library, a live telemetry server, a multi-transport swarm fabric, and a C FFI bridge — all on your own hardware, with no cloud required.

This is private, early-access research code. **Do not share or distribute.**

Contact: savagetism@icloud.com

---

## What it does

`sapient_soul` is a continuously running agent that:

- **Senses the host**: CPU, RAM, battery, photons/audio/mass proxies.
- **Encodes experience**: BPE tokenization + 2048-D grounded embeddings fused with real sensor anchors.
- **Runs a native Transformer**: 256-dim, 4-block, 8-head Candle encoder with `conscience`, `goal`, and `language` heads, trained with AdamW.
- **Thinks in a graph**: associative memory, causal world model, emotional homeostasis, and long-horizon planning.
- **Learns skills from one example**: `/skills/learn` generates, validates, and stores sandboxed Python tools.
- **Plans and replans**: decomposes active pursuits into multi-step plans, recalls skills and Sled strategies, and regenerates when steps fail.
- **Transfers across domains**: 12-task autonomous curriculum evaluates one-shot generalization, and writes a `PILOT_EVALUATION_METRICS.md` report with latency, RSS, and token telemetry.
- **Remembers who it is**: durable identity journal, persisted across restarts, steering tool selection and emotional state.
- **Improves its own policy**: caches successful tool blueprints in Sled, tracks reliability, and prunes weak strategies automatically.
- **Operates in a wild sandbox**: watches `wild_workspace/`, ingests new files, and synthesizes read-only Python cleaners without touching the network. Demo scripts live in `wild_workspace/demo_scripts/`.
- **Forms a wide-area swarm grid**: signed engrams over TCP, UDP, and WebSocket via an async `ConnectionRegistry` with exponential-backoff retries and transport auto-detection.
- **Exposes everything on `http://127.0.0.1:8080`**: live dashboards, metrics, skill runner, transfer evaluator, and identity endpoints.
- **Exposes a C FFI bridge**: `build.rs` generates `firefly_core.h` and `cargo build --release` produces `libsapient_soul.dylib` for native macOS interop.
- **Uses structured `tracing` logging**, `anyhow` error handling, and a centralized `Config` loaded from `FIREFLY_*` environment variables.
- **Routes all LLM generation through the native Apple Intelligence bridge**; no external LLM server is required.

---

## Quick start

Requires Rust, macOS with the native Apple Intelligence bridge built, and a `curriculum/` directory with `.txt` files.

```bash
git clone <private repo>
cd firefly-agi
cargo build --release
./target/release/sapient_soul
```

The system starts training immediately, opens the telemetry server, and watches `wild_workspace/`.

To build the macOS C bridge:

```bash
cargo build --release
# generates firefly_core.h and target/release/libsapient_soul.dylib
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

- `src/main.rs` — 8,047-line cognitive loop, planning, identity, memory, multi-agent wiring.
- `src/tensor_brain.rs` — Candle Transformer, BPE tokenizer, three heads, AdamW training.
- `src/conscience_oracle.rs` — LLM oracle + semantic cosine fallback.
- `src/apple_intelligence_client.rs` — Native Apple Intelligence oracle client with JSON repair and auto-fallback.
- `src/strategy_library.rs` — Sled-backed durable cache for proven tool blueprints.
- `src/wild_workspace.rs` — Async directory watcher and payload processor.
- `src/benchmark.rs` — Transfer and puzzle benchmark suites, plus `PilotReport` metrics.
- `src/telemetry.rs` — HTTP server, sandboxed tool runner, skill learner, metrics, and SVG dashboards.
- `src/metrics.rs` — Metrics logger and HTML dashboard.
- `src/protocol.rs` — Signed multi-transport engram fabric (TCP / UDP / WebSocket), `ConnectionRegistry`, and exponential-backoff retry state machine.
- `src/hyperdimensional_core.rs` — 10,000-D HDC vectors, script encoding, overhead analysis.
- `src/production_blueprint.rs` — Emotional homeostasis, memory graph, and world model.
- `src/config.rs` — Central runtime configuration.
- `src/lib.rs` + `build.rs` — C FFI bridge and generated `firefly_core.h`.

---

## Verified quality gates

```bash
cargo fmt --check   # pass
cargo clippy --all-targets -- -D warnings   # pass, zero warnings
cargo test          # 36 tests passed (17 lib + 19 bin)
cargo build --release   # pass, libsapient_soul.dylib + firefly_core.h generated
```

- `protocol::tests::wan_tcp_roundtrip` — single-packet TCP roundtrip through `ConnectionManager`.
- `benchmark::tests::wan_tcp_load_and_pilot_report` — 25-engram load test with `PILOT_EVALUATION_METRICS.md` output.

---

## Key design constraints

- **No network access for tools**: sandboxed Python runs with a restricted module list.
- **No source-code self-modification**: the agent improves its cached strategies and reliability model, not its own Rust source.
- **Thread safety**: async code uses `tokio::sync::Mutex`; `spawn_blocking` calls use `.blocking_lock()`. `std::sync::Mutex` has been removed from the async hot path.
- **State survives restarts**: memory graph, identity journal, learned skills, and Transformer weights are persisted on background threads.

## Honest caveats

This is a research runtime and a scaffold, not a finished product.

- It is **not enterprise-grade line-rate infrastructure**.
- It is **not a real AGI** or a sentient system.
- The HDC-based code synthesizer is a **pattern-matching template engine**, not a full compiler from hypervectors.
- The multi-agent fabric uses `tokio::sync::mpsc` batching, not true lock-free ring buffers.
- Latency, throughput, and thermodynamic numbers are approximations or derived from telemetry, not lab-benchmarked.

---

## License

See `LICENSE.txt`. Proprietary and confidential. No public distribution or commercial use without written permission from Adam Clark.
