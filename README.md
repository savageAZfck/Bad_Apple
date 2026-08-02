# Firefly AGI Core

**Sovereign, local-first AGI research runtime.**

Firefly is a self-contained, self-training cognitive OS written in Rust. It runs a 6,835-line async runtime with a native Transformer, a local LLM oracle, an associative memory graph, a Sled-backed strategy library, and a live telemetry server — all on your own hardware, with no cloud required.

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
- **Transfers across domains**: 12-task autonomous curriculum evaluates one-shot generalization.
- **Remembers who it is**: durable identity journal, persisted across restarts, steering tool selection and emotional state.
- **Improves its own policy**: caches successful tool blueprints in Sled, tracks reliability, and prunes weak strategies automatically.
- **Operates in a wild sandbox**: watches `wild_workspace/`, ingests new files, and synthesizes read-only Python cleaners without touching the network.
- **Talks to itself on localhost**: signed UDP engrams to sibling agents on ports 5001–5010.
- **Exposes everything on `http://127.0.0.1:8080`**.

---

## Quick start

Requires Rust, an Ollama-compatible LLM at `127.0.0.1:11434`, and a `curriculum/` directory with `.txt` files.

```bash
git clone <private repo>
cd firefly-agi
cargo build --release
./target/release/sapient_soul
```

The system starts training immediately, opens the telemetry server, and watches `wild_workspace/`.

---

## HTTP endpoints

| Endpoint | Description |
|----------|-------------|
| `/telemetry` | Live telemetry, sensors, and state-save timing. |
| `/metrics` | Training metrics and summary JSON. |
| `/dashboard` | HTML dashboard with SVG sparklines. |
| `/tools/run` | Run a sandboxed Python tool. |
| `/skills/learn` | Learn a Python skill from one example. |
| `/skills/run` | Execute a learned skill. |
| `/pursuits/add` | Inject or merge a new active pursuit. |
| `/transfer/evaluate` | Evaluate one-shot domain transfer. |
| `/identity` | Return the persisted narrative identity and journal. |

---

## Architecture at a glance

- `src/tensor_brain.rs` — Candle Transformer, BPE tokenizer, three heads, AdamW training.
- `src/conscience_oracle.rs` — LLM oracle + semantic cosine fallback.
- `src/strategy_library.rs` — Sled-backed durable cache for proven tool blueprints.
- `src/wild_workspace.rs` — Async directory watcher and payload processor.
- `src/benchmark.rs` — Transfer and puzzle benchmark suites.
- `src/telemetry.rs` — HTTP server, sandboxed tool runner, skill learner, metrics.
- `src/main.rs` — Cognitive loop, planning, identity, memory, multi-agent wiring.
- `src/protocol.rs` — Signed UDP engram protocol.
- `src/metrics.rs` — Metrics logger and HTML dashboard.

---

## Key design constraints

- **No network access for tools**: sandboxed Python runs with a restricted module list.
- **No source-code self-modification**: the agent improves its cached strategies and reliability model, not its own Rust source.
- **Thread safety**: all Sled I/O and file watcher events are offloaded with `spawn_blocking`.
- **State survives restarts**: memory graph, identity journal, learned skills, and Transformer weights are persisted on background threads.

---

## License

See `LICENSE.txt`. Proprietary and confidential. No public distribution or commercial use without written permission from Adam Clark.
