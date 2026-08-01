# Firefly AGI Core

**Private, early-access research codebase.**

Firefly is a local-first, self-training AGI research runtime built by Adam Clark. It combines a small native Transformer encoder with a local LLM oracle, an associative memory graph, multi-agent UDP exchange, and a telemetry/metrics dashboard. The system runs entirely on the host machine (no cloud required) and is designed for sovereign, private experimentation.

**THIS IS PRIVATE. DO NOT SHARE OR DISTRIBUTE.**

Contact: savagetism@icloud.com

## What it is

A Rust binary (`sapient_soul`) that continuously:

- Samples a training curriculum or external input.
- Produces a 2048-D grounded embedding from text and real system sensors (CPU, RAM, battery, photons/audio proxies).
- Runs a 256-dim, 4-block, 8-head Transformer encoder.
- Trains three heads:
  - `conscience_head`: 100-class cross-entropy classifier over a fixed token vocabulary.
  - `goal_head`: 100-class goal/intention generator trained on active pursuits.
  - `language_head`: 2048-D next-embedding predictor.
- Trains a small neural world model from brain state → next input.
- Stores experiences in an associative memory graph with offline structural cosine clustering.
- Broadcasts compact, signed engrams to other `sapient_soul` peers on localhost ports 5001–5010.
- Serves telemetry, metrics, and a dashboard on `http://127.0.0.1:8080`.

## Quick start

Requires Rust, an Ollama-compatible local LLM on `127.0.0.1:11434` (optional but recommended), and a curriculum directory with `.txt` files.

```bash
git clone <private repo>
cd firefly-agi
cargo build --release
./target/release/sapient_soul
```

The binary will start training immediately and open an HTTP server.

## HTTP endpoints

| Endpoint | Description |
|----------|-------------|
| `/telemetry` | Live telemetry and sensor snapshot as JSON. |
| `/metrics` | Recent training metrics and summary as JSON. |
| `/dashboard` | HTML dashboard with SVG sparklines and a recent-cycles table. |
| `/tools/run` | Run a sandboxed Python or shell tool. |

## Multi-agent localhost protocol

Peers discover each other on UDP ports 5001–5010. Each engram is sent as a `CompactEngramPacket` (JSON) carrying the 100-D sender brain state but not the 2048-D embedding, wrapped in a `SignedUdpPacket` with HMAC-SHA256. The shared key is read from `MULTI_AGENT_SECRET` or derived from the local hostname.

## Architecture at a glance

- `src/tensor_brain.rs` — Transformer, heads, training steps, AdamW optimizer.
- `src/conscience_oracle.rs` — LLM oracle + semantic cosine fallback for labels.
- `src/protocol.rs` — Signed UDP engram protocol.
- `src/metrics.rs` — Metrics logger and HTML dashboard.
- `src/telemetry.rs` — HTTP telemetry server.
- `src/ollama_client.rs` — Ollama client and constrained generation.
- `src/main.rs` — Main event loop, memory, agents, multi-agent wiring.

## Training status (current run)

- Conscience cross-entropy: frequently 0.5–5, spikes to ~10 on novel inputs.
- Language head loss: ~1.0 and falling.
- Goal head loss: training down from ~7 toward ~3–4.
- UDP signed engrams broadcast successfully to 10 peer ports per tick.

## License

See `LICENSE.txt`. This is proprietary, confidential, and unlicensed for public distribution or commercial use without written permission from Adam Clark.
