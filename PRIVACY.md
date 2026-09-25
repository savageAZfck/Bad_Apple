# Privacy Policy

Bad Apple is a **cloudless, on-device** cognitive runtime. This policy explains what data is processed and where.

## No cloud dependency

- No source code, embeddings, curriculum text, or state snapshots are sent to any remote server.
- The 576-D transformer trunk runs locally using `candle-core` on Apple Silicon Metal.
- The Apple Intelligence path uses on-device Apple frameworks and does not call external LLM APIs.
- The swarm fabric only sends signed engrams to peers explicitly configured by the user.

## What data stays local

| Data | Location | Notes |
|------|----------|-------|
| Transformer weights | `state.safetensors`, `sapient_agi_soul_*.safetensors` | Loaded into process memory at runtime. |
| Connectome / memory graph | `state.connectome`, `state.json` | Persisted memory-mapped structures. |
| Identity journal | `bad_apple_state.json`, `bad_apple_state*.json` | 10,000-entry durable narrative identity. |
| Learned skills & strategies | `strategy_db/` Sled store | Caches proven tool blueprints and reliability scores. |
| Curriculum | `curriculum/*.txt` | User-provided training text. |
| Wild workspace | `wild_workspace/` | Watched directory for unsupervised ingestion. |
| Telemetry / metrics | `metrics.jsonl`, `bad_apple.log` | Local logs and dashboard feed. |

## What may leave the machine

The only way data can leave the machine is if the user explicitly:

- Configures a swarm peer address for `ConnectionManager`.
- Copies state files or logs out of the runtime directory manually.
- Builds a custom tool or shortcut that performs network I/O.

The default configuration is strictly local.

## Telemetry

The `PILOT_EVALUATION_METRICS.md` report and `metrics.jsonl` files are written to disk in the runtime directory. They are not transmitted anywhere. The live dashboard at `http://127.0.0.1:8787` is served only on localhost.

## Data retention

State files, logs, and the strategy database are retained until the user deletes them. Bad Apple does not phone home or perform automatic cleanup.

## Third-party dependencies

The runtime dependencies are open-source Rust crates loaded at compile time. They are listed in `Cargo.toml` and `Cargo.lock`. No runtime third-party service is contacted.

## Contact

For privacy or data-handling questions, contact the author via the repository contact information.
