# Bad Apple performance benchmarks

Reproducible, honest comparisons between Bad Apple and other local AI tools,
run on the same Mac. The goal is real numbers, not marketing copy — including
when the numbers are unflattering.

## Running the Ollama comparison

```bash
# Requires Ollama running locally with a comparable model pulled:
ollama pull qwen2.5:7b-instruct-q4_0

cargo build --release
.venv/bin/python tools/benchmark_vs_ollama.py
```

Results are written to `benchmarks/results/bad_apple_vs_ollama_<timestamp>.{json,md}`.

## What this does and does not prove

- **Does not** compare identical model weights. Bad Apple runs an MLX-quantized
  9B model; Ollama runs a GGUF-quantized 7B model. Both are 4-bit-class,
  instruction-tuned, similar size class — it is the closest fair match
  available without training custom weights.
- **Does** measure what a real user gets from each tool's actual default local
  setup, on identical hardware, with the same prompts and output budget.
- Is sensitive to system memory pressure. This dev Mac has 16 GB of unified
  memory; loading both engines' models at once (~5 GB + ~4.5 GB) plus a full
  IDE/agent session leaves little headroom. The harness reports free memory %
  at start and warns when it's low. Re-run on an otherwise-idle machine for
  the most favorable, apples-to-apples numbers.

## Current honest result (2026-08-29, Apple M4, 16 GB RAM)

Run under heavy background load (active IDE/agent session, ~85%+ memory in use):

| Tool | Avg TTFT (s) | Avg decode tok/s | Peak mem (GB) |
|---|---:|---:|---:|
| Bad Apple (Qwen3.5-9B, MLX, 4-bit) | 13.2 | 9.1 | 5.35 |
| Ollama (Qwen2.5-7B, GGUF, llama.cpp/Metal) | 0.28 | 16.2 | ~0.01 (RSS proxy, not representative of GPU-resident weights) |

**Bad Apple is currently slower than Ollama on this machine, on both metrics.**
This does not match the general industry finding that MLX outperforms
llama.cpp by 20-30% on Apple Silicon — on this specific 16 GB machine, under
real dev-workstation load, with Bad Apple's fuller request pipeline (system
prompt cache management, persona system, retrieval hooks even when idle),
the gap runs the other way.

This benchmark run directly surfaced and led to fixing two real bugs:

1. `inference` (used by MCP, the Safari companion, and `--benchmark`) and the
   `--benchmark` code path called `render_prompt()` directly without ensuring
   the lazily-loaded main model was resident first, throwing
   `AttributeError: 'NoneType' object has no attribute 'apply_chat_template'`
   on a cold daemon. This was the exact bug behind Bad Apple occasionally
   speaking raw Python errors out loud via TTS.
2. `active_models()` (used by every `runtime_status`/dashboard poll) accessed
   `self.model` directly instead of defensively, causing a repeating
   `AttributeError: 'MLXServer' object has no attribute 'model'` during the
   startup window before `__init__` finishes.

## Known follow-up work (not yet done)

- One prompt showed a 42.7s time-to-first-token outlier — a real anomaly
  worth root-causing (candidates: system prompt cache invalidation, thermal
  throttling, or memory-pressure-induced swapping) rather than averaged away.
- Re-run on an idle machine (nothing else open) to get a genuinely fair
  best-case number instead of one confounded by this being a live dev machine.
- If Bad Apple's decode throughput remains behind Ollama/llama.cpp on
  comparable hardware even when idle, investigate `prefill_step_size`,
  `max_kv_size`, and whether the per-query context-building work in
  `render_prompt` (RAG retrieval, workspace summary, memory graph lookups)
  is adding meaningful overhead versus Ollama's simpler stateless request path.
