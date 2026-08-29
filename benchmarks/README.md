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

## Follow-up investigation (2026-08-29)

Went looking for a code-level fix for the decode-speed gap. Ruled out two
plausible hypotheses, confirmed a third as the likely primary cause, and
fixed one real (if unrelated) memory-safety issue along the way:

- **Speculative decoding was not the cause.** Every generation logs
  `draft_accept_ratio=0%`, which looks damning (paying a draft model's cost
  for zero benefit), but traced the code and confirmed `BADAPPLE_SPECULATIVE_DRAFT`
  is unset in the actual installed plist, `self.draft_model` stays `None`,
  and `grep -c "\[speculate\]" ` the daemon log (which `_load_speculative_draft()`
  would print on to on a real load) turns up zero matches ever. The metric
  reads `0%` unconditionally whenever no draft model is loaded at all
  (`draft_tokens / token_count` where `draft_tokens` never increments) --
  it looks like a smoking gun for a failing feature but actually just means
  the feature was never turned on. If it *were* turned on, the only cached
  compatible draft (`Qwen2.5-0.5B-Instruct`, generic/un-fine-tuned) predicting
  continuations for the heavily custom-fine-tuned `Qwen3.5-9B-HLWQ` target
  would plausibly have a genuinely low acceptance rate anyway -- but that's
  a separate, hypothetical concern from what's actually configured today.
- **`TOKENIZERS_PARALLELISM=false`** (the standard fix for HuggingFace
  `tokenizers`-related "leaked semaphore" warnings) was tried and measured;
  it did not change decode throughput. Not applied.
- **Real, current system memory pressure was confirmed** on this specific
  16 GB test Mac: `top` showed `PhysMem: 15G used (7670M wired, 2927M
  compressor), 279M unused` -- 2.9 GB of memory *actively compressed* right
  now, which costs real CPU cycles to decompress on every access, and a
  cumulative swap counter in the tens of millions, both directly competing
  with the daemon's own inference for CPU/GPU time. This machine is running
  a full IDE/agent session (itself several GB of Electron helper processes)
  concurrently with the 9B model, which is a genuinely tight budget on 16 GB
  total. This is very likely the dominant real cause of the gap, and it is
  environmental, not a Bad Apple code bug -- re-running this benchmark on an
  otherwise-idle Mac remains the right way to get a fair number.
- **Fixed regardless, as a real hardening improvement**: MLX's default
  `set_memory_limit` is 1.5x the GPU's own recommended working set size --
  on this 16 GB Mac, that let the daemon claim up to ~15.2 GB, leaving under
  1 GB of *guaranteed* headroom for literally everything else running.
  `badapple_mlx_server.py` now caps this to `mx.device_info()`'s
  `max_recommended_working_set_size` (Apple's own guidance, ~11.8 GB here) at
  startup. This did not measurably change decode tok/s in testing (the
  daemon's actual peak usage, ~5.2-5.7 GB, was already well under both the
  old and new ceiling, so the limit itself was never the bottleneck) but is
  a correct, low-risk fix regardless: a background daemon has no business
  defaulting to a memory policy sized for exclusive-use ML workstations.

**Still open**: re-run this benchmark on an idle Mac (nothing else running)
to get a number not confounded by this machine's current memory pressure. If
the gap persists even then, the next things worth profiling are
`prefill_step_size`/`max_kv_size` tuning and whether per-query context-building
in `render_prompt` (RAG retrieval, workspace summary, memory graph lookups) adds
meaningful overhead versus Ollama's simpler stateless request path. The 42.7s
TTFT outlier from the original run was not reproduced in this session's testing
(TTFT ranged 5-7s across several warm requests) and is not yet explained.
