# Bad Apple Architecture Substrate // ANE Brain Performance Benchmark

This directory holds the in-process Apple Neural Engine (ANE) Qwen3-4B CoreML
benchmark and the acquisition pipeline for the local model artifacts.  It
implements the **Bad Apple** zero-copy, ANE-sharded cognitive substrate.

## Bad Apple 36-shard stateful FP16 layer strategy

The Bad Apple substrate shards the Qwen3-4B model into 36 independently compiled,
stateful FP16 layer blocks.  Each layer keeps its KV cache as opaque `MLState`
so no host/ANE KV data is moved per token, and the layer outputs are passed
zero-copy from one compiled shard to the next.  The embedding table is a single
memory-mapped FP16 tensor and the vocabulary is split across four contiguous
INT8 LM-head shards.  Every artifact is recorded in `conversion_manifest.json`
and the Swift runtime selects the sharded path automatically when the manifest
is passed as `BADAPPLE_ANE_MODEL`.

## Quick start

For the current Qwen3-4B substrate, convert a local GGUF through the
repository-local pipeline (see Conversion pipeline below).

For a quick legacy benchmark, place a pre-converted Qwen2.5-3B artifact in
`artifacts/` (see `Model artifacts` below) and run:

```bash
# Build the Swift bridge if it is not already present.
bash src/platform/apple_bridge/build_apple_bridge.sh

# Run the benchmark.
cargo test --release --test ane_brain_perf -- --nocapture
```

## Model artifacts

The benchmark discovers models in `artifacts/` in this priority order:

1. `qwen3b_ane_shards/` — locally converted Qwen3-4B FP16/INT8 sharded
   artifacts (current production substrate: 36 FP16 layer shards, four INT8
   LM-head shards, and a ~742 MB FP16 embedding table).
2. `qwen3b_ane/` — `darkmaniac7/TokForge-Qwen2.5-3B-CoreML-ANE-INT8`
   (legacy stateful, per-block INT8).
3. `qwen3b/` — `finnvoorhees/coreml-Qwen2.5-3B-Instruct-4bit`
   (legacy precompiled 4-bit `.mlmodelc`).
4. `qwen0.5b/` — `finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit`
   (smoke-test 0.5B).

Only one model is loaded per run; the first existing artifact is used.

## ANE vs. CPU/GPU residency notes

The `ane_core` Swift bridge (`BadAppleANECore`) requests
`MLComputeUnits.cpuAndNeuralEngine` and then audits the loaded
`MLComputePlan` to report the fraction of operations that were actually placed
on the ANE.  A 4-bit per-block/palettized model often fails ANE specialization
and falls back to CPU/GPU, which is reflected in the report.  The Qwen3-4B
production substrate uses per-layer FP16 transformer blocks; the LM heads are
quantized to INT8 per block.  The monolithic `darkmaniac7` INT8 artifact
triggered an ANECompiler `EXC_BAD_ACCESS` on the validation host, so it is
retained only as a reference.  The per-layer sharded pipeline below avoids that
full-graph compiler fault. The `finnvoorhees` 4-bit artifact remains a
legacy generation fallback and is selected through measured compute-unit
scoring.

## Conversion pipeline

The repository-local converter accepts a GGUF source and writes every layer as
an independently resumable stateful shard. Each conversion and compilation
runs in its own subprocess, and `conversion_manifest.json` is updated after
every successful stage without deleting prior attempts.

The conversion pipeline is being ported to a native Swift/Rust
toolchain and is not currently available from the command line. Pre-converted
artifacts can be placed in `artifacts/` and validated with:

```bash
# Audit all compiled compute plans and run a stateful shard prediction.
cargo test --release --test ane_brain_perf ane_shard_residency \
  -- --ignored --nocapture
```

On the current host, all 36 FP16 layer shards compiled successfully at
approximately 193 MB each. The production bundle adds a 742 MB FP16 embedding
table and four 93 MB INT8 vocabulary heads, for an approximately 7.9 GB runtime
footprint. `MLComputePlan` placed 41.09% of layer operations on ANE.
End-to-end greedy generation selected raw compute-unit 3, opened zero network
sockets, and produced 5.7 tokens/second at approximately 174 ms/token.

To boot the complete substrate, emit an offline continuation, report the BAD
APPLE baseline, and exit without entering the autonomous daemon loop:

```bash
BADAPPLE_ANE_MODEL="$PWD/tests/ane_brain_perf/artifacts/qwen3b_ane_shards/conversion_manifest.json" \
BADAPPLE_ANE_TOKENIZER="$PWD/tests/ane_brain_perf/artifacts/qwen3b_ane_shards/tokenizer.json" \
BADAPPLE_ANE_BOOT_PROMPT="In one sentence, describe Bad Apple." \
BADAPPLE_ANE_BOOT_TOKENS=20 BADAPPLE_ANE_BOOT_ONESHOT=1 cargo run --release
```

## Licensing

- Base model **Qwen3-4B** is released by Alibaba Cloud under the Qwen
  RESEARCH LICENSE AGREEMENT (non-commercial research use with attribution).
  See `LICENSE.qwen` or the upstream
  <https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE>.
- The Qwen2.5-3B conversions (`darkmaniac7`, `finnvoorhees`) remain available
  as legacy artifacts and inherit the same Qwen license.
- Apple **coremltools** is BSD-3-Clause.

## No external runtimes guarantee

The `ane_core` path uses only:

- the in-process `libBadAppleBridge.dylib` (Swift → CoreML),
- a local `.mlmodelc` directory,
- a local `tokenizer.json` / `tokenizer_config.json`,
- the Metal UMA `MTLResourceStorageModeShared` buffer for input token IDs.

No Ollama, no `llama.cpp` daemon, no `127.0.0.1:11434`, no cloud API.
