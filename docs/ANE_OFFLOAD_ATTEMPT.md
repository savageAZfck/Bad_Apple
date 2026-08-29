# Neural Engine offload: what was tried, what's blocked, and why

## Goal

Bad Apple's semantic cache and RAG retrieval run `BAAI/bge-small-en-v1.5`
through PyTorch on CPU (`badapple_extras.SemanticCache._encode`), sitting
next to an MLX GPU workload doing the heavy lifting for the main 9B model.
The Apple Neural Engine sits completely idle. The goal was to convert the
embedding model to CoreML and route it through the ANE, freeing CPU cycles
without touching the GPU/MLX resources the main brain needs — genuine
parallel hardware use rather than a raw-speed claim.

## Why not the main 9B brain

`tests/ane_brain_perf/README.md` already measured the existing 36-shard
stateful ANE substrate for a *different, smaller* model (Qwen3-4B) at
**5.7 tokens/second**, against MLX's 15-20+ tokens/second for Bad Apple's
current 9B model on the same class of hardware. Pursuing "run the main
brain on the ANE" would very likely make Bad Apple slower, not faster; a
4-bit-per-block/palettized LLM frequently fails ANE operator specialization
and falls back to CPU/GPU anyway (documented in the same README). This is
why this effort targeted the small embedding model instead, where the ANE's
efficiency-over-throughput profile is a much better fit.

## What was tried

1. **Direct conversion** with the current venv's `torch` (2.13.0) and a
   freshly installed `coremltools` (9.0), tracing `AutoModel.from_pretrained`
   with `torch.jit.trace`. Failed: `TypeError: only 0-dimensional arrays can
   be converted to Python scalars`, inside `BertEmbeddings`' internal
   `int()` cast of a sliced position-id buffer.

2. **Explicit `position_ids`/`token_type_ids` inputs** via a tracing
   wrapper, to bypass the internal dynamic buffer slicing that caused (1).
   This got past that error, but hit a new one: `NotImplementedError:
   PyTorch convert function for op 'new_ones' not implemented`, coming from
   `transformers`' newer `masking_utils` module building an extended
   attention mask with a data-dependent shape. Tried both `sdpa` and
   `eager` attention implementations; both produce dynamic-shape ops that
   this coremltools version's torch frontend cannot lower to a static graph.

3. **Isolated conversion environment** with older, previously
   coreml-tested versions (`torch==2.2.2`, `transformers==4.36.2`,
   `coremltools==7.2`, pre-dating `transformers`' masking-utils refactor),
   built in a throwaway venv so the live daemon's dependencies were never
   touched. Hit environment-level blockers instead: NumPy 2.x/1.x ABI
   mismatch (fixed by pinning `numpy<2`), then Python 3.12's removal of
   `distutils` from the standard library (fixed by installing
   `setuptools`), then finally: **`coremltools==7.2`'s compiled native
   extensions (`libcoremlpython`, `libmilstoragepython`) failed to import
   at all** on this Python 3.12 / macOS 26 / Apple Silicon combination —
   there is no working prebuilt wheel for this exact combination, and the
   MIL graph conversion path that doesn't need those extensions still
   wouldn't be able to save a usable `.mlmodel`/`.mlpackage`.

## Conclusion

This is genuinely blocked by tooling compatibility, not a one-line code fix.
Getting a working CoreML/ANE pipeline for even a small BERT-family model on
this exact toolchain (Python 3.12, current `transformers`, current macOS)
would require one of:

- Building `coremltools` from source for this Python/macOS combination, or
- Downgrading the *entire* conversion toolchain in a fully isolated
  environment with a Python version coremltools' older wheels actually
  support (likely Python 3.10 or 3.11), or
- Converting through an ONNX intermediate step (`optimum`'s ONNX export +
  `coremltools.converters.onnx`), which has historically had better
  compatibility with newer `transformers` internals than direct
  torch-to-CoreML tracing, or
- Waiting for a `coremltools` release with prebuilt wheels that support
  both current `transformers`' masking utilities and this Python version.

`coremltools` was uninstalled from the project venv after this investigation
so it doesn't sit as unused dead weight in `requirements.txt`.

## What this means for the "beat the competition" framing

The honest ANE story right now is: Bad Apple's *design* for using all three
compute engines (CPU control plane, GPU main-brain decode via MLX, ANE for
small parallel tasks) is sound and differentiated — no competitor surveyed
(Ollama, LM Studio, Cognithor, Aiden, Tactile) attempts ANE utilization at
all. But the *execution* is currently blocked on this machine's toolchain,
not proven. Do not claim ANE offload as a shipped feature until one of the
paths above actually produces a working, benchmarked `.mlpackage`.
