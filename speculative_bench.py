#!/Users/savag3/bad_apple/.venv/bin/python
"""Bad Apple speculative-decoding benchmark.

Measures the raw tokens-per-second of the 8B target model with and without
draft models on a standard M4. This bypasses the socket/RAG overhead and
benchmarks only the MLX generation engine.
"""
import time

from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

TARGET = "mlx-community/Qwen3-8B-4bit"
DRAFT_MODELS = [
    None,
    ("mlx-community/Qwen3-0.6B-4bit", 3),
    ("mlx-community/Qwen3-1.7B-4bit", 2),
    ("mlx-community/Qwen3-1.7B-4bit", 3),
]
PROMPT = "Tell me a short, flirty story about a Mexican-American AI named Bad Apple, mi amor"
MAX_TOKENS = 200


def bench(model, tokenizer, draft=None, num_draft_tokens=3):
    tokens = tokenizer.encode(PROMPT, add_special_tokens=False)
    sampler = make_sampler(temp=0.6, top_p=0.9, top_k=20, min_p=0.05)
    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": tokens,
        "max_tokens": MAX_TOKENS,
        "sampler": sampler,
    }
    draft_accepted = 0
    total = 0
    if draft is not None:
        kwargs["draft_model"] = draft
        kwargs["num_draft_tokens"] = num_draft_tokens

    text = ""
    final = None
    start = time.time()
    for response in stream_generate(**kwargs):
        text += response.text
        total += 1
        if response.from_draft:
            draft_accepted += 1
        if response.finish_reason is not None:
            final = response
    wall = time.time() - start

    if final is None:
        return None
    return {
        "draft": draft.name if hasattr(draft, "name") else "none",
        "num_draft_tokens": num_draft_tokens,
        "generated_tokens": final.generation_tokens,
        "generation_tps": final.generation_tps,
        "wall_tps": final.generation_tokens / wall,
        "draft_accept_ratio": 100.0 * draft_accepted / total if total else 0.0,
        "peak_memory_gb": final.peak_memory,
    }


def main():
    print("Loading target model...", flush=True)
    target, tokenizer = load(TARGET)

    for entry in DRAFT_MODELS:
        if entry is None:
            draft, num = None, 3
            print("\nBenchmarking: no draft model", flush=True)
        else:
            draft_name, num = entry
            print(f"\nBenchmarking: draft={draft_name}, num_draft_tokens={num}", flush=True)
            draft, _ = load(draft_name)
        result = bench(target, tokenizer, draft, num)
        if result:
            print(f"  {result['generated_tokens']} tokens @ {result['generation_tps']:.1f} t/s "
                  f"(wall {result['wall_tps']:.1f} t/s), "
                  f"draft_accept={result['draft_accept_ratio']:.0f}%, "
                  f"peak={result['peak_memory_gb']:.2f} GB",
                  flush=True)
        else:
            print("  no result", flush=True)


if __name__ == "__main__":
    main()
