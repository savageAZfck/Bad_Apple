# Bad Apple vs Ollama — reproducible local benchmark

Run: 2026-08-29 08:49:14
Host: Apple M4, 16.0 GB RAM, macOS 26.6.2, 40% memory free at start

**Caveat 1 (models):** this does not compare identical model weights. Bad Apple runs `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` (MLX). Ollama runs `qwen2.5:7b-instruct-q4_0` (GGUF via llama.cpp/Metal). Both are 4-bit-class quantized instruction models in a similar size class. This measures what a real user gets from each tool's default local setup on identical hardware, not raw model-vs-model quality.

**Caveat 2 (system load):** this machine has 16 GB of unified memory, which both engines' models (~5 GB Bad Apple MLX + ~4.5 GB Ollama GGUF) compete for, along with whatever else is running. Numbers recorded with significant background load (an active IDE/agent session, browser, etc.) will be pessimistic versus an idle machine. Re-run with nothing else open for the most favorable, reproducible numbers.

## Summary

| Tool | Avg TTFT (s) | Avg decode tok/s | Max RSS/peak mem (GB) | Errors |
|---|---:|---:|---:|---:|
| bad_apple | 13.21 | 9.1 | 5.35 | 0 |
| ollama | 0.28 | 16.2 | 0.01 | 0 |

## Per-prompt results

| Tool | Prompt | Tokens | TTFT (s) | Decode tok/s | Mem (GB) |
|---|---|---:|---:|---:|---:|
| bad_apple | What is the capital of France? | 20 | 42.74 | 6.9 | 5.35 |
| ollama | What is the capital of France? | 8 | 0.23 | 21.0 | 0.01 |
| bad_apple | Write a haiku about the ocean. | 24 | 6.17 | 8.4 | 5.35 |
| ollama | Write a haiku about the ocean. | 20 | 0.23 | 17.9 | 0.01 |
| bad_apple | Explain how a car engine works in two se | 57 | 6.03 | 11.2 | 5.35 |
| ollama | Explain how a car engine works in two se | 46 | 0.25 | 11.1 | 0.01 |
| bad_apple | What is 15 times 24? | 30 | 5.60 | 9.4 | 5.35 |
| ollama | What is 15 times 24? | 15 | 0.26 | 18.9 | 0.01 |
| bad_apple | Summarize the plot of Romeo and Juliet i | 60 | 5.50 | 9.8 | 5.35 |
| ollama | Summarize the plot of Romeo and Juliet i | 60 | 0.43 | 11.9 | 0.01 |
