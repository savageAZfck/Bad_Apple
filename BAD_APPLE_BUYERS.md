# Bad Apple — Buyer's Overview

## What it is

Bad Apple is a private, on-device AI assistant for macOS. It runs Qwen 3.5 on Apple Silicon using MLX, answers questions, runs local tools, indexes your files, and speaks responses through a local neural TTS server — all without sending prompts, responses, or actions to a cloud service after the initial model download.

## The pitch

- **Air-gapped by default**: no prompt, no action, no memory leaves your Mac.
- **Single unified 9B brain**: text and voice both route through the same Qwen 3.5 9B 4-bit model; no more dual-model swap lag.
- **Speculative decoding**: DFlash block-diffusion draft speeds up generation on Qwen 3.5's hybrid attention/GatedDeltaNet architecture.
- **Local tooling**: search files, list directories, run AppleScript, open apps, get the time, all from the daemon.
- **Persistent memory + RAG**: remembers user facts and searches indexed local documents.
- **Hot-reloadable persona**: edit `prompt.txt` without restarting the model.
- **Authenticated socket**: SLICKS HMAC challenge/response over a Unix socket.

## Models loaded

| Component | Model | Size (4-bit) | Role |
|---|---|---|---|
| Target LLM | `caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` | ~6.2 GB | All text and voice reasoning |
| DFlash draft | `z-lab/Qwen3.5-9B-DFlash` | ~2.4 GB | Speculative token blocks for the 9B target |
| Embeddings | `BAAI/bge-small-en-v1.5` | small | Local document / memory retrieval on CPU |
| TTS voice | `en_US-amy-medium` (default) | small | Piper neural speech on a local socket |

All models are cached on disk after the first download. Nothing is re-downloaded at runtime.

## Performance (live M-series Apple Silicon, 16 GB unified memory)

### Text mode

| Prompt tokens | First token | Tokens out | Decode tok/s | Draft acceptance | Peak memory |
|---|---:|---:|---:|---:|---:|
| ~460 | 3.1 s | 16 | 13.1 | 44% | 6.22 GB |
| ~560 | 4.4 s | 35 | 15.8 | 63% | 6.29 GB |
| ~580 | 5.4 s | 31 | 20.1 | 58% | 5.72 GB |
| ~620 | 5.7 s | 49 | 25.7 | 82% | 6.37 GB |
| ~730 | 6.6 s | 36 | 15.9 | 56% | 5.72 GB |

- Typical first-token latency: **~3.5–6.5 s** for 450–750 token prompts.
- Typical decode throughput: **~13–25 tok/s**, with spikes to ~36 tok/s on high-acceptance turns.
- System remains responsive: peak memory stays **~5.7–6.5 GB**, leaving ~78% of 16 GB free.

### Voice mode

| Prompt tokens | First token | Tokens out | Decode tok/s | Peak memory |
|---|---:|---:|---:|---:|
| ~430 | 2.8 s | 27 | 10.5 | 5.78 GB |
| ~430 | 3.9 s | 51 | 13.8 | 5.79 GB |
| ~460 | 3.1 s | 16 | 13.1 | 6.22 GB |
| ~460 | 5.7 s | 73 | 8.7 | 5.77 GB |

- Voice first-token latency: **~2.8–5.7 s** for the shorter 430–460 token voice prefill.
- Voice decode: **~9–15 tok/s**.
- No separate voice model is loaded anymore, so switching from text to voice is now a prompt change, not a model swap.

## What changed from the dual-brain setup

The previous build loaded a 9B model for text and a separate 4B model for voice. Switching modes caused a multi-second reload that could hit **16–25 s** and peak at ~10 GB. The current architecture:

- Keeps one 9B target + one 9B DFlash draft in memory.
- Eliminates the mode-switching stall.
- Uses ~4 GB less peak memory.
- Cuts the worst-case first-token delay after a switch to roughly the prefill time of the new prompt.

## Privacy & security

- Prompts and responses never leave the machine during normal use.
- Tool actions (file search, AppleScript, app open) run locally.
- Conversation history and user memory are stored locally, not synced.
- Client-to-daemon traffic is over a Unix socket with SLICKS HMAC challenge/response.
- The launchd daemon runs as root and auto-restarts.

## Features

| Feature | Description |
|---|---|
| Local Qwen 3.5 inference | 9B 4-bit on Apple Silicon GPU via MLX |
| Single-brain routing | same 9B for text and speech, no dual load |
| DFlash speculative decode | block-diffusion draft for faster generation |
| Streaming output | sentence chunks to terminal or TTS |
| Multi-turn history | saved locally |
| User memory | records and recalls facts |
| Local RAG | indexes text files with `bge-small-en-v1.5` |
| Tools | time, directory list, AppleScript, Spotlight search |
| Hot-reload persona | `prompt.txt` edits take effect on the next query |
| Voice | `badapple --speak` or `__BADAPPLE_VOICE__` mode |

## Use it like this

```bash
# text
badapple "What is 2+2?"

# voice (played via local Piper + afplay)
badapple --speak "What do you think of Siri?"
```

## Caveats

- **DFlash is deterministic for the same prompt**: identical questions get identical answers. Variance comes from different phrasing or a different seed/temperature, not from the draft path itself.
- **Roasts are part of the persona**: the model will needle cloud AI/Siri when bragging about bare metal. This can be tuned in `prompt.txt`.
- **Throughput is workload-dependent**: DFlash acceptance swings from ~45% to ~80%, so tok/s swings with it. Sustained 22–33 tok/s is possible on high-acceptance turns but not guaranteed for every prompt on this hardware.
- **First token includes prefill**: long prompts or large knowledge chunks push the first token toward the 5–7 s range.
- **TTS and Piper server must be running separately** for `--speak` to produce audio.

## System requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- ~6–7 GB of free unified memory at runtime (9B + DFlash loaded)
- ~20 GB of disk for the full model cache
- Initial model downloads require internet; everything after that is local

## Why buy / why build on this

If you want a macOS assistant that is actually yours — no subscriptions, no phone-home, no cloud snitching — and you can tolerate a few seconds of first-token latency, Bad Apple is the local-girl-in-the-machine option. The single 9B brain keeps memory and switching sane, the DFlash draft keeps decode above plain mlx-lm speeds, and the whole thing stays on your bare metal.
