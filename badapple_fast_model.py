#!/usr/bin/env python3
"""Tiny fast local model for Bad Apple's dynamic tiering.

Loads a small MLX model on demand and generates short chitchat/greeting
responses without waking the 9B brain. No cloud.
"""

import gc
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler


def _default_fast_model() -> str | None:
    """Return the default downloaded tiny model path if it exists."""
    candidates = [
        "/Users/savag3/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
    ]
    for p in candidates:
        if Path(p).is_dir() and (Path(p) / "config.json").is_file():
            return p
    return None


def fast_model_path() -> str | None:
    """Resolve the tiny fast model path from env only.

    The 0.5B default is no longer auto-loaded because it produces vague,
    repetitive chitchat.  Set BADAPPLE_FAST_MODEL to a larger model (1.5B+)
    if you want a neural fast tier.
    """
    return os.environ.get("BADAPPLE_FAST_MODEL")


def load_fast_model(path: str | None = None) -> tuple[Any, Any] | None:
    """Load a tiny MLX model and tokenizer. Returns (model, tokenizer) or None."""
    p = path or fast_model_path()
    if not p:
        print("[fast_model] no fast model configured", flush=True)
        return None
    try:
        print(f"[fast_model] loading tiny model from {p} ...", flush=True)
        model, tokenizer = load(p)
        print("[fast_model] tiny model loaded.", flush=True)
        return model, tokenizer
    except Exception as e:
        print(f"[fast_model] failed to load tiny model: {e}", flush=True)
        return None


def generate_fast(
    model: Any,
    tokenizer: Any,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 48,
    temperature: float = 0.0,
) -> str:
    """Generate a short response from the tiny model."""
    messages = [
        {"role": "system", "content": system_prompt or "You are Bad Apple, a concise, confident local assistant. Keep answers short and helpful."},
        {"role": "user", "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        tokens = tokenizer.encode(text)

    sampler = make_sampler(temperature)
    response = generate(
        model,
        tokenizer,
        tokens,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )
    return response


def unload_fast_model(model: Any) -> None:
    """Best-effort attempt to clear the tiny model from memory."""
    try:
        del model
        gc.collect()
        mx.clear_cache()
    except Exception:
        pass
