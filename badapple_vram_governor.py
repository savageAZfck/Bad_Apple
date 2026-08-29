#!/usr/bin/env python3
"""VRAM governor for runtime model admission and pre-loading.

Decides whether the Mac can afford to load, keep, or pre-load a model based on
current unified-memory pressure, the model's memory budget, and what optional
models are currently resident.
"""

from __future__ import annotations

import os

import mlx.core as mx
import psutil


def _available_gb() -> float:
    """Return available physical RAM in GB, accounting for MLX cache pressure."""
    try:
        vm = psutil.virtual_memory()
        available_gb = vm.available / (1024 ** 3)
    except (OSError, AttributeError):
        available_gb = 8.0
    # Subtract the MLX active/cache working set so we don't double-count it.
    try:
        active_gb = mx.get_active_memory() / (1024 ** 3)
        cache_gb = mx.get_cache_memory() / (1024 ** 3)
        # Only account for cache above a modest pool; the active set is real usage.
        available_gb -= active_gb + max(0.0, cache_gb - 1.0)
    except Exception as e:  # noqa: BLE001
        print(f"[vram_governor] memory tracking error: {e}", flush=True)
    return max(0.0, available_gb)


def can_fit_model(memory_gb: float, headroom: float = 0.15) -> tuple[bool, float]:
    """Return (ok, available_gb) whether `memory_gb` can fit with headroom."""
    available = _available_gb()
    needed = memory_gb * (1.0 + headroom)
    return available >= needed, available


def can_fit_model_message(memory_gb: float, headroom: float = 0.15) -> tuple[bool, float, str]:
    """Return (ok, available_gb, message) with a human-friendly memory explanation."""
    ok, available = can_fit_model(memory_gb, headroom)
    needed = memory_gb * (1.0 + headroom)
    if ok:
        message = (
            f"This Mac has enough free memory to load the model safely "
            f"({available:.2f} GB available, {needed:.2f} GB needed)."
        )
    else:
        message = (
            f"This Mac does not have enough free memory to load the model safely. "
            f"{needed:.2f} GB is needed; only {available:.2f} GB is available. "
            f"Try closing other apps, unloading optional models, or choosing a smaller model."
        )
    return ok, available, message


def memory_pressure() -> str:
    """Return 'low', 'normal', or 'critical' based on physical memory pressure."""
    try:
        vm = psutil.virtual_memory()
        if vm.percent >= 90 or vm.available < 2 * (1024 ** 3):
            return "critical"
        if vm.percent >= 75:
            return "normal"
        return "low"
    except (OSError, AttributeError):
        return "unknown"


def recommend_model_for_query(query: str, available_gb: float) -> str:
    """Recommend the largest reasonable model given a query and available RAM."""
    low = query.lower()
    # Reasoning / coding / long-context queries want a big brain if possible.
    wants_big = any(k in low for k in (
        "reason", "deep", "complex", "analyze", "compare", "code review",
        "architecture", "design", "philosophy", "math proof", "debug",
    ))
    wants_small = any(k in low for k in (
        "hi", "hello", "time", "weather", "joke", "quick", "short", "simple",
        "what is", "who is", "how are", "thanks", "ping",
    ))
    if wants_small:
        return "fast_0.5b"
    if wants_big and available_gb >= 50:
        return "main_70b"
    if wants_big and available_gb >= 28:
        return "main_32b"
    # Default to the 9B workhorse.
    return "main_9b"


def recommend_for_memory(available_gb: float) -> str:
    """Recommend the best default model that fits comfortably."""
    if available_gb >= 50:
        return "main_70b"
    if available_gb >= 28:
        return "main_32b"
    if available_gb >= 10:
        return "main_9b"
    if available_gb >= 2:
        return "fast_0.5b"
    return "fast_0.5b"


def model_preload_priority() -> list[str]:
    """Default download priority list for background pre-loading."""
    priority = os.environ.get("BADAPPLE_MODEL_PRELOAD_PRIORITY", "fast_0.5b,main_9b,vision_2b,flux_4b").split(",")
    return [p.strip() for p in priority if p.strip()]
