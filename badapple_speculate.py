#!/usr/bin/env python3
"""Speculative decoding helpers for Bad Apple.

Scans the local HuggingFace cache for a small, compatible draft model and
loads it for `mlx-lm` speculative decoding. The draft must be small enough
that the verification overhead is smaller than the tokens it produces.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import mlx.core as mx


def _hub_root() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")).expanduser()


def _model_info_from_snap(snap: Path) -> dict[str, Any] | None:
    config = snap / "config.json"
    if not config.is_file():
        return None
    try:
        cfg = json.loads(config.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError):
        return None
    archs = cfg.get("architectures", [])
    arch = archs[0] if archs else "unknown"
    if "vision" in arch.lower() or "visual" in arch.lower() or "vision_config" in cfg or "visual" in cfg:
        return None
    size_bytes = sum(f.stat().st_size for f in snap.rglob("*") if f.is_file())
    return {
        "path": str(snap),
        "id": snap.parent.parent.name.replace("models--", "").replace("--", "/"),
        "size_gb": round(size_bytes / (1024 ** 3), 2),
        "architecture": arch,
        "hidden_size": cfg.get("hidden_size", 0),
        "num_layers": cfg.get("num_hidden_layers", 0),
    }


def scan_draft_candidates(max_size_gb: float = 2.0, prefer_arch: str = "Qwen") -> list[dict[str, Any]]:
    """Return cached models small enough to be plausible speculative drafts."""
    root = _hub_root()
    if not root.is_dir():
        return []
    candidates = []
    for model_dir in root.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue
        snapshots = model_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        for snap in snapshots.iterdir():
            if not snap.is_dir():
                continue
            info = _model_info_from_snap(snap)
            if info is None:
                continue
            if info["size_gb"] > max_size_gb:
                continue
            candidates.append(info)

    def sort_key(c: dict[str, Any]) -> tuple[int, float, int]:
        arch_match = 0 if prefer_arch.lower() in c["architecture"].lower() else 1
        return (arch_match, c["size_gb"], -c["num_layers"])

    return sorted(candidates, key=sort_key)


def find_draft_candidate(max_size_gb: float = 2.0, prefer_arch: str = "Qwen") -> str | None:
    """Find the best cached small draft model, or None."""
    candidates = scan_draft_candidates(max_size_gb=max_size_gb, prefer_arch=prefer_arch)
    return candidates[0]["path"] if candidates else None


def load_draft(model_ref: str | None = None) -> tuple[Any, Any] | None:
    """Load a draft model. If model_ref is None, auto-find a cached candidate."""
    try:
        from mlx_lm import load
    except ImportError as e:
        print(f"[speculate] mlx_lm not available: {e}", flush=True)
        return None

    target = model_ref or find_draft_candidate()
    if not target:
        print("[speculate] no cached draft candidate found; skipping speculative decoding.", flush=True)
        return None

    print(f"[speculate] loading draft model from {target}...", flush=True)
    t0 = time.time()
    try:
        model, tokenizer = load(target)
        print(f"[speculate] draft model loaded in {time.time() - t0:.1f}s", flush=True)
        return model, tokenizer
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        print(f"[speculate] could not load draft {target}: {e}", flush=True)
        return None


def unload_draft(model: Any) -> None:
    """Drop the draft model from memory."""
    if model is None:
        return
    try:
        del model
        gc.collect()
        mx.clear_cache()
    except Exception as e:  # noqa: BLE001 - cleanup
        print(f"[speculate] unload_draft cleanup ignored: {e}", flush=True)
