#!/usr/bin/env python3
"""Download pre-converted CoreML Qwen artifacts for the ANE brain benchmark.

This script fetches the smallest available 3B weight set (or the 0.5B smoke
artifact) from Hugging Face and stages it under
`tests/ane_brain_perf/artifacts`.  It does not convert or quantize models
itself; that pipeline is documented in `convert_qwen_coreml.py` (the
reproducible coremltools 9 recipe from the TokForge community artifact).

Usage:
    python3 tests/ane_brain_perf/download_models.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required; install with: "
        "pip install huggingface-hub 'huggingface-hub[hf_xet]'"
    ) from exc

ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"

REPOS = {
    "qwen3b": {
        "repo_id": "finnvoorhees/coreml-Qwen2.5-3B-Instruct-4bit",
        "patterns": [
            "Qwen2.5-3B-Instruct-4bit.mlmodelc/**",
            "tokenizer.json",
            "tokenizer_config.json",
            "README.md",
            "config.json",
        ],
        "model_dir": "Qwen2.5-3B-Instruct-4bit.mlmodelc",
    },
    "qwen3b_ane": {
        "repo_id": "darkmaniac7/TokForge-Qwen2.5-3B-CoreML-ANE-INT8",
        "patterns": [
            "model.mlmodelc.zip",
            "tokenizer.json",
            "tokenizer_config.json",
            "README.md",
        ],
        "zip": "model.mlmodelc.zip",
        "unzip_to": "model.mlmodelc",
    },
    "qwen0.5b": {
        "repo_id": "finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit",
        "patterns": [
            "Qwen2.5-0.5B-Instruct-4bit.mlmodelc/**",
            "tokenizer.json",
            "tokenizer_config.json",
            "README.md",
            "config.json",
        ],
        "model_dir": "Qwen2.5-0.5B-Instruct-4bit.mlmodelc",
    },
}


def download(name: str, *, force: bool = False) -> Path:
    spec = REPOS[name]
    local_dir = ARTIFACT_ROOT / name
    if local_dir.exists() and not force:
        print(f"{name}: already present at {local_dir}")
        return local_dir

    print(f"{name}: downloading from {spec['repo_id']} ...")
    shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=spec["repo_id"],
        allow_patterns=spec["patterns"],
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    if "zip" in spec:
        zip_path = local_dir / spec["zip"]
        if zip_path.exists():
            unzip_dir = local_dir / spec["unzip_to"]
            print(f"{name}: unzipping {zip_path} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(unzip_dir)
        else:
            print(f"{name}: warning: {zip_path} not found after download")

    print(f"{name}: staged at {local_dir}")
    return local_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CoreML Qwen artifacts")
    parser.add_argument(
        "model",
        nargs="?",
        default="qwen3b",
        choices=list(REPOS.keys()),
        help="which model artifact to download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the artifact is already present",
    )
    args = parser.parse_args()

    try:
        download(args.model, force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
