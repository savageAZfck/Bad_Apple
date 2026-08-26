#!/usr/bin/env python3
"""Rebuild the LM head as a single contiguous shard and compile it for CPU+ANE."""
import json
import shutil
import sys
import time
from pathlib import Path

# Make the conversion utilities available without executing __main__
repo = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo / "tests" / "ane_brain_perf"))

import convert_ane_coreml as cvt


def main():
    manifest_path = Path(
        "tests/ane_brain_perf/artifacts/qwen3b_ane_shards/conversion_manifest.json"
    ).resolve()
    output_dir = manifest_path.parent
    work_dir = output_dir / "work"
    compiled_dir = output_dir / "compiled"
    logs_dir = output_dir / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    compiled_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    cfg = cvt.GGUFModel(manifest["source"]["path"]).config()
    vocab_size = cfg["vocab_size"]
    d = cfg["d_model"]

    # Backup and remove any existing multi-shard LM head directories
    backup_dir = compiled_dir / "lm_head_4shard_backup"
    backup_dir.mkdir(exist_ok=True)
    for compiled in sorted(compiled_dir.glob("lm_head_*.mlmodelc")):
        if compiled.is_dir() and not compiled.name.startswith("lm_head_00_000000-"):
            dest = backup_dir / compiled.name
            if dest.exists():
                shutil.rmtree(dest)
            compiled.rename(dest)

    # Wipe old work packages that would collide with the new single shard
    for pkg in work_dir.glob("lm_head_*.mlpackage"):
        if pkg.is_dir():
            shutil.rmtree(pkg)

    output_name = "lm_head_00_000000-151936_q8_attempt1"
    package_path = work_dir / f"{output_name}.mlpackage"
    compiled_path = compiled_dir / f"{output_name}.mlmodelc"
    log_path = logs_dir / "lm_head_00_000000-151936_q8.log"

    print("Building single LM head .mlpackage for the full vocabulary...")
    package = Path(
        cvt.build_lm_head_shard(
            manifest["source"]["path"],
            0,
            vocab_size,
            work_dir,
            output_name,
            quant_bits=8,
            compute_units="all",
        )
    )
    print(f"Saved package: {package}")

    print("Compiling for MLComputeUnits.all (this may take several minutes)...")
    status, returncode = cvt._compile_package(
        package,
        compiled_path,
        "all",
        1800,
        log_path,
    )
    if status != "compiled":
        print(f"Compile failed: {status} (exit {returncode})")
        print(log_path.read_text())
        sys.exit(1)

    print(f"Compiled: {compiled_path}")

    manifest["shared"]["lm_head_shards"] = [{
        "name": "lm_head_00_000000-151936_q8",
        "vocab_start": 0,
        "vocab_end": vocab_size,
        "hidden_size": d,
        "quant_bits": 8,
        "status": "compiled",
        "attempts": 1,
        "package_path": str(package),
        "compiled_path": str(compiled_path),
        "compiled_size_bytes": cvt._tree_size(compiled_path),
        "sha256": cvt._tree_sha256(compiled_path),
        "log_path": str(log_path),
        "last_error": None,
    }]
    manifest["model"]["lm_head_shard_count"] = 1
    manifest["model"]["compute_units"] = "all"
    manifest["updated_at_unix"] = time.time()
    cvt._atomic_write_json(manifest_path, manifest)
    print("Manifest updated with single LM head.")


if __name__ == "__main__":
    main()
