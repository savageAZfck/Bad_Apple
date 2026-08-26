#!/usr/bin/env python3
"""Delete stale compiled layer shards and reset the manifest so the pipeline rebuilds them."""
import json
import shutil
from pathlib import Path

base = Path("/Users/savag3/bad_apple/tests/ane_brain_perf/artifacts/qwen3b_ane_shards")
compiled = base / "compiled"
manifest_path = base / "conversion_manifest.json"

# Delete old per-layer compiled mlmodelc directories
for item in compiled.glob("qwen3b_s*_q8_attempt1.mlmodelc"):
    if item.is_dir():
        print(f"Removing stale layer compiled: {item}")
        shutil.rmtree(item)

# Delete old LM-head backups
for item in compiled.glob("lm_head_*backup*"):
    if item.is_dir():
        print(f"Removing stale LM head backup: {item}")
        shutil.rmtree(item)

# Reset layer shard entries in the manifest
manifest = json.loads(manifest_path.read_text())
for shard in manifest.get("shards", []):
    shard["status"] = "pending"
    shard["attempts"] = 0
    shard["package_path"] = None
    shard["compiled_path"] = None
    shard["compiled_size_bytes"] = 0
    shard["sha256"] = None
    shard["last_error"] = None
manifest["status"] = "pending"
manifest["updated_at_unix"] = None

import time

manifest["updated_at_unix"] = time.time()
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print("Manifest reset; 36 layer shards will be rebuilt on next pipeline run.")
