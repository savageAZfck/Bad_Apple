#!/usr/bin/env python3
"""Local model registry for Bad Apple.

Discovers cached MLX models, tracks which is currently loaded, and supports
runtime (well, daemon-session) switching between them.  No cloud after the
initial cache; all metadata is read from local `config.json` / `tokenizer.json`.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import badapple_model_provenance
import badapple_slicks

HUB_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
REGISTRY_FILE = "model_registry.json"

# Recommended local-first models for different memory budgets.
# Download with `huggingface-cli download <id>` or set the MLX loader to pull on first use.
RECOMMENDED_MODELS = [
    {
        "id": "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit",
        "name": "Qwen 3.5 9B 4-bit",
        "size_gb": 5.5,
        "memory_gb": 8,
        "notes": "Default brain. Best balance for M1/M2 16GB.",
    },
    {
        "id": "mlx-community/Qwen3.5-32B-MLX-4bit",
        "name": "Qwen 3.5 32B 4-bit",
        "size_gb": 19,
        "memory_gb": 28,
        "notes": "Stronger reasoning. Set BADAPPLE_MAX_KV_SIZE=1024 or 2048 on 36GB.",
    },
    {
        "id": "mlx-community/DeepSeek-V3-Chat-4bit",
        "name": "DeepSeek V3 Chat 4-bit",
        "size_gb": 41,
        "memory_gb": 64,
        "notes": "MoE frontier model. Requires M3/M4 Max/Ultra with 64GB+.",
    },
    {
        "id": "mlx-community/Qwen3.5-1.5B-MLX-8bit",
        "name": "Qwen 3.5 1.5B 8-bit",
        "size_gb": 1.5,
        "memory_gb": 4,
        "notes": "Tiny fast tier or constrained machines.",
    },
    {
        "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "name": "Qwen 2.5 0.5B 4-bit",
        "size_gb": 0.5,
        "memory_gb": 2,
        "notes": "Best speculative-decoding draft for the 9B brain. Download and run `use draft model mlx-community/Qwen2.5-0.5B-Instruct-4bit`.",
    },
]


def _repo_id_from_dirname(name: str) -> str:
    """Convert `models--foo--bar` back to `foo/bar`."""
    return name.replace("models--", "").replace("--", "/")


class ModelRegistry:
    """Manages the local cache of downloaded model weights."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.registry_path = data_dir / REGISTRY_FILE
        self._state: dict[str, Any] = {
            "current": os.environ.get("BADAPPLE_MAIN_MODEL", ""),
            "models": [],
        }
        self._load()
        self._maybe_rescan()
        self.provenance = badapple_model_provenance.ModelProvenance(data_dir)

    def _load(self) -> None:
        if self.registry_path.is_file():
            try:
                with self.registry_path.open("r", encoding="utf-8") as f:
                    self._state = json.load(f)
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
                print(f"[model_registry] could not load registry: {e}", flush=True)

    def _save(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.registry_path.with_name(f".{self.registry_path.name}.{os.getpid()}.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.registry_path)
        except (TypeError, ValueError, OSError) as e:
            print(f"[model_registry] could not save registry: {e}", flush=True)

    def _maybe_rescan(self) -> None:
        """Re-scan the HF hub cache if the on-disk list is empty or stale."""
        if not self._state.get("models"):
            self.scan()

    def _model_info(self, model_dir: Path) -> dict[str, Any] | None:
        """Inspect a single cached model directory."""
        blobs = model_dir / "blobs"
        if not blobs.is_dir():
            return None
        snapshots = model_dir / "snapshots"
        if not snapshots.is_dir():
            return None
        # Pick the most recent snapshot.
        snap_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
        if not snap_dirs:
            return None
        snap_dir = sorted(snap_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        config_path = snap_dir / "config.json"
        if not config_path.is_file():
            return None
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            config = {}

        size_bytes = sum(f.stat().st_size for f in snap_dir.rglob("*") if f.is_file())
        size_gb = round(size_bytes / (1024 ** 3), 2)

        quant = self._detect_quantization(snap_dir)

        return {
            "id": _repo_id_from_dirname(model_dir.name),
            "path": str(snap_dir),
            "size_gb": size_gb,
            "quantization": quant,
            "context_length": config.get("max_position_embeddings") or config.get("max_seq_len") or "unknown",
            "architecture": config.get("architectures", ["unknown"])[0],
            "vocab_size": config.get("vocab_size", "unknown"),
        }

    def _detect_quantization(self, snap_dir: Path) -> str:
        """Best-effort quantization detection from weight file names."""
        for f in snap_dir.iterdir():
            if f.is_file() and f.suffix in (".safetensors", ".gguf", ".bin"):
                low = f.name.lower()
                if "4bit" in low or "4-bit" in low or "q4" in low:
                    return "4-bit"
                if "8bit" in low or "8-bit" in low or "q8" in low:
                    return "8-bit"
                if "fp16" in low:
                    return "FP16"
                if "fp32" in low:
                    return "FP32"
        return "unknown"

    def scan(self) -> str:
        """Scan the HF hub cache and rebuild the registry."""
        models: list[dict[str, Any]] = []
        if HUB_ROOT.is_dir():
            for model_dir in sorted(HUB_ROOT.iterdir()):
                if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
                    continue
                info = self._model_info(model_dir)
                if info:
                    models.append(info)
        self._state["models"] = models
        self._save()
        current = self._state.get("current", "")
        matching = [m for m in models if m["id"] == current or m["path"] == current]
        if current and not matching:
            # The current model doesn't appear in the cache scan; keep it but note it.
            self._state["current"] = current
        return f"Found {len(models)} local model(s). Current: {current or 'none'}."

    def list_models(self) -> str:
        models = self._state.get("models", [])
        current = self._state.get("current", "").lower()
        lines = ["Local model cache:", ""]
        for m in models:
            marker = " *" if m["id"].lower() == current or m["path"].lower() == current else ""
            lines.append(
                f"{m['id']}{marker} — {m['size_gb']} GB, {m['quantization']}, "
                f"ctx {m['context_length']}, arch {m['architecture']}"
            )
        if not models:
            lines.append("No models found in ~/.cache/huggingface/hub.")
        return "\n".join(lines)

    def current(self) -> str:
        return self._state.get("current", "") or "none"

    def set_current(self, model_id: str) -> bool:
        """Mark a model as the active default."""
        model_id_lower = model_id.lower()
        models = self._state.get("models", [])
        match = next((m for m in models if m["id"].lower() == model_id_lower or m["path"].lower() == model_id_lower), None)
        if not match:
            # Allow raw paths and env-style IDs too.
            if Path(model_id).is_dir():
                self._state["current"] = model_id
                self._save()
                return True
            return False
        self._state["current"] = match["id"]
        self._save()
        return True

    def info(self, model_id: str) -> str:
        model_id_lower = model_id.lower()
        for m in self._state.get("models", []):
            if m["id"].lower() == model_id_lower or m["path"].lower() == model_id_lower:
                return json.dumps(m, indent=2, default=str)
        return f"Model {model_id} not found in local cache."

    def recommend(self) -> str:
        """List recommended models and whether they are cached."""
        local = {m["id"].lower() for m in self._state.get("models", [])}
        lines = ["Recommended models:", ""]
        for m in RECOMMENDED_MODELS:
            cached = " (cached)" if m["id"].lower() in local else ""
            lines.append(
                f"{m['id']}{cached} — {m['name']}, ~{m['size_gb']} GB, "
                f"needs ~{m['memory_gb']} GB RAM\n    {m['notes']}"
            )
        lines.append("\nUse `use model <id>` after downloading with huggingface-cli or mlx_lm.load.")
        return "\n".join(lines)

    def verify(self, model_id: str | None = None) -> dict[str, Any]:
        """Verify one or all models against their stored provenance manifests."""
        if model_id:
            return self._verify_one(str(model_id))
        results = {}
        for m in self._state.get("models", []):
            results[m["id"]] = self._verify_one(m["id"])
        ok = all(r.get("status") == "verified" for r in results.values())
        return {"status": "verified" if ok else "mismatch", "models": results}

    def _verify_one(self, model_id: str) -> dict[str, Any]:
        model_id_lower = model_id.lower()
        match = next((m for m in self._state.get("models", []) if m["id"].lower() == model_id_lower), None)
        if not match:
            return {"status": "unknown", "error": f"model {model_id!r} not in registry"}
        return self.provenance.verify(match["id"], match["path"])

    def add_model(self, path: str, model_id: str | None = None) -> dict[str, Any]:
        """Import a local model directory into the registry and record provenance.

        The model directory must contain a config.json and at least one weight file.
        If model_id is not provided, a stable id is derived from the directory name.
        """
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return {"status": "error", "error": f"not a directory: {path}"}
        config_path = root / "config.json"
        if not config_path.is_file():
            # HF cache snapshot layout: models--<org>--<name>/snapshots/<ref>/
            for candidate in [root / "snapshots", root]:
                if not candidate.is_dir():
                    continue
                for child in candidate.iterdir():
                    if child.is_dir() and (child / "config.json").is_file():
                        root = child
                        config_path = child / "config.json"
                        break
                if config_path.is_file():
                    break
            if not config_path.is_file():
                return {"status": "error", "error": "no config.json found in model directory"}

        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
            return {"status": "error", "error": f"cannot read config.json: {e}"}

        size_bytes = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        size_gb = round(size_bytes / (1024 ** 3), 2)
        quant = self._detect_quantization(root)

        # Determine the model id. For HF cache directories, try to recover the repo id.
        if model_id:
            derived_id = model_id
        else:
            # Walk up to find a HF hub repo dir (models--org--name).
            repo_dir = None
            for parent in [root, *root.parents]:
                if parent.name.startswith("models--"):
                    repo_dir = parent
                    break
            if repo_dir is not None:
                derived_id = _repo_id_from_dirname(repo_dir.name)
            else:
                derived_id = _repo_id_from_dirname(root.name)
            if derived_id == root.name or (repo_dir is not None and derived_id == repo_dir.name):
                # Fallback to a safe id based on the parent dir name and a short hash.
                parent = root.parent.name
                digest = hashlib.sha256(str(root).encode()).hexdigest()[:8]
                derived_id = f"{parent}-{digest}"

        info = {
            "id": derived_id,
            "path": str(root),
            "size_gb": size_gb,
            "quantization": quant,
            "context_length": config.get("max_position_embeddings") or config.get("max_seq_len") or "unknown",
            "architecture": config.get("architectures", ["unknown"])[0],
            "vocab_size": config.get("vocab_size", "unknown"),
            "provenance": "pending",
        }

        # Record or update provenance.
        try:
            record = self.provenance.record(derived_id, derived_id, str(root))
            if record.get("status") == "recorded":
                info["provenance"] = "recorded"
            else:
                return {"status": "error", "error": f"provenance recording failed: {record}"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": f"provenance recording error: {e}"}

        # Sign the provenance manifest with the Secure Enclave if available.
        try:
            manifest_path = self.provenance._manifest_path(derived_id)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_json = json.dumps(manifest, sort_keys=True, ensure_ascii=True).encode("utf-8")
            signature = badapple_slicks.v2_sign_message(manifest_json)
            if signature:
                manifest["signature"] = signature
                manifest["public_key"] = badapple_slicks.v2_public_key_b64()
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
                info["signature"] = "se-v2"
        except Exception as e:  # noqa: BLE001
            # SE signing is best-effort; the model can still be used without it.
            info["signature_error"] = str(e)

        # Add or replace the entry in the registry.
        models = self._state.get("models", [])
        models = [m for m in models if m["id"].lower() != derived_id.lower()]
        models.append(info)
        self._state["models"] = models
        self._save()
        return {"status": "added", "model": info}

    def remove_model(self, model_id: str) -> dict[str, Any]:
        """Remove a model from the registry and delete its provenance manifest.

        The model weights themselves are not deleted to avoid accidental data loss.
        """
        model_id_lower = model_id.lower()
        models = [m for m in self._state.get("models", []) if m["id"].lower() != model_id_lower]
        removed = len(self._state.get("models", [])) - len(models)
        if not removed:
            return {"status": "error", "error": f"model {model_id!r} not in registry"}
        self._state["models"] = models
        try:
            manifest_path = self.provenance._manifest_path(model_id)
            if manifest_path.is_file():
                manifest_path.unlink()
        except OSError:
            pass
        if self._state.get("current", "").lower() == model_id_lower:
            self._state["current"] = ""
        self._save()
        return {"status": "removed", "model_id": model_id}
