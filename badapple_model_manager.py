#!/usr/bin/env python3
"""Background model download and status manager for Bad Apple.

Tracks the 9B main brain, fast 0.5B tier, vision VLM, and FLUX image model.
Downloads happen in background threads so the daemon and dashboard stay
responsive. Loading still happens in the main MLX executor on demand.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

import badapple_model_provenance



@dataclass
class ModelProfile:
    """Static description of a model the user can unlock."""

    id: str
    name: str
    repo_id: str
    size_gb: float
    kind: str  # "text", "vision", "image", "draft"
    loaded_in: str = ""  # "mlx_server", "vision_host", "mflux"


@dataclass
class ModelState:
    """Runtime state of a model download/load."""

    status: str = "missing"  # missing, queued, downloading, cached, loaded, error
    progress: float = 0.0  # 0.0 - 1.0, best-effort
    error: str = ""
    local_path: str = ""
    downloaded_bytes: int = 0
    total_bytes: int = 0
    last_updated: float = field(default_factory=time.time)
    verified: bool = False


_KNOWN_MODELS: list[ModelProfile] = [
    ModelProfile(
        id="fast_0.5b",
        name="Fast 0.5B",
        repo_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        size_gb=0.35,
        kind="text",
        loaded_in="mlx_server",
    ),
    ModelProfile(
        id="main_9b",
        name="Deep 9B",
        repo_id="caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit",
        size_gb=4.8,
        kind="text",
        loaded_in="mlx_server",
    ),
    ModelProfile(
        id="main_32b",
        name="Deep 32B",
        repo_id="mlx-community/Qwen3.5-32B-MLX-4bit",
        size_gb=19.0,
        kind="text",
        loaded_in="mlx_server",
    ),
    ModelProfile(
        id="main_70b",
        name="Deep 70B MoE",
        repo_id="mlx-community/DeepSeek-V3-Chat-4bit",
        size_gb=41.0,
        kind="text",
        loaded_in="mlx_server",
    ),
    ModelProfile(
        id="vision_2b",
        name="Ocular Vision",
        repo_id="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        size_gb=1.4,
        kind="vision",
        loaded_in="vision_host",
    ),
    ModelProfile(
        id="flux_4b",
        name="FLUX.2-klein 4B",
        repo_id="mflux/flux2-klein-4b",  # conceptual; mflux manages its own cache
        size_gb=4.0,
        kind="image",
        loaded_in="mflux",
    ),
]


class ModelManager:
    """Download, cache, and load status for the optional Bad Apple models."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir or os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple")).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._status_file = self.data_dir / "model_manager.json"
        self._profiles: dict[str, ModelProfile] = {p.id: p for p in _KNOWN_MODELS}
        self._state: dict[str, ModelState] = {p.id: ModelState() for p in _KNOWN_MODELS}
        self._lock = threading.Lock()
        self._download_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="badapple_model_dl")
        self._allow_downloads = os.environ.get("BADAPPLE_ALLOW_DOWNLOADS", "0") == "1"
        self._online_override = os.environ.get("BADAPPLE_ONLINE_MODELS", "0") == "1"
        self._download_futures: dict[str, Any] = {}
        self._provenance = badapple_model_provenance.ModelProvenance(self.data_dir)
        self._verify_hashes = os.environ.get("BADAPPLE_VERIFY_MODEL_HASHES", "1") != "0"

        # Restore persisted state (status only; we re-verify cache on demand).
        self._load_state()

    @property
    def allow_downloads(self) -> bool:
        return self._allow_downloads or self._online_override

    def set_allow_downloads(self, allowed: bool) -> None:
        with self._lock:
            self._allow_downloads = allowed
            self._online_override = allowed

    def list_profiles(self) -> list[ModelProfile]:
        return list(self._profiles.values())

    def status(self, model_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if model_id is not None:
                return self._status_for(model_id)
            return {mid: self._status_for(mid) for mid in self._profiles}

    def _status_for(self, model_id: str) -> dict[str, Any]:
        profile = self._profiles.get(model_id)
        if not profile:
            return {"error": f"unknown model {model_id}"}
        state = self._state[model_id]
        provenance = {"status": "unknown"}
        if state.local_path:
            provenance = self._provenance.verify(model_id, state.local_path)
        return {
            "id": profile.id,
            "name": profile.name,
            "repo_id": profile.repo_id,
            "kind": profile.kind,
            "size_gb": profile.size_gb,
            "status": state.status,
            "progress": round(state.progress, 3),
            "error": state.error,
            "local_path": state.local_path,
            "allow_downloads": self.allow_downloads,
            "loaded_in": profile.loaded_in,
            "verified": state.verified,
            "provenance": provenance,
        }

    def _load_state(self) -> None:
        import json

        if not self._status_file.is_file():
            return
        try:
            with open(self._status_file) as f:
                data = json.load(f)
            for mid, s in data.get("state", {}).items():
                if mid in self._state:
                    self._state[mid] = ModelState(**s)
        except (json.JSONDecodeError, TypeError, ValueError, OSError) as e:  # noqa: BLE001
            print(f"[model_manager] could not load persisted state: {e}", flush=True)

    def _save_state(self) -> None:
        import json

        if not self.data_dir.is_dir():
            return
        try:
            with self._lock:
                payload = {
                    "state": {
                        mid: {
                            "status": s.status,
                            "progress": s.progress,
                            "error": s.error,
                            "local_path": s.local_path,
                            "downloaded_bytes": s.downloaded_bytes,
                            "total_bytes": s.total_bytes,
                            "last_updated": s.last_updated,
                            "verified": s.verified,
                        }
                        for mid, s in self._state.items()
                    }
                }
            with open(self._status_file, "w") as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            print(f"[model_manager] could not save state: {e}", flush=True)

    def _update_state(self, model_id: str, **kwargs: Any) -> None:
        with self._lock:
            state = self._state[model_id]
            for k, v in kwargs.items():
                setattr(state, k, v)
            state.last_updated = time.time()
        self._save_state()

    def _with_downloads_enabled(self, fn: Callable[[], Any]) -> Any:
        """Temporarily allow network access for this thread.

        The global plist sets HF_HUB_OFFLINE=1 for air-gap certification. A
        user must opt in with BADAPPLE_ALLOW_DOWNLOADS=1 before we ever touch
        the network.
        """
        if not self.allow_downloads:
            raise RuntimeError("Downloads are disabled. Set BADAPPLE_ALLOW_DOWNLOADS=1 or enable in the dashboard.")

        # Backup and override the offline flag only for this thread's lifetime.
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        try:
            os.environ["HF_HUB_OFFLINE"] = "0"
            return fn()
        finally:
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline

    def _resolve_cache_path(self, repo_id: str, allow_download: bool = False) -> str | None:
        try:
            from huggingface_hub import snapshot_download

            # Essential model artifacts only; incomplete snapshots with missing
            # READMEs or .gitattributes should not block local use.
            allow_patterns = [
                "*.safetensors",
                "*.bin",
                "*.json",
                "*.py",
                "*.txt",
                "*.model",
                "tokenizer.*",
                "merges.*",
                "vocab.*",
                "config.*",
                "model.*",
                "preprocessor_config.*",
                "chat_template.*",
            ]

            def _download() -> str:
                return snapshot_download(
                    repo_id,
                    local_files_only=not allow_download,
                    allow_patterns=allow_patterns,
                )

            if allow_download:
                return self._with_downloads_enabled(_download)
            return _download()
        except Exception:  # noqa: BLE001
            if not allow_download:
                return None
            raise

    def _check_cached(self, repo_id: str) -> bool:
        return self._resolve_cache_path(repo_id, allow_download=False) is not None

    def _download_worker(self, model_id: str) -> None:
        profile = self._profiles.get(model_id)
        if not profile:
            return
        self._update_state(model_id, status="downloading", progress=0.0, error="")
        try:
            path = self._resolve_cache_path(profile.repo_id, allow_download=True)
            if path:
                try:
                    result = self._provenance.record(model_id, profile.repo_id, path)
                    if result.get("status") != "recorded":
                        print(f"[model_manager] provenance record for {model_id}: {result}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[model_manager] provenance recording failed for {model_id}: {e}", flush=True)
            verified = False
            if path and self._verify_hashes:
                verified = self._provenance.verify(model_id, path).get("status") == "verified"
            self._update_state(model_id, status="cached", progress=1.0, local_path=path or "", error="", verified=verified)
        except Exception as e:  # noqa: BLE001
            self._update_state(model_id, status="error", error=str(e), progress=0.0)
            print(f"[model_manager] download failed for {model_id}: {e}", flush=True)
        finally:
            with self._lock:
                self._download_futures.pop(model_id, None)

    def start_download(self, model_id: str) -> dict[str, Any]:
        """Queue a background download for the given model."""
        if not self.allow_downloads:
            raise RuntimeError("Downloads are disabled. Set BADAPPLE_ALLOW_DOWNLOADS=1 or enable in the dashboard.")
        profile = self._profiles.get(model_id)
        if not profile:
            return {"error": f"unknown model {model_id}"}

        with self._lock:
            if self._state[model_id].status in ("downloading", "queued"):
                return self._status_for(model_id)
            existing = self._download_futures.get(model_id)
            if existing and not existing.done():
                return self._status_for(model_id)

        self._update_state(model_id, status="queued", progress=0.0, error="")
        future = self._download_executor.submit(self._download_worker, model_id)
        with self._lock:
            self._download_futures[model_id] = future
        return self._status_for(model_id)

    def ensure_cached(self, model_id: str, download: bool = True) -> dict[str, Any]:
        """Return cached status; start a download if missing and allowed."""
        state = self._state.get(model_id)
        if not state:
            return {"error": f"unknown model {model_id}"}
        if state.status in ("cached", "loaded"):
            return self._status_for(model_id)
        if state.status == "downloading":
            return self._status_for(model_id)
        if download and self.allow_downloads:
            return self.start_download(model_id)
        return self._status_for(model_id)

    def wait_for_download(self, model_id: str, timeout: float | None = None) -> bool:
        """Block until the model is cached or the timeout expires.

        Returns True if the model reached cached/loaded status, False on timeout.
        """
        with self._lock:
            future = self._download_futures.get(model_id)
        if not future:
            return self._state[model_id].status in ("cached", "loaded")
        try:
            future.result(timeout=timeout)
            return self._state[model_id].status in ("cached", "loaded")
        except TimeoutError:
            return False

    def mark_loaded(self, model_id: str, local_path: str | None = None) -> None:
        """Mark a model as loaded in RAM after verifying provenance.

        A missing manifest is recorded in the background so the next load is fast.
        A mismatch is treated as fatal; a missing local path is allowed.
        """
        if model_id not in self._state:
            return
        if local_path is None:
            profile = self._profiles.get(model_id)
            if profile:
                local_path = self._resolve_cache_path(profile.repo_id, allow_download=False) or ""

        verified = False
        root = Path(local_path).expanduser().resolve() if local_path else None
        if root and self._verify_hashes:
            result = self._provenance.verify(model_id, str(root))
            status = result.get("status")
            if status == "verified":
                verified = True
            elif status == "mismatch":
                print(f"[model_manager] load rejected for {model_id}: {result.get('error')}", flush=True)
                self._update_state(model_id, status="error", progress=1.0, local_path=local_path, error=result.get("error", "provenance verification failed"), verified=False)
                return
            elif status == "unknown" and root.is_dir():
                # Record in background so future loads are fast.
                profile = self._profiles.get(model_id)
                if profile:
                    self._download_executor.submit(
                        self._record_provenance_worker, model_id, profile.repo_id, str(root)
                    )

        self._update_state(model_id, status="loaded", progress=1.0, local_path=local_path or "", error="", verified=verified)

    def mark_unloaded(self, model_id: str, local_path: str | None = None) -> None:
        """Mark a model as cached but not resident."""
        if model_id not in self._state:
            return
        if local_path is None:
            profile = self._profiles.get(model_id)
            if profile:
                local_path = self._resolve_cache_path(profile.repo_id, allow_download=False) or ""
        self._update_state(model_id, status="cached", progress=1.0, local_path=local_path or "", error="", verified=False)

    def refresh_cache_status(self, model_id: str) -> dict[str, Any]:
        """Check the local HF cache for a model without downloading.

        Do not overwrite a model that is currently loaded in RAM; a load in
        progress takes precedence over a cache scan.
        """
        profile = self._profiles.get(model_id)
        if not profile:
            return {"error": f"unknown model {model_id}"}
        if self._state[model_id].status in ("downloading", "queued", "loaded"):
            return self._status_for(model_id)
        if self._check_cached(profile.repo_id):
            path = self._resolve_cache_path(profile.repo_id, allow_download=False) or ""
            self._update_state(model_id, status="cached", progress=1.0, local_path=path, error="")
        else:
            self._update_state(model_id, status="missing", progress=0.0, local_path="", error="")
        return self._status_for(model_id)

    def background_refresh_all(self) -> None:
        """Queue cache checks for every known model without blocking the caller."""
        for model_id in self._profiles:
            self._download_executor.submit(self.refresh_cache_status, model_id)

    def memory_required(self, model_id: str) -> float:
        profile = self._profiles.get(model_id)
        return profile.size_gb * 1.4 if profile else 0.0

    def recommend_for_memory(self) -> dict[str, Any]:
        """Recommend the best model that currently fits in RAM."""
        import badapple_vram_governor as vg

        available_gb = vg._available_gb()
        pick = vg.recommend_for_memory(available_gb)
        return {
            "available_gb": round(available_gb, 2),
            "recommended_id": pick,
            "recommended": self._status_for(pick),
        }

    def recommend_for_query(self, query: str) -> dict[str, Any]:
        """Recommend a model for a specific query and memory budget."""
        import badapple_vram_governor as vg

        available_gb = vg._available_gb()
        pick = vg.recommend_model_for_query(query, available_gb)
        return {
            "available_gb": round(available_gb, 2),
            "recommended_id": pick,
            "recommended": self._status_for(pick),
        }

    def preload_priority(self) -> list[str]:
        import badapple_vram_governor as vg

        return vg.model_preload_priority()

    def record_provenance(self, model_id: str, local_path: str | None = None) -> dict[str, Any]:
        """Record the manifest for a cached model."""
        profile = self._profiles.get(model_id)
        if not profile:
            return {"error": f"unknown model {model_id}"}
        if local_path is None:
            local_path = self._resolve_cache_path(profile.repo_id, allow_download=False) or ""
        if not local_path:
            return {"error": f"model {model_id} is not cached"}
        return self._provenance.record(model_id, profile.repo_id, local_path)

    def verify_provenance(self, model_id: str, local_path: str | None = None) -> dict[str, Any]:
        """Verify the manifest for a cached model."""
        profile = self._profiles.get(model_id)
        if not profile:
            return {"error": f"unknown model {model_id}"}
        if local_path is None:
            local_path = self._resolve_cache_path(profile.repo_id, allow_download=False) or ""
        if not local_path:
            return {"status": "missing", "error": f"model {model_id} is not cached"}
        result = self._provenance.verify(model_id, local_path)
        with self._lock:
            if model_id in self._state:
                self._state[model_id].verified = result.get("status") == "verified"
        return result

    def verify_before_load(self, model_id: str) -> dict[str, Any]:
        """Fast provenance check before loading. Records in the background if no manifest exists."""
        if not self._verify_hashes:
            return {"status": "skipped"}
        profile = self._profiles.get(model_id)
        if not profile:
            return {"error": f"unknown model {model_id}"}
        local_path = self._resolve_cache_path(profile.repo_id, allow_download=False) or ""
        if not local_path:
            return {"status": "missing", "error": f"model {model_id} is not cached"}

        result = self._provenance.verify(model_id, local_path)
        if result.get("status") == "verified":
            with self._lock:
                self._state[model_id].verified = True
            return result
        if result.get("status") == "mismatch":
            with self._lock:
                self._state[model_id].verified = False
            return result
        if result.get("status") == "unknown":
            # No manifest yet; record it in the background so future loads are fast.
            self._download_executor.submit(self._record_provenance_worker, model_id, profile.repo_id, local_path)
            with self._lock:
                self._state[model_id].verified = False
            return {"status": "recording", "error": "no manifest; recording in background"}
        return result

    def _record_provenance_worker(self, model_id: str, repo_id: str, local_path: str) -> None:
        try:
            self._provenance.record(model_id, repo_id, local_path)
            result = self._provenance.verify(model_id, local_path)
            if result.get("status") == "verified":
                with self._lock:
                    self._state[model_id].verified = True
        except Exception as e:  # noqa: BLE001
            print(f"[model_manager] background provenance recording failed for {model_id}: {e}", flush=True)

    def shutdown(self) -> None:
        self._download_executor.shutdown(wait=False, cancel_futures=True)
