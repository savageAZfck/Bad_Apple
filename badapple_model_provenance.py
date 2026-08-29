#!/usr/bin/env python3
"""Model provenance and hash verification for Bad Apple.

Records SHA-256 fingerprints of model artifacts after download and verifies
them before load. The manifest is stored under the Bad Apple data directory
and can be re-computed with `badapple verify models`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_HASH_SIZE_CAP = 10 * 1024 * 1024 * 1024  # 10 GB total per model


@dataclass
class FileEntry:
    """Integrity record for a single file in a model cache."""

    relative: str
    size: int
    mtime: float
    sha256: str


@dataclass
class ModelManifest:
    """Stored manifest for a model cache snapshot."""

    repo_id: str
    local_path: str
    recorded_at: float
    files: dict[str, FileEntry] = field(default_factory=dict)
    signature: str | None = None
    public_key: str | None = None


def _sha256_file(path: Path, total_cap: int = DEFAULT_HASH_SIZE_CAP) -> str:
    """Return the SHA-256 hex digest of a file, with a read-size cap."""
    h = hashlib.sha256()
    size = path.stat().st_size
    if size > total_cap:
        raise ValueError(f"file {path} exceeds single-file hash cap: {size} bytes")
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class ModelProvenance:
    """Compute, persist, and verify model cache integrity hashes."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(
            data_dir or os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple")
        ).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_dir = self.data_dir / "model_manifests"
        self._manifest_dir.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, model_id: str) -> Path:
        # Reject path traversal; allow repo-style IDs with a slash by encoding it.
        safe = model_id.replace("/", "--").replace("..", "")
        if not safe or safe.startswith("/") or "/" in safe:
            raise ValueError(f"invalid model_id for manifest: {model_id!r}")
        return self._manifest_dir / f"{safe}.json"

    def _list_files(self, local_path: Path) -> list[Path]:
        """Return non-hidden files under local_path.

        We keep the path as it appears under local_path (so symlinks in HF
        snapshot dirs are indexed by their snapshot-relative name) but open
        them normally, which follows the link to the blob for hashing.
        """
        if not local_path.is_dir():
            return []
        files: list[Path] = []
        for p in local_path.rglob("*"):
            if not p.is_file():
                continue
            name = p.name
            if name.startswith(".") or name == "desktop.ini" or name == "Thumbs.db":
                continue
            try:
                # Verify the symlink target exists and is a regular file.
                resolved = p.resolve()
                if resolved.is_file():
                    files.append(p)
            except (OSError, ValueError):
                continue
        return sorted(files)

    def record(self, model_id: str, repo_id: str, local_path: str) -> dict[str, Any]:
        """Compute and save a manifest for the model cache at local_path."""
        root = Path(local_path).expanduser().resolve()
        if not root.is_dir():
            return {"status": "missing", "error": f"not a directory: {local_path}"}

        files: dict[str, FileEntry] = {}
        total_hashed = 0
        for f in self._list_files(root):
            rel = str(f.relative_to(root))
            st = f.stat()
            if total_hashed + st.st_size > DEFAULT_HASH_SIZE_CAP:
                raise ValueError(f"model at {root} exceeds total hash cap")
            try:
                digest = _sha256_file(f)
            except (OSError, ValueError) as e:
                return {"status": "error", "error": f"cannot hash {rel}: {e}"}
            files[rel] = FileEntry(
                relative=rel,
                size=st.st_size,
                mtime=st.st_mtime,
                sha256=digest,
            )
            total_hashed += st.st_size

        manifest = ModelManifest(
            repo_id=repo_id,
            local_path=str(root),
            recorded_at=time.time(),
            files=files,
        )
        self._save_manifest(model_id, manifest)
        return {
            "status": "recorded",
            "files": len(files),
            "total_bytes": total_hashed,
        }

    def verify(self, model_id: str, local_path: str | None = None) -> dict[str, Any]:
        """Verify the model cache against the stored manifest.

        Returns one of: unknown, verified, mismatch, missing, corrupt_manifest.
        """
        manifest_path = self._manifest_path(model_id)
        if not manifest_path.is_file():
            return {"status": "unknown", "error": "no recorded manifest"}

        try:
            manifest = self._load_manifest(manifest_path)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            return {"status": "corrupt_manifest", "error": str(e)}

        if local_path:
            root = Path(local_path).expanduser().resolve()
        else:
            root = Path(manifest.local_path).expanduser().resolve()

        if not root.is_dir():
            return {"status": "missing", "error": f"local path not found: {root}"}

        mismatches: list[str] = []
        checked = 0
        for rel, entry in manifest.files.items():
            f = (root / rel).resolve()
            if not f.is_file():
                mismatches.append(f"missing: {rel}")
                continue
            st = f.stat()
            if st.st_size != entry.size:
                mismatches.append(f"size changed: {rel}")
                continue
            # Fast path: if size and mtime are unchanged, trust the stored hash.
            if abs(st.st_mtime - entry.mtime) < 1e-9:
                checked += 1
                continue
            try:
                digest = _sha256_file(f)
            except (OSError, ValueError) as e:
                mismatches.append(f"cannot rehash {rel}: {e}")
                continue
            if digest != entry.sha256:
                mismatches.append(f"hash mismatch: {rel}")
                continue
            checked += 1

        # Also detect unexpected new files at the top level? No — HF cache
        # may legitimately gain additional files. We only verify the recorded set.

        if mismatches:
            return {
                "status": "mismatch",
                "error": "; ".join(mismatches[:10]),
                "mismatches": mismatches,
                "checked": checked,
            }

        return {
            "status": "verified",
            "files": len(manifest.files),
            "checked": checked,
            "recorded_at": manifest.recorded_at,
        }

    def _load_manifest(self, path: Path) -> ModelManifest:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        raw_files = raw.pop("files", {})
        files = {rel: FileEntry(**entry) for rel, entry in raw_files.items()}
        return ModelManifest(files=files, **raw)

    def _save_manifest(self, model_id: str, manifest: ModelManifest) -> None:
        path = self._manifest_path(model_id)
        tmp = path.with_suffix(".tmp")
        payload = {
            "repo_id": manifest.repo_id,
            "local_path": manifest.local_path,
            "recorded_at": manifest.recorded_at,
            "files": {rel: asdict(entry) for rel, entry in manifest.files.items()},
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
