#!/usr/bin/env python3
"""Checksummed state generations, artifact provenance, and encrypted portability."""

import base64
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"BADAPPLE-VAULT-1\0"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


class ArtifactManifest:
    SCHEMA_VERSION = 1

    @classmethod
    def create(cls, artifacts: Iterable[Path], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        records = []
        for artifact in artifacts:
            path = Path(artifact).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append({"path": str(path), "size": path.stat().st_size, "sha256": _hash_file(path)})
        return {
            "schema_version": cls.SCHEMA_VERSION,
            "created_at": time.time(),
            "metadata": metadata or {},
            "artifacts": records,
        }

    @staticmethod
    def signing_payload(manifest: dict[str, Any]) -> bytes:
        unsigned = dict(manifest)
        unsigned.pop("seal", None)
        return json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def seal(cls, manifest: dict[str, Any], public_key: str, signature: str) -> dict[str, Any]:
        sealed = dict(manifest)
        sealed["seal"] = {"algorithm": "secure-enclave-p256-sha256", "public_key": public_key, "signature": signature}
        return sealed

    @classmethod
    def verify(cls, manifest: dict[str, Any]) -> dict[str, Any]:
        results = []
        for record in manifest.get("artifacts", []):
            path = Path(record.get("path", ""))
            exists = path.is_file()
            actual = _hash_file(path) if exists else None
            results.append({
                "path": str(path),
                "exists": exists,
                "valid": exists and actual == record.get("sha256") and path.stat().st_size == record.get("size"),
                "expected_sha256": record.get("sha256"),
                "actual_sha256": actual,
            })
        signature_valid: bool | None = None
        seal = manifest.get("seal")
        if isinstance(seal, dict):
            try:
                key = ec.EllipticCurvePublicKey.from_encoded_point(
                    ec.SECP256R1(),
                    base64.b64decode(seal["public_key"], validate=True),
                )
                key.verify(
                    base64.b64decode(seal["signature"], validate=True),
                    cls.signing_payload(manifest),
                    ec.ECDSA(hashes.SHA256()),
                )
                signature_valid = True
            except (LookupError, TypeError, ValueError):
                signature_valid = False
        artifacts_valid = bool(results) and all(item["valid"] for item in results)
        return {
            "valid": artifacts_valid and signature_valid is not False,
            "artifacts_valid": artifacts_valid,
            "signature_valid": signature_valid,
            "artifacts": results,
        }


class GenerationStore:
    """Commits complete, checksummed state families and restores verified generations."""

    SCHEMA_VERSION = 1

    def __init__(self, data_dir: Path, keep: int = 5):
        self.root = Path(data_dir) / "generations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep = max(2, keep)
        self.current_path = self.root / "current.json"

    def commit(self, name: str, sources: dict[str, Path], metadata: dict[str, Any] | None = None) -> str:
        generation_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        staging = self.root / f".{generation_id}.staging"
        final = self.root / generation_id
        files_dir = staging / "files"
        files_dir.mkdir(parents=True)
        records = []
        try:
            for logical_name, source in sorted(sources.items()):
                source_path = Path(source).expanduser().resolve()
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                target = files_dir / logical_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)
                records.append({
                    "name": logical_name,
                    "source": str(source_path),
                    "size": target.stat().st_size,
                    "sha256": _hash_file(target),
                })
            previous = self.current().get("generation_id") if self.current() else None
            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "generation_id": generation_id,
                "name": name,
                "previous_generation": previous,
                "created_at": time.time(),
                "metadata": metadata or {},
                "files": records,
            }
            _atomic_json(staging / "manifest.json", manifest)
            marker = staging / "COMMITTED"
            marker.write_text(manifest["generation_id"] + "\n", encoding="utf-8")
            with marker.open("r+") as f:
                f.flush()
                os.fsync(f.fileno())
            _fsync_dir(files_dir)
            _fsync_dir(staging)
            os.replace(staging, final)
            _fsync_dir(self.root)
            _atomic_json(self.current_path, {"generation_id": generation_id})
            self._prune()
            return generation_id
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def current(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.current_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def verify(self, generation_id: str) -> dict[str, Any]:
        generation = self.root / generation_id
        try:
            manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
            marker = (generation / "COMMITTED").read_text(encoding="utf-8").strip()
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            return {"valid": False, "generation_id": generation_id, "error": str(e)}
        results = []
        for record in manifest.get("files", []):
            path = generation / "files" / record["name"]
            valid = path.is_file() and path.stat().st_size == record["size"] and _hash_file(path) == record["sha256"]
            results.append({"name": record["name"], "valid": valid})
        valid = marker == generation_id and manifest.get("generation_id") == generation_id and bool(results) and all(r["valid"] for r in results)
        return {"valid": valid, "generation_id": generation_id, "manifest": manifest, "files": results}

    def latest_valid(self) -> str | None:
        candidates = sorted(
            (p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")),
            reverse=True,
        )
        for generation_id in candidates:
            if self.verify(generation_id).get("valid"):
                return generation_id
        return None

    def restore(self, generation_id: str, destination: Path) -> dict[str, Any]:
        verification = self.verify(generation_id)
        if not verification.get("valid"):
            raise ValueError(f"generation {generation_id} failed verification")
        destination = Path(destination).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        restored = []
        for record in verification["manifest"]["files"]:
            source = self.root / generation_id / "files" / record["name"]
            target = destination / record["name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.restore-{os.getpid()}")
            shutil.copy2(source, tmp)
            os.replace(tmp, target)
            restored.append(str(target))
        _fsync_dir(destination)
        return {"generation_id": generation_id, "restored": restored}

    def _prune(self) -> None:
        generations = sorted(
            (p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.name,
            reverse=True,
        )
        for old in generations[self.keep:]:
            shutil.rmtree(old, ignore_errors=True)


class EncryptedBackup:
    """Portable AES-256-GCM archive derived from a user-held passphrase."""

    @staticmethod
    def _key(passphrase: str, salt: bytes) -> bytes:
        if len(passphrase) < 12:
            raise ValueError("backup passphrase must be at least 12 characters")
        return hashlib.scrypt(
            passphrase.encode("utf-8"),
            salt=salt,
            n=2**15,
            r=8,
            p=1,
            dklen=32,
            maxmem=64 * 1024 * 1024,
        )

    @classmethod
    def create(cls, source_dir: Path, output_path: Path, passphrase: str) -> dict[str, Any]:
        source_dir = Path(source_dir).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for path in sorted(source_dir.rglob("*")):
                if output_path == path.resolve() or "backups" in path.parts:
                    continue
                archive.add(path, arcname=path.relative_to(source_dir), recursive=False)
        salt = os.urandom(16)
        nonce = os.urandom(12)
        ciphertext = AESGCM(cls._key(passphrase, salt)).encrypt(nonce, buffer.getvalue(), MAGIC)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        with tmp.open("wb") as f:
            f.write(MAGIC + salt + nonce + ciphertext)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, output_path)
        _fsync_dir(output_path.parent)
        return {"path": str(output_path), "size": output_path.stat().st_size, "sha256": _hash_file(output_path)}

    @classmethod
    def extract(cls, archive_path: Path, destination: Path, passphrase: str) -> dict[str, Any]:
        raw = Path(archive_path).expanduser().read_bytes()
        if not raw.startswith(MAGIC) or len(raw) < len(MAGIC) + 28:
            raise ValueError("not a Bad Apple encrypted backup")
        offset = len(MAGIC)
        salt, nonce, ciphertext = raw[offset:offset + 16], raw[offset + 16:offset + 28], raw[offset + 28:]
        plaintext = AESGCM(cls._key(passphrase, salt)).decrypt(nonce, ciphertext, MAGIC)
        destination = Path(destination).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(plaintext), mode="r:gz") as archive:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if destination not in target.parents and target != destination:
                    raise ValueError(f"unsafe archive path: {member.name}")
            archive.extractall(destination, filter="data")
        return {"destination": str(destination), "files": sum(1 for p in destination.rglob("*") if p.is_file())}
