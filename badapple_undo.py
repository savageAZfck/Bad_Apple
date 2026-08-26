#!/usr/bin/env python3
"""Checksummed undo journal for reversible local file mutations."""

import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


class UndoJournal:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "undo"
        self.objects = self.root / "objects"
        self.journal = self.root / "journal.jsonl"
        self.lock = self.root / "journal.lock"
        self.objects.mkdir(parents=True, exist_ok=True)

    def capture_file(self, path: Path, operation: str) -> str:
        path = Path(path).expanduser().resolve()
        undo_id = uuid.uuid4().hex
        existed = path.is_file()
        object_path: Optional[Path] = None
        checksum = None
        if existed:
            object_path = self.objects / undo_id
            shutil.copy2(path, object_path)
            checksum = hashlib.sha256(object_path.read_bytes()).hexdigest()
        entry = {
            "id": undo_id,
            "operation": operation,
            "path": str(path),
            "existed": existed,
            "object": str(object_path) if object_path else None,
            "sha256": checksum,
            "created_at": time.time(),
            "undone": False,
        }
        self._append(entry)
        return undo_id

    def _append(self, entry: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            fd = os.open(self.journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, (json.dumps(entry, sort_keys=True) + "\n").encode())
                os.fsync(fd)
            finally:
                os.close(fd)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def entries(self) -> list[Dict[str, Any]]:
        if not self.journal.is_file():
            return []
        results = []
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return results

    def undo_last(self) -> str:
        entries = self.entries()
        undone_ids = {entry.get("id") for entry in entries if entry.get("operation") == "undo"}
        target = next(
            (entry for entry in reversed(entries) if entry.get("operation") != "undo" and entry.get("id") not in undone_ids),
            None,
        )
        if target is None:
            return "Nothing to undo."
        path = Path(target["path"])
        if target.get("existed"):
            object_path = Path(target["object"])
            if not object_path.is_file() or hashlib.sha256(object_path.read_bytes()).hexdigest() != target.get("sha256"):
                return "Undo refused: snapshot verification failed."
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.undo-{os.getpid()}")
            shutil.copy2(object_path, tmp)
            os.replace(tmp, path)
        elif path.exists():
            trash = Path.home() / ".Trash" / f"{path.name}.badapple-undo-{target['id'][:8]}"
            shutil.move(path, trash)
        self._append({"id": target["id"], "undone": True, "undo_at": time.time(), "operation": "undo"})
        return f"Undid {target['operation']} on {path}."
