#!/usr/bin/env python3
"""Background file watcher that auto-indexes the active workspace.

Scans the workspace every few seconds, detects new or modified text/code files,
and feeds them into BadAppleKnowledge.  Uses a simple mtime map so re-indexing
is incremental.
"""

import os
import threading
import time
from pathlib import Path
from typing import Any

SCAN_INTERVAL = float(os.environ.get("BADAPPLE_WORKSPACE_SCAN_INTERVAL", "10"))
MAX_FILE_SIZE = 512 * 1024
INDEX_EXTENSIONS = {".txt", ".md", ".rs", ".swift", ".py", ".sh", ".toml", ".json", ".yaml", ".yml", ".html", ".css", ".js"}
SKIP_DIRS = {".git", ".svn", ".venv", ".env", "venv", "env", "node_modules", "target", "build", "dist", "__pycache__", ".pytest_cache"}


class WorkspaceWatcher:
    """Polls a workspace directory and auto-indexes changed files."""

    def __init__(self, knowledge: Any):
        self.knowledge = knowledge
        self.workspace: Path | None = None
        self._mtimes: dict[str, float] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    def set_workspace(self, path: Path | None) -> None:
        with self._lock:
            self.workspace = Path(path).expanduser() if path else None
            self._mtimes.clear()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="badapple-workspace-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _should_skip(self, p: Path) -> bool:
        for part in p.parts:
            if part in SKIP_DIRS:
                return True
        return False

    def _collect_files(self, root: Path) -> set[Path]:
        files: set[Path] = set()
        if not root.is_dir():
            return files
        try:
            for f in root.rglob("*"):
                if not f.is_file():
                    continue
                if self._should_skip(f):
                    continue
                if f.suffix.lower() not in INDEX_EXTENSIONS:
                    continue
                if f.stat().st_size > MAX_FILE_SIZE:
                    continue
                files.add(f)
        except (OSError, ValueError) as e:
            print(f"[workspace_watcher] scan error: {e}", flush=True)
        return files

    def _loop(self) -> None:
        print("[workspace_watcher] loop started", flush=True)
        while self._running:
            with self._lock:
                ws = self.workspace
            if ws is None:
                time.sleep(SCAN_INTERVAL)
                continue

            print(f"[workspace_watcher] scanning {ws}", flush=True)
            files = self._collect_files(ws)
            print(f"[workspace_watcher] found {len(files)} candidate files", flush=True)
            to_index: set[Path] = set()
            new_mtimes: dict[str, float] = {}
            for f in files:
                try:
                    mtime = f.stat().st_mtime
                except (OSError, ValueError):
                    continue
                key = str(f)
                new_mtimes[key] = mtime
                if key not in self._mtimes or self._mtimes[key] != mtime:
                    to_index.add(f)

            if to_index:
                print(f"[workspace_watcher] indexing {len(to_index)} file(s)", flush=True)
                try:
                    count = self.knowledge.index_paths(list(to_index))
                    print(f"[workspace_watcher] indexed {count} chunk(s) from {len(to_index)} file(s)", flush=True)
                except Exception as e:  # noqa: BLE001 - catch-all wrapper
                    import traceback
                    print(f"[workspace_watcher] index error: {e}\n{traceback.format_exc()}", flush=True)

            with self._lock:
                self._mtimes = new_mtimes

            time.sleep(SCAN_INTERVAL)
