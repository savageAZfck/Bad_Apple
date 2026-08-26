#!/usr/bin/env python3
"""Background file watcher that auto-indexes the active workspace.

Uses macOS FSEvents via the `watchdog` observer when available. This is
near-zero CPU when idle and reacts instantly to file changes instead of
polling every 10 seconds.

Falls back to the legacy mtime polling loop if watchdog is not installed.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any

MAX_FILE_SIZE = 512 * 1024
INDEX_EXTENSIONS = {".txt", ".md", ".rs", ".swift", ".py", ".sh", ".toml", ".json", ".yaml", ".yml", ".html", ".css", ".js"}
SKIP_DIRS = {".git", ".svn", ".venv", ".env", "venv", "env", "node_modules", "target", "build", "dist", "__pycache__", ".pytest_cache"}
DEBOUNCE = float(os.environ.get("BADAPPLE_WORKSPACE_DEBOUNCE", "0.5"))


try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except Exception:  # noqa: BLE001 - optional dependency
    HAS_WATCHDOG = False


class _WorkspaceEventHandler(FileSystemEventHandler):
    """Enqueue files from FSEvents for indexing."""

    def __init__(self, watcher: WorkspaceWatcher):
        self.watcher = watcher

    def on_created(self, event):
        if not event.is_directory:
            self.watcher._enqueue(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self.watcher._enqueue(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory and getattr(event, "dest_path", None):
            self.watcher._enqueue(Path(event.dest_path))


class WorkspaceWatcher:
    """Watches a workspace directory and auto-indexes changed files."""

    def __init__(self, knowledge: Any):
        self.knowledge = knowledge
        self.workspace: Path | None = None
        self._mtimes: dict[str, float] = {}
        self._running = False
        self._lock = threading.RLock()

        self._queue: queue.Queue[Path | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._observer: Observer | None = None
        self._watch = None
        self._handler = _WorkspaceEventHandler(self)

    def set_workspace(self, path: Path | None) -> None:
        with self._lock:
            try:
                new = Path(path).expanduser().resolve() if path else None
            except (OSError, ValueError):
                new = Path(path).expanduser() if path else None
            if self.workspace == new:
                return

            if self._observer is not None and self._watch is not None:
                try:
                    self._observer.unschedule(self._watch)
                except Exception as e:  # noqa: BLE001
                    print(f"[workspace_watcher] unschedule error: {e}", flush=True)
                self._watch = None

            self.workspace = new
            self._mtimes.clear()

            if new is not None and self._observer is not None and self._running:
                try:
                    self._watch = self._observer.schedule(self._handler, str(new), recursive=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[workspace_watcher] schedule error: {e}", flush=True)

    def start(self) -> None:
        if self._running:
            return
        self._running = True

        if HAS_WATCHDOG:
            self._observer = Observer()
            if self.workspace is not None:
                try:
                    self._watch = self._observer.schedule(self._handler, str(self.workspace), recursive=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[workspace_watcher] schedule error: {e}", flush=True)
            self._observer.start()
            self._worker = threading.Thread(target=self._fsevents_worker, name="badapple-watcher-worker", daemon=True)
            print("[workspace_watcher] FSEvents watcher started", flush=True)
        else:
            self._worker = threading.Thread(target=self._poll_loop, name="badapple-watcher-poll", daemon=True)
            print("[workspace_watcher] watchdog not available; using poll fallback", flush=True)

        self._worker.start()

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)

        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=2)
            except Exception as e:  # noqa: BLE001
                print(f"[workspace_watcher] observer stop error: {e}", flush=True)
            self._observer = None
            self._watch = None

        if self._worker is not None:
            self._worker.join(timeout=2)

    def _should_skip(self, p: Path) -> bool:
        for part in p.parts:
            if part in SKIP_DIRS:
                return True
        return False

    def _should_index(self, p: Path) -> bool:
        if p.suffix.lower() not in INDEX_EXTENSIONS:
            return False
        if self._should_skip(p):
            return False
        try:
            if p.stat().st_size > MAX_FILE_SIZE:
                return False
        except (OSError, ValueError):
            return False
        return True

    def _enqueue(self, path: Path) -> None:
        if self._should_index(path):
            self._queue.put(path)

    def _drain_batch(self, first: Path) -> list[Path]:
        """Collect paths that arrive in the debounce window."""
        batch = {first}
        deadline = time.time() + DEBOUNCE
        while time.time() < deadline:
            try:
                extra = self._queue.get(timeout=0.05)
                if extra is None:
                    self._running = False
                    break
                batch.add(extra)
            except queue.Empty:
                break
        return [p for p in batch if p.is_file() and self._should_index(p)]

    def _fsevents_worker(self) -> None:
        while self._running:
            try:
                path = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if path is None:
                break

            batch = self._drain_batch(path)
            if not batch:
                continue

            try:
                count = self.knowledge.index_paths(batch)
                print(f"[workspace_watcher] indexed {count} chunk(s) from {len(batch)} file(s)", flush=True)
            except Exception as e:
                print(f"[workspace_watcher] index error: {e}\n{traceback.format_exc()}", flush=True)

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

    def _poll_loop(self) -> None:
        scan_interval = float(os.environ.get("BADAPPLE_WORKSPACE_SCAN_INTERVAL", "10"))
        print("[workspace_watcher] poll loop started", flush=True)
        while self._running:
            with self._lock:
                ws = self.workspace
            if ws is None:
                time.sleep(scan_interval)
                continue

            files = self._collect_files(ws)
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
                except Exception as e:
                    print(f"[workspace_watcher] index error: {e}\n{traceback.format_exc()}", flush=True)

            with self._lock:
                self._mtimes = new_mtimes

            time.sleep(scan_interval)
