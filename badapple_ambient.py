#!/usr/bin/env python3
"""Always-on ambient screen/app context for Bad Apple.

Runs a background thread that periodically captures the active app/window
title and a screenshot. The snapshot is stored locally and can be queried by
the assistant. No cloud.
"""

import json
import os
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path


def _ambient_dir() -> Path:
    import tempfile
    for d in (Path("/var/lib/bad_apple"), Path.home() / ".bad_apple", Path(tempfile.gettempdir())):
        try:
            d.mkdir(parents=True, exist_ok=True)
            test = d / ".write_test"
            test.write_text("x")
            test.unlink()
            return d / "ambient"
        except (PermissionError, OSError):
            continue
    return Path.home() / ".bad_apple" / "ambient"


_AMBIENT_DIR = _ambient_dir()
_AMBIENT_DIR.mkdir(parents=True, exist_ok=True)
_CONTEXT_FILE = _AMBIENT_DIR / "context.json"
_SCREEN_FILE = _AMBIENT_DIR / "screen.png"

_thread: threading.Thread | None = None
_stop_event = threading.Event()
_interval = 30.0


def _active_app_and_window() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to get name of first application process whose frontmost is true'],
            capture_output=True,
            text=True,
            timeout=5,
        check=False)
        app = (result.stdout or "unknown").strip()
    except (subprocess.SubprocessError, OSError, ValueError):
        app = "unknown"
    try:
        result = subprocess.run(
            ["osascript", "-e", f'tell application "System Events" to tell process "{app}" to get name of first window'],
            capture_output=True,
            text=True,
            timeout=5,
        check=False)
        window = (result.stdout or "unknown").strip()
    except (subprocess.SubprocessError, OSError, ValueError):
        window = "unknown"
    return {"app": app, "window": window}


def _capture_screen() -> Path | None:
    try:
        # Prefer screencapture if available (macOS built-in)
        if shutil.which("screencapture"):
            tmp = _SCREEN_FILE.with_suffix(".tmp.png")
            subprocess.run(["screencapture", "-x", str(tmp)], check=True, timeout=10)
            tmp.replace(_SCREEN_FILE)
            return _SCREEN_FILE
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        print(f"[ambient] screen capture failed: {e}", flush=True)
    return None


def _snapshot():
    try:
        context = _active_app_and_window()
        screen = _capture_screen()
        context.update({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "screen_path": str(screen) if screen else None,
        })
        _CONTEXT_FILE.write_text(json.dumps(context, indent=2), encoding="utf-8")
    except (TypeError, ValueError, OSError) as e:
        print(f"[ambient] snapshot error: {e}", flush=True)


def _loop():
    while not _stop_event.is_set():
        _snapshot()
        _stop_event.wait(_interval)


def start(interval: float = 30.0) -> str:
    global _thread, _interval
    if _thread is not None and _thread.is_alive():
        return "Ambient capture already running."
    _interval = float(interval)
    _stop_event.clear()
    _thread = threading.Thread(target=_loop, daemon=True)
    _thread.start()
    return f"Ambient capture started (every {_interval}s)."


def stop() -> str:
    if _thread is None or not _thread.is_alive():
        return "Ambient capture not running."
    _stop_event.set()
    _thread.join(timeout=_interval + 5)
    return "Ambient capture stopped."


def is_running() -> bool:
    return _thread is not None and _thread.is_alive()


def status() -> str:
    running = is_running()
    if not _CONTEXT_FILE.is_file():
        return f"Ambient capture running={running}; no snapshots yet."
    try:
        data = json.loads(_CONTEXT_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
        return f"Error reading context: {e}"
    return f"Ambient capture running={running}; latest: {data.get('timestamp')} — app='{data.get('app')}', window='{data.get('window')}', screen='{data.get('screen_path')}'"


def get_context() -> str:
    if not _CONTEXT_FILE.is_file():
        return "No ambient context captured yet."
    try:
        return _CONTEXT_FILE.read_text(encoding="utf-8")
    except (OSError, ValueError) as e:
        return f"Error reading context: {e}"


# Optionally auto-start if env var set.
if os.environ.get("BADAPPLE_AMBIENT", "0") == "1":
    start()
