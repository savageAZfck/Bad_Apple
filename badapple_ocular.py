#!/usr/bin/env python3
"""Ocular UI Stream: live screen capture and VLM description for Bad Apple.

Runs a background thread that captures the main screen and, optionally,
runs the local MLX VLM to describe it. The latest frame and description are
written to the user's `~/.bad_apple/ocular/` directory for the dashboard
and the assistant to query. Nothing leaves the Mac.
"""

from __future__ import annotations

import json
import os
import pwd
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import badapple_ambient
import badapple_vision


def _console_user() -> str | None:
    try:
        import getpass
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return None


def _user_home() -> Path:
    """Return a user-writable directory for the Ocular feed.

    In a LaunchDaemon context this is the console user's home; when running
    as the user it is the current home.
    """
    user = badapple_vision._console_user() or _console_user()
    if user:
        try:
            return Path(pwd.getpwnam(user).pw_dir)
        except (KeyError, OSError):
            pass
    return Path.home()


def _ensure_dir(path: Path) -> None:
    if path.exists():
        return
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError):
        pass

    user = badapple_vision._console_user() or _console_user()
    if os.geteuid() == 0 and user:
        try:
            info = pwd.getpwnam(user)
            os.chown(path, info.pw_uid, info.pw_gid, follow_symlinks=False)
        except (KeyError, OSError):
            pass


OCULAR_DIR = _user_home() / ".bad_apple" / "ocular"
_ensure_dir(OCULAR_DIR)

OCULAR_SCREEN = OCULAR_DIR / "screen.png"
OCULAR_CONTEXT = OCULAR_DIR / "context.json"

CAPTURE_INTERVAL = float(os.environ.get("BADAPPLE_OCULAR_CAPTURE_INTERVAL", "5"))
DESCRIBE_INTERVAL = float(os.environ.get("BADAPPLE_OCULAR_DESCRIBE_INTERVAL", "0"))
DESCRIBE_PROMPT = os.environ.get(
    "BADAPPLE_OCULAR_PROMPT",
    "Describe what is currently on the screen in a few sentences, focusing on the active window and visible text.",
)

_thread: threading.Thread | None = None
_stop_event = threading.Event()
_running = False
_capture_interval = CAPTURE_INTERVAL
_describe_interval = DESCRIBE_INTERVAL
_last_capture = 0.0
_last_describe = 0.0
_lock = threading.RLock()


def _describe_prompt() -> str:
    with _lock:
        return DESCRIBE_PROMPT


def _set_describe_prompt(prompt: str) -> None:
    with _lock:
        global DESCRIBE_PROMPT
        DESCRIBE_PROMPT = prompt


def _snapshot() -> None:
    global _last_capture, _last_describe
    try:
        _ensure_dir(OCULAR_SCREEN.parent)
        image_path = badapple_vision.capture_screen(path=OCULAR_SCREEN)
        if not image_path.is_file():
            return

        now = time.time()
        with _lock:
            _last_capture = now

        active = badapple_ambient._active_app_and_window()
        description = ""

        with _lock:
            do_describe = _describe_interval > 0 and (now - _last_describe >= _describe_interval)

        if do_describe:
            try:
                description = badapple_vision.get_vision_host().describe(
                    image_path,
                    prompt=_describe_prompt(),
                    max_tokens=256,
                )
                with _lock:
                    _last_describe = now
            except Exception as e:  # noqa: BLE001 - logged
                print(f"[ocular] VLM describe failed: {e}", flush=True)

        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "screen_path": str(image_path),
            "active_app": active.get("app", "unknown"),
            "window": active.get("window", "unknown"),
            "description": description,
            "capture_interval": _capture_interval,
            "describe_interval": _describe_interval,
        }
        OCULAR_CONTEXT.write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - logged
        print(f"[ocular] snapshot error: {e}\n{traceback.format_exc()}", flush=True)


def _loop() -> None:
    while not _stop_event.is_set():
        _snapshot()
        _stop_event.wait(_capture_interval)


def is_running() -> bool:
    return _running and _thread is not None and _thread.is_alive()


def start(capture_interval: float = CAPTURE_INTERVAL, describe_interval: float = DESCRIBE_INTERVAL, prompt: str | None = None) -> str:
    global _running, _thread, _capture_interval, _describe_interval
    with _lock:
        if is_running():
            return "Ocular stream already running."
        _capture_interval = float(capture_interval)
        _describe_interval = float(describe_interval)
        if prompt:
            _set_describe_prompt(prompt)
        _running = True
        _stop_event.clear()
        _thread = threading.Thread(target=_loop, name="badapple-ocular", daemon=True)
        _thread.start()
    return f"Ocular stream started (capture={_capture_interval}s, describe={_describe_interval}s)."


def stop() -> str:
    global _running
    with _lock:
        if not is_running():
            return "Ocular stream not running."
        _running = False
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=_capture_interval + 5)
    try:
        badapple_vision.unload_vision_model()
    except Exception as e:  # noqa: BLE001
        print(f"[ocular] unload vision model error: {e}", flush=True)
    return "Ocular stream stopped."


def capture_now(prompt: str | None = None) -> dict[str, Any]:
    with _lock:
        if prompt:
            _set_describe_prompt(prompt)
    _snapshot()
    return status()


def status() -> dict[str, Any]:
    with _lock:
        ctx: dict[str, Any] = {}
        if OCULAR_CONTEXT.is_file():
            try:
                ctx = json.loads(OCULAR_CONTEXT.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
                ctx = {"error": f"Error reading context: {e}"}
        return {
            "running": is_running(),
            "capture_interval": _capture_interval,
            "describe_interval": _describe_interval,
            "last_capture": _last_capture,
            "last_describe": _last_describe,
            "screen_url": "/api/ocular/screen.png",
            "context": ctx,
        }


def get_context() -> str:
    if not OCULAR_CONTEXT.is_file():
        return "No Ocular context captured yet."
    try:
        return OCULAR_CONTEXT.read_text(encoding="utf-8")
    except (OSError, ValueError) as e:
        return f"Error reading Ocular context: {e}"
