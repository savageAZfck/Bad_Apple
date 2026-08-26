#!/usr/bin/env python3
"""Aqua user-session helper for Bad Apple.

Runs in the macOS user/Aqua session (launched by the menu bar) and handles
shortcuts, AppleScript GUI, and other user-context actions that fail when called
from the system LaunchDaemon.

Listens on a Unix domain socket and speaks line-delimited JSON. No TCP, no cloud.
"""

import json
import os
import socketserver
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_SOCKET_PATH = "/var/run/badapple/aqua_helper.sock"


def _socket_path() -> str:
    return os.environ.get("BADAPPLE_AQUA_SOCKET", DEFAULT_SOCKET_PATH)


def _remove_stale(path: str) -> None:
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except Exception:
        pass


def _run_shortcut(name: str, input_text: str = "", timeout: int = 60) -> Dict[str, Any]:
    if not name:
        return {"ok": False, "error": "shortcut name is required"}
    try:
        cmd = ["shortcuts", "run", name]
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip() or result.stdout.strip() or "shortcut failed"}
        return {"ok": True, "output": (result.stdout or "").strip()}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"shortcut '{name}' timed out"}
    except Exception as e:
        return {"ok": False, "error": f"shortcut error: {e}"}


def _list_shortcuts(timeout: int = 15) -> Dict[str, Any]:
    try:
        result = subprocess.run(["shortcuts", "list"], capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip() or result.stdout.strip() or "list failed"}
        lines = [l.strip() for l in (result.stdout or "").splitlines() if l.strip()][:100]
        return {"ok": True, "shortcuts": lines}
    except Exception as e:
        return {"ok": False, "error": f"list error: {e}"}


def _capture_screen(path: str, region: str = "") -> Dict[str, Any]:
    """Capture the main screen to a PNG using the Bad Apple screen-capture helper.

    Falls back to macOS screencapture if the helper is missing.
    """
    try:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Prefer the Bad Apple helper so the Screen Recording prompt is
        # attributed to the signed Bad Apple.app bundle, not python3.
        app_paths = [
            "/Applications/Bad Apple.app/Contents/Helpers/BadAppleScreenCapture",
            str(Path.home() / "bad_apple/target/release/Bad Apple.app/Contents/Helpers/BadAppleScreenCapture"),
            str(Path(__file__).parent / "Bad Apple.app/Contents/Helpers/BadAppleScreenCapture"),
        ]
        helper = next((p for p in app_paths if Path(p).is_file()), None)

        if helper and not region:
            result = subprocess.run([helper, "--output", str(out)], capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and out.is_file() and out.stat().st_size > 0:
                return {"ok": True, "path": str(out)}
            if result.returncode != 0:
                return {"ok": False, "error": result.stderr.strip() or result.stdout.strip() or "BadAppleScreenCapture failed"}

        # Fallback to the system screencapture utility.
        cmd = ["screencapture", "-x"]
        if region:
            cmd.extend(["-R", region])
        else:
            cmd.append("-S")  # main screen
        cmd.append(str(out))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip() or result.stdout.strip() or "screencapture failed"}
        if not out.is_file() or out.stat().st_size == 0:
            return {"ok": False, "error": "screencapture produced no image"}
        return {"ok": True, "path": str(out)}
    except Exception as e:
        return {"ok": False, "error": f"screen capture error: {e}"}


def _helper_executable(name: str) -> Optional[str]:
    for path in [
        f"/Applications/Bad Apple.app/Contents/Helpers/{name}",
        str(Path.home() / "bad_apple/target/release/Bad Apple.app/Contents/Helpers/{name}"),
        str(Path(__file__).parent / f"Bad Apple.app/Contents/Helpers/{name}"),
    ]:
        if Path(path).is_file():
            return path
    return None


def _ui_info() -> Dict[str, Any]:
    """Return the frontmost app/window and a list of named accessible UI elements."""
    helper = _helper_executable("BadAppleUI")
    if helper:
        result = subprocess.run([helper, "--action", "info"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            parts = result.stdout.strip().split("|", 2)
            if len(parts) >= 3:
                return {"ok": True, "app": parts[0], "window": parts[1], "elements": [e.strip() for e in parts[2].strip("{}").split(",") if e.strip()]}
        return {"ok": False, "error": result.stderr.strip() or result.stdout.strip() or "BadAppleUI info failed"}
    script = '''
    tell application "System Events"
        set p to first application process whose frontmost is true
        set appName to name of p
        set w to front window of p
        set winName to name of w
        set elements to {}
        set counter to 0
        repeat with e in (entire contents of w)
            try
                if counter > 100 then exit repeat
                if exists e then
                    set n to name of e
                    set r to role of e
                    if n is not missing value then
                        set end of elements to (r & ": " & n)
                        set counter to counter + 1
                    end if
                end if
            end try
        end repeat
        return appName & "|" & winName & "|" & (elements as string)
    end tell
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip() or result.stdout.strip() or "ui_info failed"}
    parts = result.stdout.strip().split("|", 2)
    if len(parts) < 3:
        return {"ok": False, "error": "unexpected ui_info output"}
    return {"ok": True, "app": parts[0], "window": parts[1], "elements": [e.strip() for e in parts[2].strip("{}").split(",") if e.strip()]}


def _ui_click(target: str) -> Dict[str, Any]:
    """Click the first accessible element whose name matches the target."""
    if not target:
        return {"ok": False, "error": "target name is required"}
    helper = _helper_executable("BadAppleUI")
    if helper:
        result = subprocess.run([helper, "--action", "click", "--target", target], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {"ok": True, "result": result.stdout.strip()}
        return {"ok": False, "error": result.stderr.strip() or result.stdout.strip() or "BadAppleUI click failed"}
    script = f'''
    tell application "System Events"
        set p to first application process whose frontmost is true
        set w to front window of p
        repeat with e in (entire contents of w)
            try
                if name of e is "{target.replace('"', '\\"')}" then
                    click e
                    return "clicked {target.replace('"', '\\"')}"
                end if
            end try
        end repeat
        return "not found"
    end tell
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip() or result.stdout.strip() or "ui_click failed"}
    text = result.stdout.strip()
    if text == "not found":
        return {"ok": False, "error": f"element '{target}' not found"}
    return {"ok": True, "result": text}


def _ui_type(target: str, text: str) -> Dict[str, Any]:
    """Type text into the named text field of the frontmost window."""
    if not target or text is None:
        return {"ok": False, "error": "target name and text are required"}
    helper = _helper_executable("BadAppleUI")
    if helper:
        result = subprocess.run([helper, "--action", "type", "--target", target, "--text", text], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {"ok": True, "result": result.stdout.strip()}
        return {"ok": False, "error": result.stderr.strip() or result.stdout.strip() or "BadAppleUI type failed"}
    script = f'''
    tell application "System Events"
        set p to first application process whose frontmost is true
        set w to front window of p
        repeat with e in (entire contents of w)
            try
                if name of e is "{target.replace('"', '\\"')}" then
                    set value of e to "{text.replace('"', '\\"').replace(chr(10), '\\n')}"
                    return "typed into {target.replace('"', '\\"')}"
                end if
            end try
        end repeat
        return "not found"
    end tell
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip() or result.stdout.strip() or "ui_type failed"}
    out = result.stdout.strip()
    if out == "not found":
        return {"ok": False, "error": f"text field '{target}' not found"}
    return {"ok": True, "result": out}


def _handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    command = req.get("command")
    if command == "list_shortcuts":
        return _list_shortcuts(int(req.get("timeout") or 15))
    if command == "run_shortcut":
        return _run_shortcut(req.get("name", ""), req.get("input", ""), int(req.get("timeout") or 60))
    if command == "capture_screen":
        return _capture_screen(req.get("path", ""), req.get("region", ""))
    if command == "ui_info":
        return _ui_info()
    if command == "ui_click":
        return _ui_click(req.get("target", ""))
    if command == "ui_type":
        return _ui_type(req.get("target", ""), req.get("text", ""))
    return {"ok": False, "error": f"unknown command '{command}'"}


class _AquaHelperHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line.decode("utf-8"))
                resp = _handle_request(req)
            except Exception as e:
                resp = {"ok": False, "error": f"invalid request: {e}"}
            self.wfile.write(json.dumps(resp).encode("utf-8") + b"\n")
            self.wfile.flush()


def call_aqua(command: str, timeout: float = 15.0, **kwargs) -> Optional[Dict[str, Any]]:
    """Call an Aqua helper over its Unix socket and return its JSON response."""
    path = _socket_path()
    if not Path(path).exists():
        return None
    try:
        import socket
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(path)
            req = json.dumps({"command": command, **kwargs}).encode("utf-8") + b"\n"
            s.sendall(req)
            with s.makefile("rb") as f:
                line = f.readline()
                if not line:
                    return None
                return json.loads(line.decode("utf-8"))
    except Exception:
        return None


def start() -> None:
    path = _socket_path()
    _remove_stale(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    server = socketserver.ThreadingUnixStreamServer(path, _AquaHelperHandler)
    os.chmod(path, 0o666)
    print(f"[aqua_helper] listening on {path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _remove_stale(path)


if __name__ == "__main__":
    start()
