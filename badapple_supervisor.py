#!/usr/bin/env python3
"""Bounded launchd health supervisor for the Bad Apple service family."""

import argparse
import json
import os
import pwd
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from badapple_runtime import RuntimeControl

DATA_DIR = Path(os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple"))
STATE_PATH = DATA_DIR / "supervisor_state.json"
CHECK_INTERVAL = int(os.environ.get("BADAPPLE_SUPERVISOR_INTERVAL", "30"))
FAILURE_THRESHOLD = 3
RESTART_BUDGET = 2
RESTART_WINDOW = 600
STARTUP_GRACE = int(os.environ.get("BADAPPLE_SUPERVISOR_STARTUP_GRACE", "180"))


def _console_uid() -> int:
    try:
        user = subprocess.run(
            ["stat", "-f", "%Su", "/dev/console"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout.strip()
        return pwd.getpwnam(user).pw_uid
    except (subprocess.SubprocessError, OSError, ValueError):
        return os.getuid()


def _services() -> dict[str, dict[str, Any]]:
    uid = _console_uid()
    return {
        "gatekeeper": {"domain": "system/com.badapple.gatekeeper", "socket": "/var/run/badapple/substrate.sock"},
        "mlx": {"domain": "system/com.badapple.mlx", "socket": "/var/run/badapple/substrate_mlx.sock"},
        "tts": {"domain": f"gui/{uid}/com.badapple.tts", "socket": "/tmp/badapple_tts.sock"},
    }


def _atomic_state(state: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_name(f".{STATE_PATH.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_PATH)


def _load_state() -> dict[str, Any]:
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _launchd_running(domain: str) -> bool:
    result = subprocess.run(
        ["launchctl", "print", domain],
        capture_output=True,
        text=True,
        timeout=5,
    check=False)
    return result.returncode == 0 and "state = running" in result.stdout


def _socket_ready(path: str) -> bool:
    if not Path(path).exists():
        return False
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(1.0)
    try:
        client.connect(path)
        return True
    except OSError:
        return False
    finally:
        client.close()


def _healthy(service: dict[str, Any], entry: dict[str, Any], now: float) -> tuple[bool, dict[str, Any]]:
    running = _launchd_running(service["domain"])
    socket_ok = True if service["socket"] is None else _socket_ready(service["socket"])
    start_at = float(entry.get("start_at", 0.0))
    in_grace = (
        running
        and service["socket"] is not None
        and not socket_ok
        and start_at > 0
        and (now - start_at) < STARTUP_GRACE
    )
    detail = {"running": running, "socket_ready": socket_ok, "in_grace": in_grace}
    return running and (socket_ok or in_grace), detail


def _restart_allowed(entry: dict[str, Any], now: float) -> bool:
    attempts = [stamp for stamp in entry.get("restart_attempts", []) if now - stamp < RESTART_WINDOW]
    entry["restart_attempts"] = attempts
    return len(attempts) < RESTART_BUDGET


def _restart(domain: str) -> bool:
    return subprocess.run(
        ["launchctl", "kickstart", "-k", domain],
        capture_output=True,
        text=True,
        timeout=20,
    check=False).returncode == 0


def check_once(repair: bool = True) -> dict[str, Any]:
    now = time.time()
    state = _load_state()
    state.setdefault("services", {})
    report: dict[str, Any] = {"timestamp": now, "services": {}, "safe_mode": False}
    runtime = RuntimeControl(DATA_DIR)

    for name, service in _services().items():
        entry = state["services"].setdefault(
            name, {"consecutive_failures": 0, "restart_attempts": [], "start_at": 0.0}
        )
        running = _launchd_running(service["domain"])
        if running:
            if not entry.get("start_at"):
                entry["start_at"] = now
        else:
            entry["start_at"] = 0.0
        healthy, detail = _healthy(service, entry, now)
        if healthy:
            entry["consecutive_failures"] = 0
            detail["action"] = "starting" if detail.get("in_grace") else "none"
        else:
            entry["consecutive_failures"] = int(entry.get("consecutive_failures", 0)) + 1
            detail["action"] = "observe"
            if repair and entry["consecutive_failures"] >= FAILURE_THRESHOLD:
                if _restart_allowed(entry, now):
                    attempted = _restart(service["domain"])
                    entry["restart_attempts"].append(now)
                    entry["consecutive_failures"] = 0 if attempted else entry["consecutive_failures"]
                    detail["action"] = "restart" if attempted else "restart_failed"
                else:
                    reason = f"{name} exceeded restart budget"
                    runtime.enter_safe_mode(reason)
                    detail["action"] = "safe_mode"
                    report["safe_mode"] = True
        entry["last_check"] = now
        entry["healthy"] = healthy
        detail["healthy"] = healthy
        report["services"][name] = {**detail, "consecutive_failures": entry["consecutive_failures"]}

    state["last_report"] = report
    _atomic_state(state)
    return report


def supervisor_status() -> str:
    """Return the last supervisor health report as a string."""
    state = _load_state()
    report = state.get("last_report")
    if not report:
        return "No supervisor report yet."
    return json.dumps(report, indent=2, sort_keys=True)


def heal() -> str:
    """Run one supervisor check with repair enabled and return the report."""
    report = check_once(repair=True)
    healthy = all(s.get("healthy", False) for s in report.get("services", {}).values())
    if healthy and not report.get("safe_mode"):
        return "Self-heal check passed. All services healthy."
    if report.get("safe_mode"):
        return "Self-heal check performed. System is in SAFE MODE.\n" + json.dumps(report, indent=2, sort_keys=True)
    return "Self-heal check performed. Some services may still be recovering:\n" + json.dumps(report, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-repair", action="store_true")
    args = parser.parse_args()
    if args.once:
        print(json.dumps(check_once(repair=not args.no_repair), indent=2))
        return 0
    while True:
        report = check_once(repair=not args.no_repair)
        print(json.dumps(report, sort_keys=True), flush=True)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
