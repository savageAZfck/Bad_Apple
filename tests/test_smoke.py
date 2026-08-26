#!/usr/bin/env python3
"""Smoke tests for the Bad Apple runtime, tools, and new features.

These tests assume the daemon is running on the default Unix socket. They use
agent_client.py and the badapple CLI to avoid loading the full model stack.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AGENT_CLIENT = [sys.executable, str(REPO / "agent_client.py")]
CLI = REPO / "target" / "release" / "badapple"


def _agent(*args: str) -> dict:
    result = subprocess.run(AGENT_CLIENT + list(args), capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"agent_client {' '.join(args)} failed: {result.stderr or result.stdout}")
    return json.loads(result.stdout)


def _cli(prompt: str, max_tokens: int = 60) -> str:
    if not CLI.is_file():
        raise RuntimeError(f"CLI not built: {CLI}")
    result = subprocess.run([str(CLI), "-n", str(max_tokens), prompt], capture_output=True, text=True, timeout=120, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"CLI failed: {result.stderr or result.stdout}")
    return result.stdout.strip()


def test_runtime_status() -> None:
    resp = _agent("status")
    assert resp["type"] == "response", "status should return a response"
    result = resp["result"]
    assert result["runtime"]["mode"] == "READY", f"runtime should be READY, got {result['runtime']['mode']}"
    assert "hibernating" in result, "status should include hibernation"
    assert "idle_seconds" in result, "status should include idle_seconds"
    print("OK: runtime_status")


def test_hibernation() -> None:
    # Set a short hibernate window and wait for it to fire.
    _cli("hibernate after 5")
    time.sleep(6)
    resp = _agent("status")
    result = resp["result"]
    assert result.get("hibernating") is True, f"should be hibernating, got {result.get('hibernating')}"

    # A new query should wake it up and stay awake for a short window.
    _cli("ping")
    time.sleep(0.5)
    resp = _agent("status")
    result = resp["result"]
    assert result.get("hibernating") is False, f"should be awake after query, got {result.get('hibernating')}"

    # Restore the default.
    _cli("hibernate after 300")
    print("OK: hibernation")


def test_ui_action_tool_exists() -> None:
    resp = _agent("discover")
    tools = resp["result"]["tools"]
    names = {t["function"]["name"] for t in tools}
    assert "ui_action" in names, f"ui_action should be exposed; got {names}"
    print("OK: ui_action tool registered")


def main() -> int:
    tests = [test_runtime_status, test_hibernation, test_ui_action_tool_exists]
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            print(f"FAIL: {t.__name__}: {e}")
            return 1
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
