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


def _poll_until(predicate, timeout: float, interval: float = 0.3):
    """Poll `predicate()` until it returns a truthy value or timeout elapses.

    Returns the last (falsy) value on timeout rather than raising, so callers
    can produce a descriptive assertion message with the final observed state.
    """
    deadline = time.monotonic() + timeout
    result = None
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    return result


def test_hibernation() -> None:
    # Set a short hibernate window and poll for it to fire, rather than a
    # fixed sleep -- a loaded machine can easily take longer than a fixed
    # 6s window to actually hibernate, causing a flaky failure.
    _cli("hibernate after 5")
    hibernating = _poll_until(lambda: _agent("status")["result"].get("hibernating") is True, timeout=20)
    assert hibernating, "should be hibernating within 20s of `hibernate after 5`"

    # Restore a long hibernate window *before* waking it back up. The awake
    # window after a wake-up ping only lasts as long as hibernate_after, and
    # each status poll has its own subprocess/SLICKS-handshake latency -- with
    # a 5s window, polling can race past a real but brief awake blip. Widening
    # the window first removes the race instead of chasing it.
    _cli("hibernate after 300")
    _cli("ping")
    awake = _poll_until(lambda: _agent("status")["result"].get("hibernating") is False, timeout=10)
    assert awake, "should be awake within 10s of a new query"
    print("OK: hibernation")


def test_ui_action_and_browser_action_tools_registered() -> None:
    resp = _agent("discover")
    tools = resp["result"]["tools"]
    names = {t["function"]["name"] for t in tools}
    assert "ui_action" in names, f"ui_action should be exposed; got {names}"
    assert "browser_action" in names, f"browser_action should be exposed; got {names}"
    print("OK: ui_action and browser_action tools registered")


def test_ui_action_info_returns_without_crashing() -> None:
    # A light functional check: invoking ui_action's "info" action should
    # always return *some* string (either a UI tree or a friendly "helper
    # not available" message), never raise or return nothing. This does not
    # require a GUI session to be present, so it's safe in headless CI too.
    resp = _agent("invoke", "ui_action", json.dumps({"action": "info"}))
    result = resp["result"]["result"]
    assert isinstance(result, str) and result, f"ui_action info should return a non-empty string, got {result!r}"
    print("OK: ui_action info tool call completes")


def main() -> int:
    tests = [
        test_runtime_status,
        test_hibernation,
        test_ui_action_and_browser_action_tools_registered,
        test_ui_action_info_returns_without_crashing,
    ]
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
