#!/usr/bin/env python3
"""Fast-path tests for direct answers, kill switch, and air-gap state.

These tests assume the daemon is running on the default Unix socket. They use
the badapple CLI and agent_client.py to exercise the short-circuit paths
without loading the full 9B brain for every assertion.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AGENT_CLIENT = [sys.executable, str(REPO / "agent_client.py")]
CLI = REPO / "target" / "release" / "badapple"


def _agent(*args: str) -> dict:
    result = subprocess.run(
        AGENT_CLIENT + list(args),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"agent_client {' '.join(args)} failed: {result.stderr or result.stdout}")
    return json.loads(result.stdout)


def _cli(prompt: str, max_tokens: int = 120) -> str:
    if not CLI.is_file():
        raise RuntimeError(f"CLI not built: {CLI}")
    result = subprocess.run(
        [str(CLI), "-n", str(max_tokens), prompt],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CLI failed: {result.stderr or result.stdout}")
    return result.stdout.strip()


def _unique(prompt: str) -> str:
    # Append a random nonce so semantic cache lookups do not shadow the fast path.
    import random
    return f"{prompt} nonce{random.randrange(10**9)}"


def test_identity_shortcut() -> None:
    """Identity questions should hit the fast meta-response path."""
    text = _cli(_unique("who are you"), max_tokens=60)
    assert "Bad Apple" in text, f"identity response should mention Bad Apple: {text}"
    assert len(text.split()) < 80, f"identity response should be short, got {len(text.split())} words"
    print("OK: identity_shortcut")


def test_capabilities_shortcut() -> None:
    """Capability questions return the fast-tier list without running the 9B or a directory tool."""
    text = _cli(_unique("what can you do"), max_tokens=120)
    expected = [
        "answer questions",
        "local tools",
        "mcp",
        "kill switch",
        "air-gap",
        "benchmarks",
        "on your Mac",
    ]
    missing = [e for e in expected if e.lower() not in text.lower()]
    assert not missing, f"capability response missing {missing}: {text}"
    assert "No output firewall" not in text, f"capability response should not be blocked: {text}"
    print("OK: capabilities_shortcut")


def test_feature_list_shortcut() -> None:
    """'list all your features' must not be mistaken for a directory-listing tool call."""
    text = _cli(_unique("list all your features"), max_tokens=120)
    # The capability fast path answers with the "what I can do" list, not the word "feature".
    assert "what i can do" in text.lower(), f"feature list should be answered directly: {text}"
    assert "directory" not in text.lower() and not re.search(r"\bls\b", text.lower()), f"feature list should not run directory tools: {text}"
    print("OK: feature_list_shortcut")


def test_kill_switch_and_resume() -> None:
    """Engaging the kill switch should set the runtime to STOPPED; resuming clears it."""
    _agent("kill")
    status = _agent("status")
    assert status.get("type") == "response", f"status should be a response: {status}"
    assert status["result"]["runtime"]["mode"] == "STOPPED", f"runtime should be STOPPED after kill: {status}"

    _agent("resume")
    status = _agent("status")
    assert status["result"]["runtime"]["mode"] == "READY", f"runtime should be READY after resume: {status}"
    print("OK: kill_switch_and_resume")


def test_airgap_status() -> None:
    """Air-gap status should report whether local-only mode is enabled."""
    status = _agent("airgap")
    assert status.get("type") == "response", f"airgap should be a response: {status}"
    assert "airgap" in status["result"], f"airgap status missing 'airgap' key: {status}"
    print("OK: airgap_status")


if __name__ == "__main__":
    test_identity_shortcut()
    test_capabilities_shortcut()
    test_feature_list_shortcut()
    test_kill_switch_and_resume()
    test_airgap_status()
