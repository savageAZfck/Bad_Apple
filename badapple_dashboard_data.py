#!/usr/bin/env python3
"""Data-collection helpers for the Bad Apple dashboard.

These helpers do not depend on the global ``_server_instance`` and are used
by both the dashboard HTTP handlers and ``badapple_tools.system_dashboard``.
"""

import json
import re
import subprocess
import urllib.parse
from collections import deque
from pathlib import Path
from typing import Any

import psutil

import badapple_mcp_marketplace


def _run(*args, timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return (result.stdout or "").strip()
    except (subprocess.SubprocessError, OSError, ValueError):
        return ""


def _battery() -> dict[str, Any]:
    """Return battery percentage and power source on macOS."""
    out = _run("pmset", "-g", "batt")
    pct = None
    source = "unknown"
    if out:
        m = re.search(r"(\d+)%", out)
        if m:
            pct = int(m.group(1))
        if "AC Power" in out:
            source = "ac"
        elif "Battery Power" in out:
            source = "battery"
    return {"percent": pct, "source": source}


def _thermal() -> str:
    """Return a quick CPU thermal pressure string from memory_pressure."""
    out = _run("memory_pressure")
    for line in out.splitlines():
        if "System-wide memory free percentage" in line:
            return line.strip()
    return out.splitlines()[0] if out else "unknown"


def _badapple_proc() -> dict[str, Any] | None:
    """Find the Bad Apple MLX server process and return its stats."""
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "cpu_percent"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if "badapple_mlx_server" in cmd or p.info.get("name") == "badapple_mlx_server":
                with p.oneshot():
                    return {
                        "pid": p.pid,
                        "rss_mb": p.memory_info().rss // (1024 * 1024),
                        "vms_mb": p.memory_info().vms // (1024 * 1024),
                        "cpu_percent": p.cpu_percent(interval=0.1),
                    }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def _latest_log_perf() -> dict[str, Any] | None:
    """Parse the latest [perf] line from the current Bad Apple process log."""
    log = Path("/var/log/bad_apple_mlx_server.log")
    if not log.is_file():
        return None
    try:
        with log.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:  # noqa: BLE001 - catch-all wrapper
        return None
    # Only consider the current process by finding the most recent start marker.
    for i in range(len(lines) - 1, -1, -1):
        if _START_MARKER in lines[i]:
            lines = lines[i:]
            break
    for line in reversed(lines):
        m = re.search(r"\[perf\]\s+(.*)", line)
        if m:
            return {"raw": m.group(1).strip()}
    return None


_START_MARKER = "Bad Apple MLX server started"


def _tail_lines(path: str, n: int = 20, current_process_only: bool = True) -> list[str]:
    """Return the last n lines of a text file.

    If current_process_only is True, skip all lines before the most recent
    daemon startup marker so the tail only shows the running process.
    """
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except (OSError, ValueError):
        return []
    if current_process_only:
        for i in range(len(lines) - 1, -1, -1):
            if _START_MARKER in lines[i]:
                lines = lines[i:]
                break
    if n <= 0:
        return lines
    return lines[-n:] if len(lines) >= n else lines


def _tail_ledger(path: str, n: int = 20) -> list[Any]:
    """Return the last n JSONL entries, falling back to raw strings."""
    entries: list[Any] = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in deque(f, maxlen=n):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    entries.append(line)
    except (OSError, ValueError) as e:
        print(f"[dashboard] open failed: {e}", flush=True)
    return entries


def _voice_activity(n: int = 20) -> list[dict[str, Any]]:
    """Parse the menu bar voice debug log into a short activity feed."""
    events: list[dict[str, Any]] = []
    consume_re = re.compile(r"consume: isFinal=(true|false) transcript='(.*?)' .*?awaitingNextUtterance=(true|false)")
    deliver_re = re.compile(r"deliver: (.*)")
    complete_re = re.compile(r"completeVoiceResponse: (.*)")
    spoken_re = re.compile(r"spoken: (.*)")
    error_re = re.compile(r"consume error: (.*)")
    try:
        with open("/tmp/badapple_voice_debug.log", encoding="utf-8", errors="ignore") as f:
            for line in deque(f, maxlen=5000):
                if "consume:" in line:
                    m = consume_re.search(line)
                    if m:
                        events.append({
                            "type": "transcript",
                            "time": line[:20] if len(line) >= 20 else "",
                            "is_final": m.group(1) == "true",
                            "text": m.group(2),
                            "awaiting": m.group(3) == "true",
                        })
                elif "deliver:" in line:
                    m = deliver_re.search(line)
                    if m:
                        events.append({
                            "type": "command",
                            "time": line[:20] if len(line) >= 20 else "",
                            "text": m.group(1),
                        })
                elif "completeVoiceResponse:" in line:
                    m = complete_re.search(line)
                    if m:
                        events.append({
                            "type": "response",
                            "time": line[:20] if len(line) >= 20 else "",
                            "text": m.group(1).strip(),
                        })
                elif "spoken:" in line and "parsed actions:" in line:
                    m = spoken_re.search(line)
                    if m:
                        events.append({
                            "type": "spoken",
                            "time": line[:20] if len(line) >= 20 else "",
                            "text": m.group(1).strip(),
                        })
                elif "consume error:" in line:
                    m = error_re.search(line)
                    if m:
                        events.append({
                            "type": "error",
                            "time": line[:20] if len(line) >= 20 else "",
                            "text": m.group(1).strip(),
                        })
    except (OSError, ValueError) as e:
        print(f"[dashboard] open failed: {e}", flush=True)
    return list(reversed(events[-n:]))


def _query_int(path: str, key: str, default: int) -> int:
    """Parse a query string integer parameter."""
    parsed = urllib.parse.urlparse(path)
    values = urllib.parse.parse_qs(parsed.query).get(key)
    if not values:
        return default
    try:
        return max(1, int(values[0]))
    except ValueError:
        return default


def _load_mcp_servers() -> dict[str, Any]:
    try:
        registry = badapple_mcp_marketplace._load_registry()
        servers = [dict(s) for s in registry.get("servers", [])]
        for s in servers:
            s["airgap_blocked"] = badapple_mcp_marketplace.is_airgap() and s.get("name") in badapple_mcp_marketplace._NETWORK_MCP_SERVERS
        return {"servers": servers, "airgap": badapple_mcp_marketplace.is_airgap()}
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        return {"servers": [], "error": str(e)}
