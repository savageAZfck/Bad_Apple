#!/usr/bin/env python3
"""Power and performance dashboard for Bad Apple.

Gathers local system metrics, the Bad Apple daemon process stats, and the
latest performance snapshot from the log. No cloud.
"""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import psutil


def _run(*args, timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (result.stdout or "").strip()
    except Exception:
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


def _badapple_proc() -> Optional[dict[str, Any]]:
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


def _latest_log_perf() -> Optional[dict[str, Any]]:
    """Parse the latest [perf] line from the Bad Apple log."""
    log = Path("/var/log/bad_apple_mlx_server.log")
    if not log.is_file():
        return None
    try:
        with log.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return None
    for line in reversed(lines):
        m = re.search(r"\[perf\]\s+(.*)", line)
        if m:
            return {"raw": m.group(1).strip()}
    return None


def snapshot() -> str:
    """Return a JSON string with the current system/daemon snapshot."""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage("/")
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "cpu": {
            "count": psutil.cpu_count(logical=True),
            "percent": psutil.cpu_percent(interval=0.5),
            "load_avg_1m": os.getloadavg()[0],
        },
        "memory": {
            "total_gb": round(mem.total / 1e9, 2),
            "used_gb": round(mem.used / 1e9, 2),
            "free_gb": round(mem.free / 1e9, 2),
            "percent": mem.percent,
        },
        "swap": {
            "total_gb": round(swap.total / 1e9, 2),
            "used_gb": round(swap.used / 1e9, 2),
        },
        "disk_root": {
            "total_gb": round(disk.total / 1e9, 2),
            "used_gb": round(disk.used / 1e9, 2),
            "free_gb": round(disk.free / 1e9, 2),
            "percent": round((disk.used / disk.total) * 100, 1),
        },
        "battery": _battery(),
        "thermal_pressure": _thermal(),
        "bad_apple_process": _badapple_proc(),
        "latest_log_perf": _latest_log_perf(),
    }
    return json.dumps(data, indent=2, default=str)
