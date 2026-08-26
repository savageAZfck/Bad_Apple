#!/usr/bin/env python3
"""Runtime safety, health, circuit breaking, and resource governance."""

import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class RuntimeControl:
    """Durable operating mode plus an in-process generation cancellation event."""

    SCHEMA_VERSION = 1

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "runtime_state.json"
        self._lock = threading.RLock()
        self.cancel_event = threading.Event()
        self._state = self._load()
        if os.environ.get("BADAPPLE_PRIVATE_MODE") == "1":
            self._state["private_mode"] = True
        if self._state.get("killed"):
            self.cancel_event.set()

    def _load(self) -> dict[str, Any]:
        default = {
            "schema_version": self.SCHEMA_VERSION,
            "mode": "STARTING",
            "private_mode": False,
            "killed": False,
            "safe_mode_reason": None,
            "revision": 0,
            "updated_at": time.time(),
        }
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("schema_version") == self.SCHEMA_VERSION:
                default.update(loaded)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return default

    def _update(self, **changes: Any) -> dict[str, Any]:
        with self._lock:
            self._state.update(changes)
            self._state["revision"] = int(self._state.get("revision", 0)) + 1
            self._state["updated_at"] = time.time()
            _atomic_json(self.path, self._state)
            return dict(self._state)

    @property
    def private_mode(self) -> bool:
        with self._lock:
            return bool(self._state.get("private_mode"))

    @property
    def killed(self) -> bool:
        with self._lock:
            return bool(self._state.get("killed"))

    @property
    def safe_mode(self) -> bool:
        with self._lock:
            return self._state.get("mode") == "SAFE_MODE"

    def set_ready(self) -> dict[str, Any]:
        if self.killed or self.safe_mode:
            return self.status()
        return self._update(mode="READY")

    def set_private_mode(self, enabled: bool) -> dict[str, Any]:
        return self._update(private_mode=bool(enabled))

    def engage_kill_switch(self, reason: str = "user requested") -> dict[str, Any]:
        self.cancel_event.set()
        return self._update(killed=True, mode="STOPPED", kill_reason=reason)

    def reset_kill_switch(self) -> dict[str, Any]:
        self.cancel_event.clear()
        mode = "SAFE_MODE" if self._state.get("safe_mode_reason") else "READY"
        return self._update(killed=False, mode=mode, kill_reason=None)

    def enter_safe_mode(self, reason: str) -> dict[str, Any]:
        return self._update(mode="SAFE_MODE", safe_mode_reason=reason)

    def leave_safe_mode(self) -> dict[str, Any]:
        mode = "STOPPED" if self.killed else "READY"
        return self._update(mode=mode, safe_mode_reason=None)

    def allows_generation(self) -> bool:
        return not self.killed

    def allows_mutation(self) -> bool:
        return not self.killed and not self.safe_mode

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)


@dataclass
class CircuitSnapshot:
    name: str
    state: str
    failures: int
    retry_after_seconds: float


class CircuitBreaker:
    """Capability-scoped closed/open/half-open breaker with cooldown."""

    def __init__(self, name: str, failure_threshold: int = 3, recovery_seconds: float = 30.0):
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_seconds = max(0.1, recovery_seconds)
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_probe = False

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if time.monotonic() - self._opened_at < self.recovery_seconds:
                return False
            if self._half_open_probe:
                return False
            self._half_open_probe = True
            return True

    def success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._half_open_probe = False

    def failure(self) -> None:
        with self._lock:
            self._half_open_probe = False
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()

    def snapshot(self) -> CircuitSnapshot:
        with self._lock:
            if self._opened_at is None:
                state = "closed"
                retry = 0.0
            else:
                retry = max(0.0, self.recovery_seconds - (time.monotonic() - self._opened_at))
                state = "half_open" if self._half_open_probe else "open"
            return CircuitSnapshot(self.name, state, self._failures, retry)


class HealthRegistry:
    """Separates process liveness, semantic readiness, and correctness checks."""

    def __init__(self):
        self._checks: dict[str, tuple[str, Callable[[], Any]]] = {}
        self._lock = threading.Lock()

    def register(self, name: str, level: str, check: Callable[[], Any]) -> None:
        if level not in {"liveness", "readiness", "correctness"}:
            raise ValueError(f"invalid health level: {level}")
        with self._lock:
            self._checks[name] = (level, check)

    def snapshot(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        with self._lock:
            checks = dict(self._checks)
        for name, (level, check) in checks.items():
            started = time.monotonic()
            try:
                detail = check()
                ok = detail if isinstance(detail, bool) else True
                results[name] = {
                    "level": level,
                    "ok": bool(ok),
                    "detail": detail,
                    "latency_ms": round((time.monotonic() - started) * 1000, 2),
                }
            except (LookupError, TypeError, ValueError) as e:
                results[name] = {
                    "level": level,
                    "ok": False,
                    "detail": str(e),
                    "latency_ms": round((time.monotonic() - started) * 1000, 2),
                }
        summary = {
            level: all(v["ok"] for v in results.values() if v["level"] == level)
            for level in ("liveness", "readiness", "correctness")
        }
        return {"summary": summary, "checks": results, "timestamp": time.time()}


class ResourceGovernor:
    """Fail-soft admission control for heavy local capabilities."""

    HEAVY_CAPABILITIES = {"image_generation", "lora_training", "document_index", "vision", "translation"}

    def __init__(self, max_memory_percent: float = 85.0, min_battery_percent: float = 20.0):
        self.max_memory_percent = max_memory_percent
        self.min_battery_percent = min_battery_percent

    @staticmethod
    def _battery() -> tuple[int | None, bool]:
        try:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=3, check=False).stdout
            match = re.search(r"(\d+)%", out)
            return (int(match.group(1)) if match else None, "AC Power" in out)
        except (subprocess.SubprocessError, OSError, ValueError):
            return None, False

    def snapshot(self) -> dict[str, Any]:
        memory = psutil.virtual_memory()
        battery, plugged_in = self._battery()
        load = os.getloadavg()[0]
        return {
            "memory_percent": memory.percent,
            "available_gb": round(memory.available / 1e9, 2),
            "battery_percent": battery,
            "plugged_in": plugged_in,
            "load_1m": round(load, 2),
            "cpu_count": psutil.cpu_count() or 1,
        }

    def admit(self, capability: str) -> tuple[bool, str, dict[str, Any]]:
        state = self.snapshot()
        if capability not in self.HEAVY_CAPABILITIES:
            return True, "routine capability", state
        if state["memory_percent"] >= self.max_memory_percent:
            return False, "deferred because memory pressure is high", state
        battery = state["battery_percent"]
        if battery is not None and battery < self.min_battery_percent and not state["plugged_in"]:
            return False, "deferred until the Mac is plugged in or battery recovers", state
        return True, "admitted", state
