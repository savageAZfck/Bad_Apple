#!/usr/bin/env python3
"""In-memory metrics collector for Bad Apple inference turns.

Stores the last N turn metrics and provides rolling aggregates. Data stays in
memory and is not sent to the cloud.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import badapple_actor

MAX_ENTRIES = 1_000


class MetricsCollector:
    """Thread-safe ring buffer of per-turn inference metrics."""

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._entries: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def record(self, **kwargs: Any) -> None:
        """Record a single turn."""
        entry = {"timestamp": time.time(), **kwargs}
        with self._lock:
            self._entries.append(entry)

    def recent(self, n: int = 100) -> list[dict[str, Any]]:
        """Return the most recent n entries (newest last)."""
        with self._lock:
            return list(self._entries)[-n:]

    def aggregates(self, n: int = 100) -> dict[str, Any]:
        """Return rolling averages over the last n entries."""
        with self._lock:
            entries = list(self._entries)[-n:]
        if not entries:
            return {"count": 0}

        numeric: dict[str, list[float]] = {}
        for e in entries:
            for k, v in e.items():
                if k == "timestamp" or not isinstance(v, (int, float)):
                    continue
                numeric.setdefault(k, []).append(float(v))

        result: dict[str, Any] = {"count": len(entries)}
        for k, vals in numeric.items():
            result[f"{k}_avg"] = round(sum(vals) / len(vals), 4)
            result[f"{k}_min"] = round(min(vals), 4)
            result[f"{k}_max"] = round(max(vals), 4)

        return result

    def summary(self) -> dict[str, Any]:
        return {
            "count": len(self._entries),
            "max_entries": self._max_entries,
            "aggregates": self.aggregates(),
            "recent": self.recent(10),
        }


class MetricsActor(badapple_actor.Actor):
    """Actor wrapper around MetricsCollector.

    Messages:
        {"method": "record", **metrics} -> None
        {"method": "recent", "n": int} -> list
        {"method": "summary"} -> dict
    """

    def __init__(self, existing_collector: MetricsCollector | None = None, max_entries: int = MAX_ENTRIES) -> None:
        super().__init__("metrics")
        self._collector = existing_collector or MetricsCollector(max_entries=max_entries)

    def receive(self, message: Any) -> Any:
        if isinstance(message, badapple_actor.Ask):
            payload = message.payload
        else:
            payload = message
        if not isinstance(payload, dict):
            return None
        method = payload.get("method")
        if method == "record":
            entry = {k: v for k, v in payload.items() if k != "method"}
            self._collector.record(**entry)
            return None
        if method == "recent":
            return self._collector.recent(payload.get("n", 100))
        if method == "summary":
            return self._collector.summary()
        if method == "aggregates":
            return self._collector.aggregates(payload.get("n", 100))
        return None
