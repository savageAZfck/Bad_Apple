#!/usr/bin/env python3
"""In-memory metrics collector for Bad Apple inference turns.

Stores the last N turn metrics and provides rolling aggregates. Data stays in
memory and is not sent to the cloud.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from typing import Any


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
