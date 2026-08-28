#!/usr/bin/env python3
"""Actor wrapper around the hash-chained audit ledger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import badapple_actor
import badapple_extras


class AuditActor(badapple_actor.Actor):
    """Actor that owns the append-only audit ledger.

    Messages:
        {"method": "record", "event_type": str, "data": Any} -> None
        {"method": "verify"} -> list[dict]
        {"method": "ledger_path"} -> str
    """

    def __init__(self, data_dir: Path | None = None, existing: badapple_extras.AuditLedger | None = None) -> None:
        super().__init__("audit")
        self._ledger = existing or badapple_extras.AuditLedger(data_dir or Path("/var/lib/bad_apple"))

    def receive(self, message: Any) -> Any:
        if isinstance(message, badapple_actor.Ask):
            payload = message.payload
        else:
            payload = message
        if not isinstance(payload, dict):
            return None
        method = payload.get("method")
        if method == "record":
            self._ledger.record(payload.get("event_type", "unknown"), payload.get("data"))
            return None
        if method == "verify":
            return self._ledger.verify()
        if method == "ledger_path":
            return str(self._ledger.ledger_path)
        return None
