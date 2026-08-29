#!/usr/bin/env python3
"""Actor wrapper for the Bad Apple semantic cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import badapple_actor
import badapple_extras

WHITELIST = frozenset({
    "lookup",
    "store",
    "reset",
    "classify_intent",
    "threshold",
    "clear",
})


class CacheActor(badapple_actor.Actor):
    """Actor that owns the SemanticCache."""

    def __init__(self, data_dir: Path | None = None, threshold: float | None = None) -> None:
        super().__init__("cache")
        kwargs: dict[str, Any] = {}
        if data_dir is not None:
            kwargs["data_dir"] = data_dir
        if threshold is not None:
            kwargs["threshold"] = threshold
        self._cache = badapple_extras.SemanticCache(**kwargs)

    def receive(self, message: Any) -> Any:
        if isinstance(message, badapple_actor.Ask):
            payload = message.payload
        else:
            payload = message
        if not isinstance(payload, dict):
            return None
        method = payload.get("method")
        if method not in WHITELIST:
            return {"error": f"method {method!r} not whitelisted"}
        target = getattr(self._cache, method, None)
        if target is None:
            return {"error": f"method {method!r} not found"}
        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        try:
            if callable(target):
                return target(*args, **kwargs)
            return target
        except Exception as e:  # noqa: BLE001 - actor boundary
            return {"error": str(e)}


class CacheActorProxy:
    """Synchronous proxy for the cache actor."""

    def __init__(self, actor: CacheActor) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> Any:
        if name not in WHITELIST and not name.startswith("_"):
            raise AttributeError(f"CacheActorProxy has no attribute {name!r}")

        def _call(*args: Any, **kwargs: Any) -> Any:
            payload = {"method": name, "args": list(args), "kwargs": kwargs}
            return self._actor.ask(payload, timeout=30.0)

        return _call
