#!/usr/bin/env python3
"""Actor wrapper around the Bad Apple health registry."""

from __future__ import annotations

from typing import Any

import badapple_actor
import badapple_runtime


WHITELIST = frozenset({
    "register",
    "snapshot",
})


class HealthActor(badapple_actor.Actor):
    """Actor that owns the HealthRegistry."""

    def __init__(self) -> None:
        super().__init__("health")
        self._registry = badapple_runtime.HealthRegistry()

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
        target = getattr(self._registry, method, None)
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


class HealthActorProxy:
    """Synchronous proxy for the health actor."""

    def __init__(self, actor: HealthActor) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> Any:
        if name not in WHITELIST and not name.startswith("_"):
            raise AttributeError(f"HealthActorProxy has no attribute {name!r}")

        def _call(*args: Any, **kwargs: Any) -> Any:
            payload = {"method": name, "args": list(args), "kwargs": kwargs}
            return self._actor.ask(payload, timeout=10.0)

        return _call
