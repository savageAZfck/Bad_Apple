#!/usr/bin/env python3
"""Actor wrapper for the Bad Apple resource governor."""

from __future__ import annotations

from typing import Any

import badapple_actor
import badapple_runtime


WHITELIST = frozenset({
    "admit",
    "snapshot",
})


class ResourcesActor(badapple_actor.Actor):
    """Actor that owns the ResourceGovernor."""

    def __init__(self) -> None:
        super().__init__("resources")
        self._governor = badapple_runtime.ResourceGovernor()

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
        target = getattr(self._governor, method, None)
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


class ResourcesActorProxy:
    """Synchronous proxy for the resource governor actor."""

    def __init__(self, actor: ResourcesActor) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> Any:
        if name not in WHITELIST and not name.startswith("_"):
            raise AttributeError(f"ResourcesActorProxy has no attribute {name!r}")

        def _call(*args: Any, **kwargs: Any) -> Any:
            payload = {"method": name, "args": list(args), "kwargs": kwargs}
            return self._actor.ask(payload, timeout=10.0)

        return _call
