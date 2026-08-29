#!/usr/bin/env python3
"""Actor wrapper for the Bad Apple circuit breakers."""

from __future__ import annotations

from typing import Any

import badapple_actor
import badapple_runtime

WHITELIST = frozenset({
    "allow",
    "success",
    "failure",
    "snapshot",
    "snapshot_all",
    "names",
    "state",
})


class BreakersActor(badapple_actor.Actor):
    """Actor that owns the collection of CircuitBreaker instances."""

    def __init__(self, names: tuple[str, ...] | None = None) -> None:
        super().__init__("breakers")
        if names is None:
            names = (
                "main_model",
                "dflash",
                "embedding",
                "tts",
                "tools",
                "ledger",
                "p2p",
            )
        self._breakers: dict[str, badapple_runtime.CircuitBreaker] = {
            name: badapple_runtime.CircuitBreaker(name) for name in names
        }

    def _breaker(self, name: str) -> badapple_runtime.CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = badapple_runtime.CircuitBreaker(name)
        return self._breakers[name]

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
        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        try:
            if method in {"allow", "success", "failure", "snapshot"}:
                name = args[0] if args else kwargs.get("name")
                if not isinstance(name, str):
                    return {"error": "breaker name required"}
                breaker = self._breaker(name)
                target = getattr(breaker, method)
                return target() if callable(target) else target
            if method == "snapshot_all":
                return {
                    name: vars(breaker.snapshot())
                    for name, breaker in self._breakers.items()
                }
            if method == "names":
                return list(self._breakers.keys())
            if method == "state":
                return {
                    name: vars(breaker.snapshot())
                    for name, breaker in self._breakers.items()
                }
            return {"error": f"method {method!r} not implemented"}
        except Exception as e:  # noqa: BLE001 - actor boundary
            return {"error": str(e)}


class BreakersActorProxy:
    """Synchronous proxy for the breakers actor."""

    def __init__(self, actor: BreakersActor) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> Any:
        if name not in WHITELIST and not name.startswith("_"):
            raise AttributeError(f"BreakersActorProxy has no attribute {name!r}")

        def _call(*args: Any, **kwargs: Any) -> Any:
            payload = {"method": name, "args": list(args), "kwargs": kwargs}
            return self._actor.ask(payload, timeout=10.0)

        return _call
