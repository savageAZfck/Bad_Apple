#!/usr/bin/env python3
"""Actor wrapper for the Bad Apple model manager.

This keeps model downloads, cache verification, and provenance checks on a
dedicated actor thread so the main MLX thread is not blocked by disk I/O or
hash computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import badapple_actor
import badapple_model_manager

# Methods that may block for a long time and therefore need longer ask timeouts.
LONG_METHODS = {"wait_for_download", "record_provenance"}

# Whitelisted methods and properties the proxy is allowed to dispatch.
WHITELIST = frozenset({
    "status",
    "list_profiles",
    "set_allow_downloads",
    "allow_downloads",
    "ensure_cached",
    "wait_for_download",
    "mark_loaded",
    "mark_unloaded",
    "refresh_cache_status",
    "background_refresh_all",
    "memory_required",
    "recommend_for_memory",
    "recommend_for_query",
    "preload_priority",
    "start_download",
    "verify_before_load",
    "record_provenance",
    "verify_provenance",
})


class ModelActor(badapple_actor.Actor):
    """Actor that owns the ModelManager and Provenance."""

    def __init__(self, data_dir: Path | None = None) -> None:
        super().__init__("model_manager")
        self._manager = badapple_model_manager.ModelManager(data_dir)

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
        target = getattr(self._manager, method, None)
        if target is None:
            return {"error": f"method {method!r} not found"}
        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        try:
            if callable(target):
                return target(*args, **kwargs)
            # property / simple attribute
            return target
        except Exception as e:  # noqa: BLE001 - actor boundary
            return {"error": str(e)}

    def shutdown(self, timeout: float = 5.0) -> None:
        try:
            self._manager.shutdown()
        except Exception:  # noqa: BLE001,S110 - cleanup
            pass
        super().stop(timeout=timeout)


class ModelActorProxy:
    """Synchronous proxy that exposes the ModelManager API over an actor.

    This allows the rest of the codebase to keep writing `self.model_manager.foo()`
    while the actual work runs on the ModelActor thread.
    """

    def __init__(self, actor: ModelActor) -> None:
        self._actor = actor

    def shutdown(self, timeout: float = 5.0) -> None:
        """Shut down the underlying ModelManager and the actor thread."""
        self._actor.shutdown(timeout=timeout)

    def __getattr__(self, name: str) -> Any:
        if name not in WHITELIST and not name.startswith("_"):
            raise AttributeError(f"ModelActorProxy has no attribute {name!r}")

        def _call(*args: Any, **kwargs: Any) -> Any:
            timeout = 30.0
            if name in LONG_METHODS:
                timeout = max(timeout, kwargs.pop("_actor_timeout", 0))
                timeout = max(timeout, 700.0)
            if name == "wait_for_download":
                timeout = kwargs.pop("_actor_timeout", 700.0)
            payload = {
                "method": name,
                "args": list(args),
                "kwargs": {k: v for k, v in kwargs.items() if not k.startswith("_actor_")},
            }
            return self._actor.ask(payload, timeout=timeout)

        return _call
