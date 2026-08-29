#!/usr/bin/env python3
"""Actor wrapper around the Bad Apple agent task manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import badapple_actor
import badapple_agent_tasks


WHITELIST = frozenset({
    "create",
    "get",
    "list",
    "status",
    "update_status",
    "add_step",
    "cancel",
    "pause",
    "resume",
    "delete",
    "register_callback",
})


class TaskActor(badapple_actor.Actor):
    """Actor that owns the AgentTaskManager."""

    def __init__(self, data_dir: Path | None = None) -> None:
        super().__init__("agent_task_manager")
        self._manager = badapple_agent_tasks.AgentTaskManager(data_dir)

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
            return target
        except Exception as e:  # noqa: BLE001 - actor boundary
            return {"error": str(e)}


class TaskActorProxy:
    """Synchronous proxy for the task actor."""

    def __init__(self, actor: TaskActor) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> Any:
        if name not in WHITELIST and not name.startswith("_"):
            raise AttributeError(f"TaskActorProxy has no attribute {name!r}")

        def _call(*args: Any, **kwargs: Any) -> Any:
            payload = {"method": name, "args": list(args), "kwargs": kwargs}
            return self._actor.ask(payload, timeout=30.0)

        return _call
