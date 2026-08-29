#!/usr/bin/env python3
"""Actor wrapper for the Bad Apple persona pack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import badapple_actor
import badapple_extras


WHITELIST = frozenset({
    "get_system_prompt",
    "get_roast_bank",
    "switch",
    "list_personas",
    "handle_command",
    "reload_personas",
})

PROPERTIES = frozenset({
    "active",
})


class PersonaActor(badapple_actor.Actor):
    """Actor that owns the PersonaPack."""

    def __init__(self, data_dir: Path, prompt_file: Path | None = None, existing: badapple_extras.PersonaPack | None = None) -> None:
        super().__init__("persona")
        if existing is not None:
            self._personas = existing
        else:
            self._personas = badapple_extras.PersonaPack(
                data_dir,
                prompt_file or (data_dir / "prompt.txt"),
            )

    def receive(self, message: Any) -> Any:
        if isinstance(message, badapple_actor.Ask):
            payload = message.payload
        else:
            payload = message
        if not isinstance(payload, dict):
            return None
        method = payload.get("method")
        if method not in WHITELIST and method not in PROPERTIES:
            return {"error": f"method {method!r} not whitelisted"}
        target = getattr(self._personas, method, None)
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


class PersonaActorProxy:
    """Synchronous proxy for the persona actor."""

    def __init__(self, actor: PersonaActor) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> Any:
        if name in PROPERTIES:
            return self._actor.ask({"method": name}, timeout=10.0)
        if name not in WHITELIST and not name.startswith("_"):
            raise AttributeError(f"PersonaActorProxy has no attribute {name!r}")

        def _call(*args: Any, **kwargs: Any) -> Any:
            payload = {"method": name, "args": list(args), "kwargs": kwargs}
            return self._actor.ask(payload, timeout=10.0)

        return _call
