#!/usr/bin/env python3
"""Actor wrapper for the Bad Apple MCP marketplace."""

from __future__ import annotations

from typing import Any

import badapple_actor
import badapple_mcp_marketplace


WHITELIST = frozenset({
    "list_servers",
    "add_server",
    "remove_server",
    "list_tools",
    "invoke",
    "stop_all",
    "install_catalog_server",
    "set_airgap",
    "add_mcp_server",
    "remove_mcp_server",
    "list_mcp_servers",
    "list_mcp_tools",
    "invoke_mcp_tool",
    "mcp_marketplace",
    "mcp_install",
    "install_mcp_server_from_marketplace",
    "marketplace_catalog",
})


class MCPActor(badapple_actor.Actor):
    """Actor that owns the MCPMarketplace."""

    def __init__(self) -> None:
        super().__init__("mcp")
        self._marketplace = badapple_mcp_marketplace.MCPMarketplace()
        # Keep module-level registry in sync so standalone helpers still hit the
        # same in-memory clients.
        badapple_mcp_marketplace._MARKETPLACE = self._marketplace

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
            if method == "set_airgap":
                enabled = bool(args[0] if args else kwargs.get("value"))
                badapple_mcp_marketplace.set_airgap(enabled)
                return None
            if method == "list_servers":
                return self._marketplace.list_servers()
            if method == "add_server":
                return self._marketplace.add_server(args[0], args[1], args[2] if len(args) > 2 else None)
            if method == "remove_server":
                return self._marketplace.remove_server(args[0])
            if method == "list_tools":
                return self._marketplace.list_tools(args[0])
            if method == "invoke":
                return self._marketplace.invoke(args[0], args[1], args[2] if len(args) > 2 else {})
            if method == "stop_all":
                self._marketplace.stop_all()
                return None
            if method == "install_catalog_server":
                return badapple_mcp_marketplace.install_catalog_server(
                    args[0],
                    args[1] if len(args) > 1 else None,
                )
            # Tool-facing aliases that match the module-level convenience API.
            if method == "add_mcp_server":
                return badapple_mcp_marketplace.add_mcp_server(
                    args[0] if args else kwargs.get("name", ""),
                    args[1] if len(args) > 1 else kwargs.get("command", ""),
                    args[2] if len(args) > 2 else kwargs.get("env"),
                )
            if method == "remove_mcp_server":
                return badapple_mcp_marketplace.remove_mcp_server(
                    args[0] if args else kwargs.get("name", "")
                )
            if method == "list_mcp_servers":
                return badapple_mcp_marketplace.list_mcp_servers()
            if method == "list_mcp_tools":
                return badapple_mcp_marketplace.list_mcp_tools(
                    args[0] if args else kwargs.get("server", "")
                )
            if method == "invoke_mcp_tool":
                return badapple_mcp_marketplace.invoke_mcp_tool(
                    args[0] if args else kwargs.get("server", ""),
                    args[1] if len(args) > 1 else kwargs.get("tool", ""),
                    args[2] if len(args) > 2 else kwargs.get("arguments") or {},
                )
            if method in ("mcp_marketplace", "marketplace_catalog"):
                return badapple_mcp_marketplace.marketplace_catalog()
            if method in ("mcp_install", "install_mcp_server_from_marketplace"):
                return badapple_mcp_marketplace.install_mcp_server_from_marketplace(
                    args[0] if args else kwargs.get("name", "")
                )
            return {"error": f"method {method!r} not implemented"}
        except Exception as e:  # noqa: BLE001 - actor boundary
            return {"error": str(e)}


class MCPActorProxy:
    """Synchronous proxy for the MCP marketplace actor."""

    def __init__(self, actor: MCPActor) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> Any:
        if name not in WHITELIST and not name.startswith("_"):
            raise AttributeError(f"MCPActorProxy has no attribute {name!r}")

        def _call(*args: Any, **kwargs: Any) -> Any:
            payload = {"method": name, "args": list(args), "kwargs": kwargs}
            return self._actor.ask(payload, timeout=60.0)

        return _call
