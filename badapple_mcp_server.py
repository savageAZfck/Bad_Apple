#!/usr/bin/env python3
"""Minimal local Model Context Protocol (MCP) server for Bad Apple.

- Runs on a Unix domain socket (default ``/var/run/badapple/mcp.sock``).
- Falls back to ``stdio`` JSON-RPC transport when invoked with the ``stdio``
  argument.
- Talks to the Bad Apple MLX daemon through the authenticated SLICKS Unix
  socket protocol via ``agent_client.call_agent``.

Air-gap properties:
- No TCP sockets are opened.
- All external access is mediated by the local ``badapple_mlx_server.py``
  Unix socket.
- Uses only the Python standard library plus the local ``agent_client`` helper.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any

from agent_client import call_agent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "Bad Apple MCP"
SERVER_VERSION = "0.1.0"
DEFAULT_SOCKET_PATH = "/var/run/badapple/mcp.sock"

# Security limits
MAX_REQUEST_BYTES = 64 * 1024
MAX_TOOL_NAME_LEN = 256
MAX_PROMPT_NAME_LEN = 256
MAX_URI_LEN = 1024
MAX_ARGUMENTS_DEPTH = 8
CALL_AGENT_TIMEOUT = 120.0

# Tool calls can be expensive or destructive. By default the MCP server blocks the
# small set of tools that can mutate the system outside the daemon's approval gate.
_DANGEROUS_TOOLS = {
    "run_shell",
    "run_applescript",
    "write_file",
    "delete_file",
    "index_documents",
    "run_shortcut",
}

# Methods that are exposed directly by the Bad Apple LAP daemon.
_DIRECT_AGENT_METHODS = {
    "runtime_status",
    "invoke_tool",
    "p2p_peers",
    "p2p_sync",
    "set_workspace",
    "inference",
}


# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------

def _result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="mcp-agent-caller"
)


def _is_safe_value(value: Any, depth: int = 0) -> bool:
    if depth > MAX_ARGUMENTS_DEPTH:
        return False
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_safe_value(v, depth + 1) for v in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_safe_value(v, depth + 1)
            for k, v in value.items()
        )
    return False


def _is_safe_tool_name(name: str) -> bool:
    return isinstance(name, str) and 0 < len(name) <= MAX_TOOL_NAME_LEN and re.match(
        r"^[A-Za-z0-9_:-]+$", name
    ) is not None


def _is_dangerous_allowed() -> bool:
    return os.environ.get("BADAPPLE_MCP_ALLOW_DANGEROUS", "0") == "1"


def _mcp_text_content(obj: Any, is_error: bool = False) -> dict[str, Any]:
    """Wrap an arbitrary object as a single MCP ``TextContent`` result."""
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2, default=str)
    return {
        "content": [{"type": "text", "text": text}],
        **({"isError": True} if is_error else {}),
    }


def _normalize_response(response: dict[str, Any]) -> tuple[str, bool]:
    """Extract display text and error flag from a daemon response frame."""
    is_error = response.get("type") == "error" or bool(response.get("error"))
    if is_error:
        msg = response.get("message") or response.get("error")
        if isinstance(msg, str):
            return msg, True
    payload = response.get("result", response)
    return json.dumps(payload, indent=2, default=str), is_error


# ---------------------------------------------------------------------------
# Tool discovery / listing
# ---------------------------------------------------------------------------

def _core_tools() -> list[dict[str, Any]]:
    """The conceptual Bad Apple tools the request asked us to expose."""
    return [
        {
            "name": "runtime_status",
            "description": "Return the Bad Apple runtime, health, resource, active model and circuit-breaker status.",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "invoke_tool",
            "description": "Invoke any Bad Apple daemon tool by name and pass an arguments object.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the Bad Apple tool to invoke."},
                    "args": {"type": "object", "description": "Arguments passed to the tool."},
                },
                "required": ["name"],
            },
        },
        {
            "name": "get_ambient_context",
            "description": "Return the latest ambient screen/app context (active app, window, screenshot path).",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "p2p_peers",
            "description": "List Bad Apple peers discovered on the local network via encrypted link-local broadcast.",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "p2p_sync",
            "description": "Trigger a sync of working memory with discovered local Bad Apple peers.",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "set_workspace",
            "description": "Set the active Bad Apple workspace path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the workspace."},
                },
                "required": ["path"],
            },
        },
        {
            "name": "inference",
            "description": "Run local Bad Apple text inference through the authenticated Unix socket.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "max_new_tokens": {"type": "integer", "default": 120},
                },
                "required": ["prompt"],
            },
        },
    ]


def _tools() -> list[dict[str, Any]]:
    """Merge the requested core tools with the daemon's live ``discover_tools`` list."""
    by_name: dict[str, dict[str, Any]] = {t["name"]: t for t in _core_tools()}

    try:
        response = call_agent("discover_tools")
        payload = response.get("result", response)
        schemas = payload.get("tools", []) if isinstance(payload, dict) else []
        for schema in schemas:
            function = schema.get("function", schema)
            name = function.get("name")
            if not name or name in by_name:
                continue
            by_name[name] = {
                "name": name,
                "description": function.get("description", ""),
                "inputSchema": function.get(
                    "parameters",
                    {"type": "object", "properties": {}, "required": []},
                ),
            }
    except (LookupError, TypeError, ValueError) as exc: # pragma: no cover - daemon may be down
        print(f"[mcp] discover_tools failed: {exc}", file=sys.stderr, flush=True)

    return list(by_name.values())


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _call_agent_or_report(method: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Call ``agent_client.call_agent`` and normalize errors, with a timeout."""
    try:
        future = _executor.submit(call_agent, method, params)
        return future.result(timeout=CALL_AGENT_TIMEOUT)
    except concurrent.futures.TimeoutError:
        return {"type": "error", "error": f"agent_client call to {method} timed out after {CALL_AGENT_TIMEOUT}s"}
    except Exception as exc:  # noqa: BLE001 - catch-all wrapper
        return {"type": "error", "error": f"agent_client error: {exc}"}


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if not _is_safe_tool_name(name):
        return _mcp_text_content(
            f"Invalid or unsupported tool name: {name!r}",
            is_error=True,
        )
    if not isinstance(arguments, dict):
        arguments = {}
    if not _is_safe_value(arguments):
        return _mcp_text_content(
            "Tool arguments contain unsafe types or nested too deeply",
            is_error=True,
        )
    if name in _DANGEROUS_TOOLS and not _is_dangerous_allowed():
        return _mcp_text_content(
            f"Tool {name!r} is disabled over MCP. Set BADAPPLE_MCP_ALLOW_DANGEROUS=1 to allow.",
            is_error=True,
        )

    # Map the requested conceptual tool names onto the LAP protocol.
    if name == "get_ambient_context":
        response = _call_agent_or_report("invoke_tool", {"name": "ambient_context", "args": arguments})
    elif name in _DIRECT_AGENT_METHODS:
        response = _call_agent_or_report(name, arguments)
    else:
        # Passthrough for any other Bad Apple tool discovered at runtime.
        response = _call_agent_or_report("invoke_tool", {"name": name, "args": arguments})

    text, is_error = _normalize_response(response)
    return _mcp_text_content(text, is_error)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def _resources() -> list[dict[str, Any]]:
    return [
        {
            "uri": "badapple://status",
            "name": "Bad Apple Status",
            "description": "Runtime, health, resource, active model and circuit-breaker snapshot.",
            "mimeType": "application/json",
        },
        {
            "uri": "badapple://ambient",
            "name": "Ambient Context",
            "description": "Latest ambient screen/app context including active app and window.",
            "mimeType": "application/json",
        },
        {
            "uri": "badapple://workspace",
            "name": "Workspace",
            "description": "Current Bad Apple workspace path and summary.",
            "mimeType": "application/json",
        },
        {
            "uri": "badapple://peers",
            "name": "P2P Peers",
            "description": "Discovered local Bad Apple peers.",
            "mimeType": "application/json",
        },
    ]


def _read_resource(uri: str) -> dict[str, Any]:
    if uri == "badapple://status":
        response = _call_agent_or_report("runtime_status", {})
    elif uri == "badapple://ambient":
        response = _call_agent_or_report("invoke_tool", {"name": "ambient_context", "args": {}})
    elif uri == "badapple://workspace":
        response = _call_agent_or_report("get_workspace", {})
    elif uri == "badapple://peers":
        response = _call_agent_or_report("p2p_peers", {})
    else:
        return {"contents": []}

    text, is_error = _normalize_response(response)
    return {
        "contents": [
            {
                "uri": uri,
                "mimeType": "application/json",
                "text": text,
            }
        ],
        **({"isError": True} if is_error else {}),
    }


# ---------------------------------------------------------------------------
# Prompts (optional / minimal)
# ---------------------------------------------------------------------------

def _prompts() -> list[dict[str, Any]]:
    return []


def _get_prompt(name: str) -> dict[str, Any]:
    if name == "badapple-inference":
        return {
            "description": "Run a prompt through local Bad Apple inference.",
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": "Please run Bad Apple inference."},
                }
            ],
        }
    raise ValueError(f"prompt not found: {name}")


# ---------------------------------------------------------------------------
# Request dispatch
# ---------------------------------------------------------------------------

def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    """Process one JSON-RPC/MCP request and return a response, or ``None`` for
    notifications that do not require a response."""
    request_id = request.get("id")
    method = request.get("method")
    if not isinstance(method, str):
        return _error(request_id, -32600, "method must be a string")
    params = request.get("params") or {}
    if not isinstance(params, dict):
        return _error(request_id, -32602, "params must be an object")

    if method == "initialize":
        return _result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return _result(request_id, {})

    if method == "tools/list":
        return _result(request_id, {"tools": _tools()})

    if method == "tools/call":
        name = params.get("name", "")
        if not _is_safe_tool_name(name):
            return _error(request_id, -32602, "invalid tool name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _error(request_id, -32602, "arguments must be an object")
        return _result(request_id, _call_tool(name, arguments))

    if method == "resources/list":
        return _result(request_id, {"resources": _resources()})

    if method == "resources/read":
        uri = params.get("uri", "")
        if not isinstance(uri, str) or not uri or len(uri) > MAX_URI_LEN:
            return _error(request_id, -32602, "invalid uri")
        return _result(request_id, _read_resource(uri))

    if method == "prompts/list":
        return _result(request_id, {"prompts": _prompts()})

    if method == "prompts/get":
        name = params.get("name", "")
        if not isinstance(name, str) or not name or len(name) > MAX_PROMPT_NAME_LEN:
            return _error(request_id, -32602, "invalid prompt name")
        try:
            return _result(request_id, _get_prompt(name))
        except ValueError as exc:
            return _error(request_id, -32602, str(exc))

    return _error(request_id, -32601, f"method not found: {method}")


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------

def _write_line(stream, obj: dict[str, Any]) -> None:
    stream.write(json.dumps(obj, separators=(",", ":")) + "\n")
    stream.flush()


class _MCPStreamRequestHandler(socketserver.StreamRequestHandler):
    """One connection handler per Unix socket client; handles multiple requests."""

    def handle(self) -> None:
        for raw in self.rfile:
            if len(raw) > MAX_REQUEST_BYTES:
                response = _error(None, -32600, f"request exceeded {MAX_REQUEST_BYTES} bytes")
                self.wfile.write(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")
                self.wfile.flush()
                continue
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = handle(request)
            except json.JSONDecodeError as exc:
                response = _error(None, -32700, f"invalid JSON: {exc}")
            except (TypeError, ValueError, AttributeError) as exc:
                response = _error(None, -32603, f"internal error: {exc}")

            if response is not None:
                self.wfile.write(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")
                self.wfile.flush()


def _run_stdio_server() -> int:
    """Read newline-delimited JSON-RPC from stdin and write to stdout."""
    for raw in sys.stdin:
        if len(raw) > MAX_REQUEST_BYTES:
            _write_line(sys.stdout, _error(None, -32600, f"request exceeded {MAX_REQUEST_BYTES} bytes"))
            continue
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle(request)
        except json.JSONDecodeError as exc:
            response = _error(None, -32700, f"invalid JSON: {exc}")
        except (TypeError, ValueError, AttributeError) as exc:
            response = _error(None, -32603, f"internal error: {exc}")

        if response is not None:
            _write_line(sys.stdout, response)

    return 0


def _prepare_socket_path(path: str) -> None:
    """Create parent directory and remove stale socket files."""
    parent = Path(path).parent
    if parent:
        parent.mkdir(parents=True, exist_ok=True)

    if Path(path).exists():
        try:
            Path(path).unlink()
        except OSError as exc:
            print(f"[mcp] cannot remove stale socket {path}: {exc}", file=sys.stderr, flush=True)
            raise


def _run_unix_server() -> int:
    """Run the MCP server on a Unix domain socket."""
    socket_path = os.environ.get("BADAPPLE_MCP_SOCKET", DEFAULT_SOCKET_PATH)

    try:
        _prepare_socket_path(socket_path)
    except Exception as exc:  # noqa: BLE001 - catch-all wrapper
        print(f"[mcp] socket setup failed: {exc}", file=sys.stderr, flush=True)
        return 1

    server = socketserver.ThreadingUnixStreamServer(socket_path, _MCPStreamRequestHandler)
    server.daemon_threads = True

    # Run the server in a daemon thread so the main thread can wait for signals.
    server_thread = threading.Thread(target=server.serve_forever, name="mcp-server", daemon=True)
    server_thread.start()
    print(f"[mcp] listening on {socket_path}", file=sys.stderr, flush=True)

    stop_event = threading.Event()

    def _shutdown(_signum=None, _frame=None) -> None:
        if not stop_event.is_set():
            stop_event.set()

    try:
        import signal

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
    except Exception:  # noqa: BLE001,S110 - cleanup
        # Signals may not be available on all platforms.
        pass

    try:
        while not stop_event.is_set() and server_thread.is_alive():
            stop_event.wait(0.5)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        print("[mcp] shutting down...", file=sys.stderr, flush=True)
        server.shutdown()
        server.server_close()
        try:
            if Path(socket_path).exists():
                Path(socket_path).unlink()
        except OSError:
            pass
        server_thread.join(timeout=2.0)

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("stdio", "--stdio"):
        return _run_stdio_server()
    return _run_unix_server()


if __name__ == "__main__":
    sys.exit(main())
