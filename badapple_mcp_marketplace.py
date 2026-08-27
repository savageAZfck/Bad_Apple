#!/usr/bin/env python3
"""Experimental MCP (Model Context Protocol) client for Bad Apple.

This is a minimal stdio-based MCP client that can:
- Maintain a registry of local MCP servers (command + args + env).
- Start/stop a server via subprocess.
- Initialize, list tools, and call tools over JSON-RPC.

It is intentionally small and runs entirely on the user's Mac without any
cloud dependency.  Servers must be installed locally (npx, uvx, python, etc.).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


_extra_path = ":".join([
    "/usr/local/bin",
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
])


def _resolve_command(command: list[str]) -> list[str]:
    """Expand ~ and resolve the executable in common bin paths if needed."""
    if not command:
        return command
    home = Path.home()
    expanded: list[str] = []
    for c in command:
        if c == "~":
            expanded.append(str(home))
        elif c.startswith("~/"):
            expanded.append(str(home / c[2:]))
        else:
            expanded.append(c)
    exe = expanded[0]
    if exe and not Path(exe).is_absolute():
        search_path = os.environ.get("PATH", "") + ":" + _extra_path
        resolved = shutil.which(exe, path=search_path)
        if resolved:
            expanded[0] = resolved
    return expanded


def _is_safe_name(name: str) -> bool:
    return bool(SAFE_NAME_RE.match(name)) and len(name) <= 64

DEFAULT_REGISTRY = Path("/var/lib/bad_apple/mcp_servers.json")
DEFAULT_TIMEOUT = 30
DEFAULT_CATALOG = Path(__file__).with_name("mcp_registry.json")


def _load_registry() -> dict[str, Any]:
    if DEFAULT_REGISTRY.is_file():
        try:
            return json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
            print(f"[mcp_marketplace] could not load registry: {e}", flush=True)
    return {"servers": []}


def _save_registry(data: dict[str, Any]) -> None:
    DEFAULT_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_REGISTRY, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog_path = path or DEFAULT_CATALOG
    if catalog_path.is_file():
        try:
            return json.loads(catalog_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
            print(f"[mcp_marketplace] could not load catalog {catalog_path}: {e}", flush=True)
    return {"servers": []}


def _install_pip_server(entry: dict[str, Any]) -> list[str]:
    pkg = entry.get("pip_package") or entry.get("name")
    name = entry["name"]
    venv_dir = Path.home() / ".bad_apple" / "mcp_venvs" / name
    venv_dir.mkdir(parents=True, exist_ok=True)
    python = venv_dir / "bin" / "python"
    if not python.is_file():
        subprocess.run([sys.executable or "python3", "-m", "venv", str(venv_dir)], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-q", pkg], check=True)

    command = list(entry.get("command", [str(python), "-m", pkg.replace("-", "_")]))
    # Prefer a console script entry point if it exists (e.g. mcp-server-time, mcp-server-sqlite).
    entry_point = venv_dir / "bin" / pkg
    if entry_point.is_file():
        # Drop the common [python, -m, module] prefix and keep the remaining args.
        if len(command) >= 3 and command[0] in ("python", str(python)) and command[1] == "-m" and command[2] == pkg.replace("-", "_"):
            args = command[3:]
        else:
            args = command[1:]
        command = [str(entry_point), *args]

    # Expand any ~ or $HOME in command strings for this venv.
    expanded = []
    for c in command:
        if c == "python":
            expanded.append(str(python))
        elif c == "~":
            expanded.append(str(Path.home()))
        elif c.startswith("~/"):
            expanded.append(str(Path.home() / c[2:]))
        else:
            expanded.append(c)
    return expanded


def install_catalog_server(name: str, catalog_path: Path | None = None) -> str:
    if not _is_safe_name(name):
        return "MCP server name must be alphanumeric, hyphens or underscores."
    catalog = _load_catalog(catalog_path)
    for entry in catalog.get("servers", []):
        if entry.get("name") == name:
            install_type = entry.get("install_type", "command")
            if install_type == "pip":
                command = _install_pip_server(entry)
            elif install_type == "command":
                command = _resolve_command(list(entry.get("command", [])))
                if not command:
                    return f"MCP server '{name}' has no command in the catalog."
            else:
                return f"MCP server '{name}' has unknown install_type '{install_type}'."
            return _MARKETPLACE.add_server(name, command, entry.get("env"))
    return f"MCP server '{name}' not found in catalog."


def list_catalog_servers(catalog_path: Path | None = None) -> list[dict[str, Any]]:
    return _load_catalog(catalog_path).get("servers", [])


class MCPClient:
    """A lightweight stdio MCP client."""

    def __init__(self, name: str, command: list[str], env: dict[str, str] | None = None):
        self.name = name
        self.command = _resolve_command(command)
        self.env = env or {}
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._next_id = 1
        self._pending: dict[str, Any] = {}
        self._reader: threading.Thread | None = None
        self._tools: list[dict[str, Any]] = []

    def _next_jsonrpc_id(self) -> int:
        with self._lock:
            i = self._next_id
            self._next_id += 1
            return i

    def _send_line(self, obj: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("MCP server is not running")
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        self._proc.stdin.flush()

    def _reader_loop(self) -> None:
        while True:
            try:
                line = self._proc.stdout.readline()
            except Exception:  # noqa: BLE001 - catch-all wrapper
                break
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in msg:
                with self._lock:
                    event = self._pending.pop(str(msg["id"]), None)
                if event is not None:
                    event["msg"] = msg
                    event["done"].set()

    def _call(self, method: str, params: dict[str, Any] | None = None, timeout: int = DEFAULT_TIMEOUT) -> Any:
        req_id = self._next_jsonrpc_id()
        req = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            req["params"] = params
        done = threading.Event()
        event = {"done": done, "msg": None}
        with self._lock:
            self._pending[str(req_id)] = event
        self._send_line(req)
        if not done.wait(timeout=timeout):
            with self._lock:
                self._pending.pop(str(req_id), None)
            raise TimeoutError(f"MCP call {method} timed out")
        msg = event["msg"] or {}
        if "error" in msg:
            raise RuntimeError(msg["error"].get("message", str(msg["error"])))
        return msg.get("result")

    def start(self) -> bool:
        env = os.environ.copy()
        env["PATH"] = (env.get("PATH", "") + ":" + _extra_path).strip(":")
        env.update(self.env)
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=False,
            )
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            print(f"[mcp_client] could not start {self.name}: {e}", flush=True)
            return False
        self._reader = threading.Thread(target=self._reader_loop, name=f"mcp-{self.name}-reader", daemon=True)
        self._reader.start()
        time.sleep(0.5)
        if self._proc and self._proc.poll() is not None:
            self._reader.join(timeout=1)
            self._proc = None
            print(f"[mcp_client] could not start {self.name}: process exited with code {self._proc} before initialize", flush=True)
            return False
        try:
            self._call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "badapple", "version": "0.1"}}, timeout=60)
            self._send_line({"jsonrpc": "2.0", "method": "notifications/initialized"})
            tools = self._call("tools/list", timeout=30)
            self._tools = tools.get("tools", []) if isinstance(tools, dict) else []
            return True
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            self.stop()
            print(f"[mcp_client] init failed for {self.name}: {e}", flush=True)
            return False

    def stop(self) -> None:
        with self._lock:
            for event in self._pending.values():
                event["done"].set()
            self._pending.clear()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:  # noqa: BLE001,S110 - cleanup
                pass
            self._proc = None

    def tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    def call_tool(self, tool_name: str, arguments: dict[str, Any], timeout: int = 60) -> Any:
        return self._call("tools/call", {"name": tool_name, "arguments": arguments}, timeout=timeout)


class MCPMarketplace:
    """Registry + active clients for local MCP servers."""

    def __init__(self):
        self._clients: dict[str, MCPClient] = {}
        self._lock = threading.RLock()

    def list_servers(self) -> list[dict[str, Any]]:
        data = _load_registry()
        return data.get("servers", [])

    def add_server(self, name: str, command: list[str], env: dict[str, str] | None = None) -> str:
        if not _is_safe_name(name):
            return "MCP server name must be alphanumeric, hyphens or underscores."
        data = _load_registry()
        servers = data.get("servers", [])
        for s in servers:
            if s["name"] == name:
                return f"MCP server '{name}' already exists."
        servers.append({"name": name, "command": command, "env": env or {}})
        data["servers"] = servers
        _save_registry(data)
        return f"Added MCP server '{name}'."

    def remove_server(self, name: str) -> str:
        if not _is_safe_name(name):
            return "MCP server name must be alphanumeric, hyphens or underscores."
        data = _load_registry()
        servers = [s for s in data.get("servers", []) if s["name"] != name]
        data["servers"] = servers
        _save_registry(data)
        with self._lock:
            client = self._clients.pop(name, None)
        if client:
            client.stop()
        return f"Removed MCP server '{name}'."

    def get_client(self, name: str) -> MCPClient | None:
        if not _is_safe_name(name):
            return None
        with self._lock:
            client = self._clients.get(name)
            if client is not None:
                return client
            for s in self.list_servers():
                if s["name"] == name:
                    client = MCPClient(name, s["command"], s.get("env"))
                    if client.start():
                        self._clients[name] = client
                        return client
                    return None
        return None

    def list_tools(self, name: str) -> list[dict[str, Any]]:
        if not _is_safe_name(name):
            return []
        client = self.get_client(name)
        if client is None:
            return []
        return client.tools()

    def invoke(self, server: str, tool: str, arguments: dict[str, Any]) -> str:
        if not _is_safe_name(server):
            return f"Error: invalid MCP server name '{server}'."
        client = self.get_client(server)
        if client is None:
            return f"Error: MCP server '{server}' not found or failed to start."
        try:
            result = client.call_tool(tool, arguments)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as e:
            return f"MCP tool error: {e}"

    def stop_all(self) -> None:
        with self._lock:
            for client in list(self._clients.values()):
                client.stop()
            self._clients.clear()


_MARKETPLACE = MCPMarketplace()


def add_mcp_server(name: str, command: str, env: dict[str, str] | None = None) -> str:
    """command is a shell-style string; split with shell semantics."""
    if not _is_safe_name(name):
        return "MCP server name must be alphanumeric, hyphens or underscores."
    import shlex
    return _MARKETPLACE.add_server(name, shlex.split(command), env)


def remove_mcp_server(name: str) -> str:
    if not _is_safe_name(name):
        return "MCP server name must be alphanumeric, hyphens or underscores."
    return _MARKETPLACE.remove_server(name)


def list_mcp_servers() -> str:
    servers = _MARKETPLACE.list_servers()
    if not servers:
        return "No MCP servers registered."
    return "\n".join(f"- {s['name']}: {' '.join(s['command'])}" for s in servers)


def list_mcp_tools(server: str) -> str:
    if not _is_safe_name(server):
        return f"MCP server name '{server}' is invalid."
    tools = _MARKETPLACE.list_tools(server)
    if not tools:
        return f"No tools from MCP server '{server}'."
    return "\n".join(f"- {t['name']}: {t.get('description', '')}" for t in tools)


def invoke_mcp_tool(server: str, tool: str, arguments: dict[str, Any]) -> str:
    if not _is_safe_name(server):
        return f"MCP server name '{server}' is invalid."
    if not _is_safe_name(tool):
        return f"MCP tool name '{tool}' is invalid."
    return _MARKETPLACE.invoke(server, tool, arguments)


def stop_all_mcp_servers() -> None:
    _MARKETPLACE.stop_all()


def _mcp_data_dir() -> Path:
    """Return a user-writable MCP data directory."""
    path = Path.home() / ".bad_apple" / "mcp_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mcp_workspace_dir() -> Path:
    """Return the default user workspace for the filesystem MCP server."""
    path = Path.home() / "Documents"
    return path


def marketplace_catalog() -> str:
    """Return a curated list of local MCP servers the user can install."""
    catalog = _load_catalog(Path(__file__).with_name("mcp_registry.json"))
    lines = []
    for c in catalog.get("servers", []):
        cmd = _resolve_command(list(c.get("command", [])))
        install = c.get("install_note") or (cmd and " ".join(cmd)) or f"add mcp server {c['name']}"
        lines.append(f"- {c['name']}: {c['description']}\n  install: {install}")
    if not lines:
        return "No MCP servers are currently available."
    return "Available MCP servers:\n" + "\n".join(lines)


def install_mcp_server_from_marketplace(name: str) -> str:
    """Install a server from the built-in catalog by name."""
    return install_catalog_server(name, catalog_path=Path(__file__).with_name("mcp_registry.json"))
