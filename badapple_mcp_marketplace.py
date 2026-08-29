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


# Safety limits and helpers.
MAX_JSON_BYTES = 1_048_576
MAX_JSON_LINE_BYTES = 10 * 1_048_576
MAX_SERVER_NAME_LEN = 64
MAX_CATALOG_SERVERS = 256
MAX_COMMAND_ARGS = 64
MAX_ARG_LEN = 4096
MAX_ENV_VARS = 64
MAX_ENV_LEN = 4096
MAX_DESCRIPTION_LEN = 4096

_EXECUTABLE_ALLOW_BASES = tuple(
    {Path(p).resolve() for p in (
        "/usr/local",
        "/usr/local/bin",
        "/opt/homebrew",
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/Library/Frameworks",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    )}
)

_DANGEROUS_SHELLS = frozenset(
    {
        "sh",
        "bash",
        "zsh",
        "csh",
        "tcsh",
        "fish",
        "cmd",
        "cmd.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
    }
)
_DANGEROUS_ENV_KEYS = frozenset(
    {
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "PYTHONPATH",
        "PATH",
    }
)

_BASE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_NPM_PACKAGE_RE = re.compile(r"^@[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SAFE_NON_PATH_RE = re.compile(r"^[A-Za-z0-9_.~=@=:-]+$")
_SAFE_PATH_CHARS_RE = re.compile(r"^[A-Za-z0-9_.~/-]+$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SecurityError(ValueError):
    """Raised when an input fails a security check."""


_extra_path = ":".join([
    "/usr/local/bin",
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
])

# Tool names that mutate the filesystem across MCP servers.
_MCP_WRITE_TOOL_PATTERNS = ("write", "edit", "create", "move", "delete", "remove", "update", "append")

# Servers that reach the network. These are blocked when air-gap mode is on.
_NETWORK_MCP_SERVERS = {"fetch", "brave", "puppeteer"}

# Runtime air-gap flag.
_AIRGAP = os.environ.get("BADAPPLE_AIRGAP", "0") == "1"


def set_airgap(enabled: bool) -> None:
    """Enable or disable air-gap mode for MCP tool use."""
    global _AIRGAP
    _AIRGAP = bool(enabled)


def is_airgap() -> bool:
    """Return True if the MCP marketplace is currently in air-gap mode."""
    return _AIRGAP


def is_mcp_write_tool(server: str, tool_name: str) -> bool:
    """Return True if the named MCP tool is considered a destructive/mutative action."""
    low = tool_name.lower()
    return any(p in low for p in _MCP_WRITE_TOOL_PATTERNS)


def _is_safe_name(name: str) -> bool:
    return (
        isinstance(name, str)
        and bool(SAFE_NAME_RE.match(name))
        and len(name) <= MAX_SERVER_NAME_LEN
    )


def _is_path_arg(arg: str) -> bool:
    """Return True when an argument looks like a filesystem path."""
    if not isinstance(arg, str):
        return False
    if ".." in arg:
        return True
    if arg.startswith(("~", ".", os.sep, "\\")) or Path(arg).is_absolute():
        return True
    if os.sep in arg and not _NPM_PACKAGE_RE.match(arg):
        return True
    return False


def _is_safe_path_arg(arg: str, base: Path | None = None) -> bool:
    """Return True when a path argument stays inside the allowed base (default user home)."""
    if not isinstance(arg, str) or len(arg) > MAX_ARG_LEN:
        return False
    if not _SAFE_PATH_CHARS_RE.match(arg):
        return False
    if base is None:
        base = Path.home().resolve()
    else:
        base = base.resolve()
    try:
        p = Path(arg)
        if p.is_absolute() or str(p).startswith("~"):
            expanded = p.expanduser().resolve()
        else:
            expanded = (base / p).expanduser().resolve()
    except (OSError, ValueError):
        return False
    try:
        expanded.relative_to(base)
        return True
    except ValueError:
        return False


def _is_allowed_executable_path(arg: str) -> bool:
    """Return True when an executable path is under the user home or system bin directories."""
    if not isinstance(arg, str) or len(arg) > MAX_ARG_LEN:
        return False
    if not _SAFE_PATH_CHARS_RE.match(arg):
        return False
    home = Path.home().resolve()
    try:
        p = Path(arg)
        if p.is_absolute() or str(p).startswith("~"):
            resolved = p.expanduser().resolve()
        else:
            resolved = (home / p).expanduser().resolve()
    except (OSError, ValueError):
        return False
    for base in (home,) + _EXECUTABLE_ALLOW_BASES:
        try:
            resolved.relative_to(base)
            return True
        except ValueError:
            continue
    return False


def _is_safe_base_name(arg: str) -> bool:
    return (
        isinstance(arg, str)
        and len(arg) <= MAX_ARG_LEN
        and bool(_BASE_NAME_RE.match(arg))
        and arg not in _DANGEROUS_SHELLS
    )


def _is_dangerous_executable(name: str, command: list[str]) -> bool:
    low = name.lower()
    if low in _DANGEROUS_SHELLS:
        return True
    if low.startswith(("python", "python3")) and "-c" in command[1:]:
        return True
    return False


def _has_shell_metacharacters(arg: str) -> bool:
    """Return True when a non-path argument contains characters we treat as unsafe."""
    if _NPM_PACKAGE_RE.match(arg):
        return False
    return not bool(_SAFE_NON_PATH_RE.match(arg))


def _validate_command(command: list[Any]) -> str | None:
    """Validate a command list for length, path traversal, and dangerous executables."""
    if not isinstance(command, list):
        return "command must be a list"
    if not command or len(command) > MAX_COMMAND_ARGS:
        return f"command must contain between 1 and {MAX_COMMAND_ARGS} arguments"
    for i, arg in enumerate(command):
        if not isinstance(arg, str):
            return "command arguments must be strings"
        if len(arg) == 0 or len(arg) > MAX_ARG_LEN:
            return f"command argument at position {i} exceeds length limits"
        if "\x00" in arg or "\n" in arg or "\r" in arg:
            return "command argument contains disallowed characters"
        if i == 0:
            if _is_path_arg(arg):
                if not _is_allowed_executable_path(arg):
                    return f"executable path is not allowed: {arg!r}"
                exe_name = Path(arg).name
            else:
                if not _is_safe_base_name(arg):
                    return f"executable name is not allowed: {arg!r}"
                exe_name = arg
            if _is_dangerous_executable(exe_name, command):
                return f"executable is not allowed: {exe_name}"
        else:
            if _is_path_arg(arg):
                if not _is_safe_path_arg(arg):
                    return f"path argument is not allowed: {arg!r}"
            elif _has_shell_metacharacters(arg):
                return f"command argument contains disallowed characters: {arg!r}"
    return None


def _validate_env(env: Any) -> str | None:
    """Validate that an env mapping only contains safe keys and values."""
    if not isinstance(env, dict):
        return "env must be a dictionary"
    if len(env) > MAX_ENV_VARS:
        return "too many environment variables"
    for k, v in env.items():
        if not isinstance(k, str) or not isinstance(v, str):
            return "env keys and values must be strings"
        if len(k) > MAX_ENV_LEN or len(v) > MAX_ENV_LEN:
            return "env variable too long"
        if not _ENV_KEY_RE.match(k):
            return f"invalid env key: {k!r}"
        if k in _DANGEROUS_ENV_KEYS:
            return f"unsafe env key: {k!r}"
    return None


def _sanitize_env(env: dict[str, Any] | None) -> dict[str, str]:
    """Return a sanitized copy of a server env mapping."""
    if not env:
        return {}
    safe: dict[str, str] = {}
    for k, v in env.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if len(k) > MAX_ENV_LEN or len(v) > MAX_ENV_LEN:
            continue
        if not _ENV_KEY_RE.match(k) or k in _DANGEROUS_ENV_KEYS:
            continue
        safe[k] = v
        if len(safe) >= MAX_ENV_VARS:
            break
    return safe


def _has_path_traversal(path: Path) -> bool:
    """Reject paths that contain parent-directory references or null bytes."""
    if "\x00" in str(path):
        return True
    if ".." in path.parts:
        return True
    return False


def _safe_load_json(path: Path, max_bytes: int = MAX_JSON_BYTES) -> Any | None:
    """Load a JSON file with a strict size bound."""
    if not isinstance(path, Path) or not path.is_file():
        return None
    try:
        size = path.stat().st_size
    except (OSError, ValueError):
        return None
    if size > max_bytes:
        print(f"[mcp_marketplace] JSON payload too large ({size} bytes): {path}", flush=True)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
        print(f"[mcp_marketplace] could not parse JSON {path}: {e}", flush=True)
    return None


def _validate_catalog_entry(entry: Any) -> str | None:
    """Validate one catalog server entry."""
    if not isinstance(entry, dict):
        return "catalog entry is not a dict"
    name = entry.get("name")
    if not _is_safe_name(name):
        return f"invalid server name: {name!r}"
    install_type = entry.get("install_type", "command")
    if install_type not in {"command", "pip"}:
        return f"invalid install_type: {install_type!r}"
    if install_type == "pip":
        pkg = entry.get("pip_package") or name
        if not isinstance(pkg, str) or len(pkg) > MAX_ARG_LEN or not re.fullmatch(
            r"^[A-Za-z0-9_.-]+$", pkg
        ):
            return f"invalid pip_package: {pkg!r}"
    command = entry.get("command")
    if command is not None:
        err = _validate_command(command)
        if err:
            return err
    env = entry.get("env")
    if env is not None:
        err = _validate_env(env)
        if err:
            return err
    description = entry.get("description", "")
    if not isinstance(description, str) or len(description) > MAX_DESCRIPTION_LEN:
        return "invalid description"
    publisher = entry.get("publisher", "")
    if not isinstance(publisher, str) or len(publisher) > MAX_ARG_LEN:
        return "invalid publisher"
    if publisher and not re.fullmatch(r"^[A-Za-z0-9_.-]+$", publisher):
        return "invalid publisher"
    return None


def _validate_catalog(data: Any) -> list[dict[str, Any]]:
    """Return only the valid server entries from a catalog dict."""
    if not isinstance(data, dict):
        return []
    servers = data.get("servers", [])
    if not isinstance(servers, list) or len(servers) > MAX_CATALOG_SERVERS:
        return []
    valid: list[dict[str, Any]] = []
    for entry in servers:
        err = _validate_catalog_entry(entry)
        if err is None:
            valid.append(entry)
        else:
            print(f"[mcp_marketplace] catalog entry rejected: {err}", flush=True)
    return valid


def _resolve_command(command: list[str], path: str | None = None) -> list[str]:
    """Expand ~ and resolve the executable in common bin paths if needed.

    Raises SecurityError when the command list contains disallowed values.
    """
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
    err = _validate_command(expanded)
    if err:
        raise SecurityError(err)
    exe = expanded[0]
    if exe and not Path(exe).is_absolute():
        search_path = (path or os.environ.get("PATH", "")) + ":" + _extra_path
        resolved = shutil.which(exe, path=search_path)
        if resolved:
            expanded[0] = resolved
            err = _validate_command(expanded)
            if err:
                raise SecurityError(err)
    return expanded


DEFAULT_REGISTRY = Path("/var/lib/bad_apple/mcp_servers.json")
DEFAULT_TIMEOUT = 30
DEFAULT_CATALOG = Path(__file__).with_name("mcp_registry.json")


def _load_registry() -> dict[str, Any]:
    data = _safe_load_json(DEFAULT_REGISTRY, max_bytes=MAX_JSON_BYTES)
    if not isinstance(data, dict):
        return {"servers": []}
    servers = data.get("servers", [])
    if not isinstance(servers, list) or len(servers) > MAX_CATALOG_SERVERS:
        return {"servers": []}
    return data


def _save_registry(data: dict[str, Any]) -> None:
    DEFAULT_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_REGISTRY, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog_path = path or DEFAULT_CATALOG
    if _has_path_traversal(catalog_path):
        print(f"[mcp_marketplace] catalog path traversal rejected: {catalog_path}", flush=True)
        return {"servers": []}
    data = _safe_load_json(catalog_path, max_bytes=MAX_JSON_BYTES)
    if not isinstance(data, dict):
        return {"servers": []}
    return {"servers": _validate_catalog(data)}


def _install_pip_server(entry: dict[str, Any]) -> list[str]:
    err = _validate_catalog_entry(entry)
    if err:
        raise SecurityError(err)
    pkg = entry.get("pip_package") or entry["name"]
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
        if (
            len(command) >= 3
            and command[0] in ("python", str(python))
            and command[1] == "-m"
            and command[2] == pkg.replace("-", "_")
        ):
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
    return _resolve_command(expanded)


def install_catalog_server(name: str, catalog_path: Path | None = None) -> str:
    if not _is_safe_name(name):
        return "MCP server name must be alphanumeric, hyphens or underscores."
    catalog = _load_catalog(catalog_path)
    for entry in catalog.get("servers", []):
        if entry.get("name") == name:
            install_type = entry.get("install_type", "command")
            if install_type == "pip":
                try:
                    command = _install_pip_server(entry)
                except (SecurityError, ValueError) as e:
                    return f"MCP server '{name}' command rejected: {e}"
            elif install_type == "command":
                command = list(entry.get("command", []))
                if not command:
                    return f"MCP server '{name}' has no command in the catalog."
                try:
                    command = _resolve_command(command)
                except (SecurityError, ValueError) as e:
                    return f"MCP server '{name}' command rejected: {e}"
            else:
                return f"MCP server '{name}' has unknown install_type '{install_type}'."
            try:
                return _MARKETPLACE.add_server(name, command, entry.get("env"))
            except (SecurityError, ValueError) as e:
                return f"MCP server '{name}' rejected: {e}"
    return f"MCP server '{name}' not found in catalog."


def list_catalog_servers(catalog_path: Path | None = None) -> list[dict[str, Any]]:
    return _load_catalog(catalog_path).get("servers", [])


class MCPClient:
    """A lightweight stdio MCP client."""

    def __init__(self, name: str, command: list[str], env: dict[str, str] | None = None):
        self.name = name
        self.command = _resolve_command(command)
        self.env = _sanitize_env(env)
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
        payload = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        if len(payload) > MAX_JSON_LINE_BYTES:
            raise ValueError(f"JSON-RPC payload exceeds {MAX_JSON_LINE_BYTES} bytes")
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()

    def _reader_loop(self) -> None:
        while True:
            try:
                line = self._proc.stdout.readline()
            except Exception:  # noqa: BLE001 - catch-all wrapper
                break
            if not line:
                break
            if len(line) > MAX_JSON_LINE_BYTES:
                continue
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
        env.update(self.env)
        for k in _DANGEROUS_ENV_KEYS:
            env.pop(k, None)
        env["PATH"] = (os.environ.get("PATH", "") + ":" + _extra_path).strip(":")

        try:
            command = _resolve_command(self.command, path=env["PATH"])
        except (SecurityError, ValueError) as e:
            print(f"[mcp_client] could not start {self.name}: {e}", flush=True)
            return False

        try:
            self._proc = subprocess.Popen(
                command,
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
            print(f"[mcp_client] could not start {self.name}: process exited before initialize", flush=True)
            return False
        try:
            self._call(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "badapple", "version": "0.1"},
                },
                timeout=60,
            )
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
        try:
            command = _resolve_command(command)
        except (SecurityError, ValueError) as e:
            return f"MCP server command rejected: {e}"
        err = _validate_env(env)
        if err:
            return f"MCP server env rejected: {err}"
        data = _load_registry()
        servers = data.get("servers", [])
        for s in servers:
            if s["name"] == name:
                return f"MCP server '{name}' already exists."
        servers.append({"name": name, "command": command, "env": _sanitize_env(env)})
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
                    try:
                        client = MCPClient(name, s["command"], s.get("env"))
                    except (SecurityError, ValueError) as e:
                        print(f"[mcp_marketplace] server {name} rejected: {e}", flush=True)
                        return None
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
        if _AIRGAP and server in _NETWORK_MCP_SERVERS:
            return f"Air-gap mode is on. MCP server '{server}' has been blocked."
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
    try:
        import shlex
        parts = shlex.split(command)
    except ValueError as e:
        return f"MCP server command rejected: {e}"
    return _MARKETPLACE.add_server(name, parts, env)


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
        try:
            cmd = _resolve_command(list(c.get("command", [])))
        except (SecurityError, ValueError) as e:
            print(f"[mcp_marketplace] catalog entry {c.get('name')} rejected: {e}", flush=True)
            continue
        install = c.get("install_note") or (cmd and " ".join(cmd)) or f"add mcp server {c['name']}"
        lines.append(f"- {c['name']}: {c['description']}\n  install: {install}")
    if not lines:
        return "No MCP servers are currently available."
    return "Available MCP servers:\n" + "\n".join(lines)


def install_mcp_server_from_marketplace(name: str) -> str:
    """Install a server from the built-in catalog by name."""
    return install_catalog_server(name, catalog_path=Path(__file__).with_name("mcp_registry.json"))
