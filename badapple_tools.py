#!/usr/bin/env python3
"""Tool execution helpers for the Bad Apple MLX server.

This module contains the tool dispatcher and shell helpers that used to live in
badapple_mlx_server.py. It is imported by the server so the two files can be
tested and maintained independently.
"""
import datetime
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import badapple_ambient
import badapple_aqua_helper
import badapple_dashboard
import badapple_documents
import badapple_git
import badapple_image_gen
import badapple_keychain
import badapple_lora
import badapple_macos_apps
import badapple_mcp_marketplace
import badapple_ocular
import badapple_p2p
import badapple_scheduler
import badapple_spotlight
import badapple_stt
import badapple_supervisor
import badapple_translate
import badapple_undo
import badapple_vision
import badapple_working_memory
import badapple_xcode
from badapple_knowledge import BadAppleKnowledge

# Pinned random seed for reproducible sessions. Set at startup via BADAPPLE_SEED
# or changed at runtime with the set_session_seed tool. 0 means random.
_SESSION_SEED: int | None = None
if os.environ.get("BADAPPLE_SEED"):
    try:
        _SESSION_SEED = int(os.environ.get("BADAPPLE_SEED"))
    except ValueError:
        _SESSION_SEED = None


def get_session_seed() -> int | None:
    return _SESSION_SEED


def set_session_seed(seed: int | None) -> None:
    global _SESSION_SEED
    _SESSION_SEED = seed


SHELL_ALLOWED_COMMANDS = {
    "ls", "cat", "head", "tail", "find", "grep", "wc", "file",
    "pwd", "mdfind", "ps", "df", "du", "echo", "whoami", "id",
    # NOTE: Interpreters/compilers (python3, swift, cargo, rustc, git) were
    # removed because they can execute arbitrary attacker-supplied code.
}
SHELL_DANGEROUS_CHARS = set(";|&$`\"'\n\r<>{}[]*?")

# Commands whose path-like arguments must be jailed to allowed roots.
_SHELL_PATH_ARG_COMMANDS = {"cat", "head", "tail", "grep", "find", "file", "wc"}


def _jail_shell_args(tokens: list[str]) -> str | None:
    """Check path-like tokens and reject paths outside allowed roots.

    Returns an error string if a path-like argument (one starting with ``/``,
    ``~``, or ``.``) resolves outside the allowed roots (user home, ``/tmp``,
    ``/var/tmp``). Returns ``None`` if all path-like arguments are safe.
    """
    home = Path("~").expanduser()
    allowed_roots = [home, Path("/tmp"), Path("/var/tmp")]
    allowed_resolved = []
    for root in allowed_roots:
        try:
            allowed_resolved.append(str(root.resolve()))
        except (OSError, ValueError):
            continue
    for tok in tokens[1:]:
        if not tok or not (tok.startswith("/") or tok.startswith("~") or tok.startswith(".")):
            continue
        try:
            p = Path(tok).expanduser()
            resolved = p.resolve() if p.exists() else p.parent.resolve() / p.name
        except (OSError, ValueError):
            return f"Error: path '{tok}' is outside allowed roots"
        for root_str in allowed_resolved:
            if str(resolved).startswith(root_str):
                break
        else:
            return f"Error: path '{tok}' is outside allowed roots"
    return None


def _run_shell(command: str) -> str:
    if not command:
        return "Error: no command"
    # Reject any command that contains shell metacharacters.
    if any(c in command for c in SHELL_DANGEROUS_CHARS):
        return "Error: command contains dangerous characters or operators"
    try:
        tokens = shlex.split(command)
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        return f"Error: invalid command syntax: {e}"
    if not tokens:
        return "Error: empty command"
    base = tokens[0]
    # Resolve the command to a known-safe executable.  If the user passed a
    # bare name, look it up on PATH.  If they passed an absolute path, verify
    # it resolves to the same file as `shutil.which(name)` for an allowlisted
    # name — this prevents an attacker from planting a malicious binary named
    # after an allowed command (e.g. /tmp/ls).
    import shutil
    if base.startswith("/"):
        name = os.path.basename(base)
        if name not in SHELL_ALLOWED_COMMANDS:
            return f"Error: '{name}' is not in the allowed command list"
        resolved = shutil.which(name)
        if resolved is None or os.path.realpath(base) != os.path.realpath(resolved):
            return f"Error: '{base}' does not resolve to the trusted '{name}' on PATH"
        tokens[0] = resolved
    else:
        name = base
        if name not in SHELL_ALLOWED_COMMANDS:
            return f"Error: '{name}' is not in the allowed command list"
        resolved = shutil.which(name)
        if resolved is None:
            return f"Error: '{name}' not found on PATH"
        tokens[0] = resolved
    # Jailing: for commands that take file arguments, verify any path-like
    # arguments stay within the allowed roots (home, /tmp, /var/tmp).
    if name in _SHELL_PATH_ARG_COMMANDS:
        err = _jail_shell_args(tokens)
        if err:
            return err
    try:
        result = subprocess.run(
            tokens,
            capture_output=True,
            text=True,
            timeout=15,
        check=False)
        out = (result.stdout or "").strip()
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            return f"Error ({result.returncode}): {err or 'command failed'}"
        return out[:5000] or "(no output)"
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        return f"Error: {e}"


def _resolve_tool_path(args: dict, key: str, workspace: Any | None = None) -> Path:
    maybe = args.get(key)
    if maybe:
        return Path(maybe).expanduser()
    if workspace is not None:
        return workspace.resolve_path(None)
    return Path("~").expanduser()


# Safe roots that tools are allowed to read/list/search without a workspace.
# These are user-owned directories that don't contain secrets.
_SAFE_READ_ROOTS: list[Path] | None = None


def _safe_read_roots() -> list[Path]:
    """Return the list of root paths that tools are allowed to access."""
    global _SAFE_READ_ROOTS
    if _SAFE_READ_ROOTS is None:
        home = Path("~").expanduser()
        _SAFE_READ_ROOTS = [
            home,
            Path("/tmp"),
            Path("/var/tmp"),
        ]
    return _SAFE_READ_ROOTS


def _jail_path(path: Path, workspace: Any | None = None) -> Path:
    """Resolve symlinks and verify the path is under an allowed root.

    Raises ValueError if the resolved path escapes all allowed roots.
    """
    resolved = path.resolve() if path.exists() else path.parent.resolve() / path.name
    # Check workspace first if set
    if workspace is not None:
        ws_path = workspace.resolve_path(None)
        try:
            ws_resolved = ws_path.resolve()
            if str(resolved).startswith(str(ws_resolved)):
                return resolved
        except (OSError, ValueError):
            pass
    # Check safe read roots
    for root in _safe_read_roots():
        try:
            root_resolved = root.resolve()
            if str(resolved).startswith(str(root_resolved)):
                return resolved
        except (OSError, ValueError):
            continue
    raise ValueError(f"path {resolved} is outside allowed roots")


def _console_user() -> str | None:
    """Return the name of the current console (Aqua/session) user, if any."""
    try:
        result = subprocess.run(
            ["stat", "-f", "%Su", "/dev/console"],
            capture_output=True,
            text=True,
            timeout=5,
        check=False)
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def _run_as_user(cmd: list[str], user: str | None = None, input_text: str | None = None, timeout: int = 30):
    """Run a subprocess as the console user when the daemon is root."""
    target = user or _console_user()
    if target and target != "root":
        full = ["sudo", "-n", "-u", target] + cmd
    else:
        full = cmd
    try:
        return subprocess.run(
            full,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        check=False)
    except subprocess.TimeoutExpired:
        return type("TimeoutResult", (), {"returncode": -1, "stdout": "", "stderr": f"timed out after {timeout}s"})()


def _safari_do_javascript(js: str, timeout: int = 15) -> str:
    """Run JavaScript in the frontmost Safari document via AppleScript."""
    # Escape double quotes and backslashes for the AppleScript string.
    escaped = js.replace("\\", "\\\\").replace('"', '\\"')
    script = f'tell application "Safari" to do JavaScript "{escaped}" in front document'
    result = _run_as_user(["osascript", "-e", script], timeout=timeout)
    out = (result.stdout or "").strip()
    if result.returncode != 0:
        err = (result.stderr or "").strip()
        if "not authorized" in err.lower() or "not allowed" in err.lower():
            return "Error: Safari does not allow Apple Events. Enable Safari > Develop > Allow Apple Events in Automation."
        return f"Error ({result.returncode}): {err or 'AppleScript failed'}"
    return out[:8000] or "(no output)"


def _browser_action(args: dict) -> str:
    """Drive Safari via AppleScript and do JavaScript."""
    action = args.get("action", "")
    if not action:
        return "Error: no action specified"
    try:
        if action == "navigate":
            url = args.get("url", "")
            if not url:
                return "Error: navigate requires a url"
            # Reject dangerous URL schemes that could access local files or
            # probe internal services.
            url_lower = url.lower()
            blocked_schemes = ("file://", "smb://", "dict://", "ftp://", "ssh://", "vnc://")
            if any(url_lower.startswith(s) for s in blocked_schemes):
                return "Error: URL scheme blocked for security"
            escaped_url = url.replace("\\", "\\\\").replace('"', '\\"')
            script = f'tell application "Safari" to open location "{escaped_url}"'
            result = _run_as_user(["osascript", "-e", script], timeout=10)
            if result.returncode != 0:
                return f"Error: {result.stderr.strip() or 'navigate failed'}"
            return f"Navigated to {url}"
        if action == "url":
            result = _run_as_user(["osascript", "-e", 'tell application "Safari" to get URL of front document'], timeout=10)
            return result.stdout.strip() or "Error: could not get URL"
        if action == "title":
            result = _run_as_user(["osascript", "-e", 'tell application "Safari" to get name of front document'], timeout=10)
            return result.stdout.strip() or "Error: could not get title"
        if action == "text":
            return _safari_do_javascript("document.body.innerText.slice(0, 8000)")
        if action == "click":
            selector = args.get("selector", "")
            if not selector:
                return "Error: click requires a selector"
            escaped_sel = selector.replace("\\", "\\\\").replace("'", "\\'")
            js = f"var el = document.querySelector('{escaped_sel}'); if (el) {{ el.click(); 'clicked'; }} else {{ 'element not found'; }}"
            return _safari_do_javascript(js)
        if action == "type":
            selector = args.get("selector", "")
            text = args.get("text", "")
            if not selector:
                return "Error: type requires a selector"
            escaped_sel = selector.replace("\\", "\\\\").replace("'", "\\'")
            escaped_text = text.replace("\\", "\\\\").replace("'", "\\'")
            js = f"var el = document.querySelector('{escaped_sel}'); if (el) {{ el.value = '{escaped_text}'; 'typed'; }} else {{ 'element not found'; }}"
            return _safari_do_javascript(js)
        if action == "scroll":
            amount = int(args.get("amount", 500))
            js = f"window.scrollBy(0, {amount}); 'scrolled ' + window.scrollY"
            return _safari_do_javascript(js)
        if action == "exec":
            js_code = args.get("javascript", "")
            if not js_code:
                return "Error: exec requires javascript"
            return _safari_do_javascript(js_code, timeout=20)
        return f"Error: unknown action '{action}'"
    except (TypeError, ValueError, KeyError) as e:
        return f"Error: {e}"


def run_tool(
    name: str,
    args: dict,
    knowledge: BadAppleKnowledge | None = None,
    approval: Any | None = None,
    policy: Any | None = None,
    workspace: Any | None = None,
    user_prompt: str = "",
    mcp_marketplace: Any | None = None,
) -> str:
    if policy is not None:
        if not policy.is_allowed(name):
            return f"Policy: tool '{name}' is not allowed."
        error = policy.validate(name, args)
        if error:
            return f"Policy: {error}"
    if approval is not None and approval.needs_approval(name):
        proposal_id = approval.propose(name, args)
        return (
            f"Approval required before I can run {name}. "
            f"Reply with 'approve {proposal_id}' to proceed. "
            f"(Set BADAPPLE_AUTOPILOT=1 to skip these prompts.)"
        )
    if name == "invoke_mcp_tool" and approval is not None and not approval.autopilot:
        server = str(args.get("server", ""))
        tool = str(args.get("tool", ""))
        if badapple_mcp_marketplace.is_mcp_write_tool(server, tool):
            proposal_id = approval.propose("invoke_mcp_tool", args, user_prompt)
            return (
                f"Approval required before I can run MCP write tool '{tool}' on '{server}'. "
                f"Reply with 'approve {proposal_id}' to proceed. "
                f"(Set BADAPPLE_AUTOPILOT=1 to skip these prompts.)"
            )
    try:
        if name == "get_current_time":
            return datetime.datetime.now(tz=datetime.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        if name == "list_directory":
            p = _resolve_tool_path(args, "path", workspace)
            try:
                p = _jail_path(p, workspace)
            except ValueError as e:
                return f"Error: {e}"
            if not p.is_dir():
                return f"Error: {p} is not a directory"
            items = sorted(p.iterdir())[:50]
            return "\n".join(str(i.name) for i in items)
        if name == "read_file":
            p = _resolve_tool_path(args, "path", workspace)
            try:
                p = _jail_path(p, workspace)
            except ValueError as e:
                return f"Error: {e}"
            if not p.is_file():
                return f"Error: {p} is not a file"
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except (OSError, ValueError):
                return f"Error: could not read {p} as text"
            limit = int(args.get("limit") or 10000)
            if len(text) > limit:
                text = text[:limit] + f"\n... ({len(text)} characters total)"
            return text
        if name == "write_file":
            notes_dir = Path(os.environ.get("BADAPPLE_NOTES_DIR", "~/.bad_apple/notes")).expanduser()
            notes_dir.mkdir(parents=True, exist_ok=True)
            filename = os.path.basename(args.get("filename", "note.txt"))
            p = notes_dir / filename
            if not str(p.resolve()).startswith(str(notes_dir.resolve())):
                return "Error: filename is not allowed"
            content = args.get("content", "")
            undo = badapple_undo.UndoJournal(Path(os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple")))
            undo_id = undo.capture_file(p, "append_file" if args.get("append") else "write_file")
            if args.get("append"):
                with open(p, "a", encoding="utf-8") as f:
                    f.write(content + "\n")
                return f"Appended to {p.name} (undo {undo_id[:8]})"
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Wrote {p} (undo {undo_id[:8]})"
        if name == "undo_last":
            return badapple_undo.UndoJournal(
                Path(os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple"))
            ).undo_last()
        if name == "search_content":
            query = args.get("query", "")
            p = _resolve_tool_path(args, "path", workspace)
            try:
                p = _jail_path(p, workspace)
            except ValueError as e:
                return f"Error: {e}"
            if not p.is_dir():
                return f"Error: {p} is not a directory"
            max_results = int(args.get("max_results") or 20)
            result = subprocess.run(
                [
                    "grep", "-R", "-n", "-i", "--max-count=1",
                    "--binary-files=without-match",
                    "--exclude-dir=.git", "--exclude-dir=target", "--exclude-dir=.build",
                    "--exclude-dir=.venv", "--exclude-dir=node_modules", "--exclude-dir=Pods",
                    "--", query, str(p),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            check=False)
            lines = [line for line in (result.stdout or "").splitlines() if line][:max_results]
            return "\n".join(lines) or "No matches found"
        if name == "run_shell":
            return _run_shell(args.get("command", ""))
        if name == "run_applescript":
            script = args.get("script", "")
            # Defense in depth: reject scripts that attempt to break out of the
            # AppleScript sandbox. These patterns are checked here in addition to
            # any policy.yaml deny list so they cannot be relaxed by config.
            _APPLESCRIPT_FORBIDDEN = {
                "do shell",
                "shell script",
                "system attribute",
                "current application",
                "NSTask",
                "NSAppleScript",
                "POSIX path of",
                "/bin/",
                "/usr/",
                "/sbin/",
                "osascript",
                "terminal",
                "do script",
            }
            lower = script.lower()
            for pat in _APPLESCRIPT_FORBIDDEN:
                if pat.lower() in lower:
                    return "Error: AppleScript contains a forbidden operation"
            result = _run_as_user(["osascript", "-e", script], timeout=15)
            return (result.stdout or result.stderr or "done").strip()
        if name == "run_shortcut":
            shortcut_name = args.get("name", "")
            shortcut_input = args.get("input", "")
            aqua = badapple_aqua_helper.call_aqua(
                "run_shortcut",
                name=shortcut_name,
                input=shortcut_input or "",
                timeout=60,
            )
            if aqua and aqua.get("ok"):
                return aqua.get("output") or "done"
            if aqua:
                return f"Error running shortcut: {aqua.get('error')}"
            # Fallback: try from the daemon's context (usually fails without Aqua).
            result = _run_as_user(["shortcuts", "run", shortcut_name], input_text=shortcut_input or "", timeout=60)
            return (result.stdout or result.stderr or "done").strip()
        if name == "ui_action":
            action = args.get("action", "info")
            aqua = badapple_aqua_helper.call_aqua(
                f"ui_{action}",
                target=args.get("target", ""),
                role=args.get("role", ""),
                text=args.get("text", ""),
                timeout=30,
            )
            if aqua and aqua.get("ok"):
                if action == "info":
                    return json.dumps(aqua, indent=2, default=str)
                return aqua.get("result") or "done"
            if aqua:
                return f"UI action error: {aqua.get('error')}"
            return (
                "I can't control other apps' UI right now -- the menu bar helper isn't "
                "running. Open Bad Apple from the menu bar (not just the daemon) and try again."
            )
        if name == "browser_action":
            return _browser_action(args)
        if name == "list_shortcuts":
            aqua = badapple_aqua_helper.call_aqua("list_shortcuts", timeout=15)
            if aqua and aqua.get("ok"):
                shortcuts = aqua.get("shortcuts") or []
                return "\n".join(shortcuts) or "No shortcuts found"
            if aqua:
                return f"Error listing shortcuts: {aqua.get('error')}"
            # Fallback.
            result = _run_as_user(["shortcuts", "list"], timeout=15)
            if result.returncode != 0:
                return f"Error listing shortcuts: {result.stderr or result.stdout}"
            lines = [line for line in (result.stdout or "").splitlines() if line][:100]
            return "\n".join(lines) or "No shortcuts found"
        if name == "read_working_memory":
            return badapple_working_memory.read_memory(int(args.get("limit") or 5000))
        if name == "write_working_memory":
            return badapple_working_memory.write_memory(
                args.get("content", ""),
                mode=args.get("mode", "replace"),
            )
        if name == "clear_working_memory":
            return badapple_working_memory.clear_memory()

        if name == "accessibility_action":
            action = args.get("action", "")
            target = args.get("target", "")
            value = args.get("value", "")
            # Escape all user-controlled strings for AppleScript safety.
            from badapple_macos_apps import _esc_applescript
            esc_target = _esc_applescript(target)
            esc_value = _esc_applescript(value)
            if action == "type":
                script = f'tell application "{esc_target}" to activate\ntell application "System Events" to keystroke "{esc_value}"'
            elif action == "key":
                # key code is numeric, validate it
                if not value.strip().isdigit():
                    return "Error: key code must be a number"
                script = f'tell application "System Events" to key code {value.strip()}'
            elif action == "menu":
                parts = value.split(">")
                esc_parts = [_esc_applescript(p) for p in parts]
                script = f'tell application "{esc_target}" to activate\ntell application "System Events" to tell process "{esc_target}" to click menu item "{esc_parts[-1]}" of menu "{esc_parts[0]}" of menu bar 1'
            elif action == "click":
                script = f'tell application "{esc_target}" to activate\ntell application "System Events" to tell process "{esc_target}" to click UI element "{esc_value}"'
            else:
                return f"Error: unknown accessibility action '{action}'"
            result = _run_as_user(["osascript", "-e", script], timeout=15)
            return (result.stdout or result.stderr or "done").strip()
        if name == "set_session_seed":
            try:
                seed = int(args.get("seed", 0))
                set_session_seed(seed if seed != 0 else None)
                return f"Session seed pinned to {get_session_seed()}"
            except ValueError:
                return "Error: seed must be an integer"
        if name == "get_session_seed":
            return str(get_session_seed()) if get_session_seed() is not None else "random"
        if name == "ambient_start":
            return badapple_ambient.start(float(args.get("interval") or 30))
        if name == "ambient_stop":
            return badapple_ambient.stop()
        if name == "ambient_context":
            return badapple_ambient.get_context()
        if name == "ocular_start":
            return badapple_ocular.start(
                float(args.get("capture_interval") or 5),
                float(args.get("describe_interval") or 0),
                args.get("prompt"),
            )
        if name == "ocular_stop":
            return badapple_ocular.stop()
        if name == "ocular_context":
            return badapple_ocular.get_context()
        if name == "spotlight_search":
            return badapple_spotlight.search(
                query=args.get("query", ""),
                max_results=int(args.get("max_results") or 20),
            )
        if name == "xcode_index_project":
            if knowledge is None:
                return "Xcode RAG unavailable: no knowledge store."
            return badapple_xcode.index_project(args.get("project_path", ""), knowledge)
        if name == "xcode_search":
            if knowledge is None:
                return "Xcode RAG unavailable: no knowledge store."
            return badapple_xcode.search_project(
                args.get("query", ""),
                knowledge,
                max_results=int(args.get("max_results") or 10),
            )
        if name == "xcode_project_info":
            return badapple_xcode.project_info(args.get("project_path", ""))
        if name == "xcode_build_diagnostics":
            return badapple_xcode.build_diagnostics(
                args.get("project_path", ""),
                args.get("scheme", ""),
                args.get("configuration", "Debug"),
            )
        if name == "slicks_keychain_store":
            return badapple_keychain.store_secret(
                badapple_keychain.get_or_create_secret(
                    args.get("service", badapple_keychain.DEFAULT_SERVICE),
                    args.get("account", badapple_keychain.DEFAULT_ACCOUNT),
                ),
                args.get("service", badapple_keychain.DEFAULT_SERVICE),
                args.get("account", badapple_keychain.DEFAULT_ACCOUNT),
            )
        if name == "p2p_peers":
            daemon = badapple_p2p.get_p2p_daemon()
            if daemon is None:
                return "P2P daemon is not running."
            return daemon.get_peers()
        if name == "p2p_list_adapters":
            daemon = badapple_p2p.get_p2p_daemon()
            if daemon is None:
                return "P2P daemon is not running."
            return daemon.list_local_adapters(badapple_lora.LORA_ADAPTERS_DIR)
        if name == "supervisor_status":
            return badapple_supervisor.supervisor_status()
        if name == "heal":
            return badapple_supervisor.heal()
        if name == "run_benchmark":
            import shutil
            badapple_bin = shutil.which("badapple")
            if not badapple_bin:
                # Fall back to the binary in the same package/install tree.
                candidate = Path(__file__).resolve().parent / "target" / "release" / "badapple"
                if candidate.is_file():
                    badapple_bin = str(candidate)
            if not badapple_bin:
                return "Bad Apple benchmark binary not found."
            prompt = args.get("prompt", "")
            max_tokens = int(args.get("max_tokens") or 120)
            if not Path(badapple_bin).is_file():
                return "Bad Apple benchmark binary not found."
            cmd = [badapple_bin, "--benchmark", "-n", str(max_tokens)]
            if prompt:
                cmd.append(prompt)
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
                return (result.stdout or result.stderr or "Benchmark completed with no output.").strip()
            except (subprocess.SubprocessError, OSError, ValueError) as e:
                return f"Benchmark error: {e}"
        if name == "p2p_send_adapter":
            daemon = badapple_p2p.get_p2p_daemon()
            if daemon is None:
                return "P2P daemon is not running."
            return daemon.send_adapter_sync(
                args.get("peer_id", ""),
                args.get("adapter", ""),
                badapple_lora.LORA_ADAPTERS_DIR,
            )
        if name == "transcribe_audio":
            p = Path(args.get("path", "")).expanduser()
            lang = args.get("language", "en")
            return badapple_stt.transcribe(str(p), language=lang)
        if name == "generate_image":
            return badapple_image_gen.generate(
                prompt=args.get("prompt", ""),
                width=int(args.get("width") or 512),
                height=int(args.get("height") or 512),
                steps=int(args.get("steps") or 4),
                seed=int(args.get("seed")) if args.get("seed") is not None else None,
            )
        if name == "translate_text":
            return badapple_translate.translate(
                text=args.get("text", ""),
                source=args.get("source", "en"),
                target=args.get("target", "en"),
            )
        if name == "lora_add_example":
            return badapple_lora.write_example(
                args.get("dataset", "personal"),
                [
                    {"role": "user", "content": args.get("prompt", "")},
                    {"role": "assistant", "content": args.get("completion", "")},
                ],
            )
        if name == "lora_train":
            return badapple_lora.train(
                dataset=args.get("dataset", ""),
                adapter=args.get("adapter", ""),
                iters=int(args.get("iters") or 100),
                learning_rate=float(args.get("learning_rate") or 1e-4),
            )
        if name == "lora_adapters":
            return badapple_lora.get_summary()
        if name == "lora_generate":
            return badapple_lora.generate_with_adapter(
                adapter=args.get("adapter", ""),
                prompt=args.get("prompt", ""),
                max_tokens=int(args.get("max_tokens") or 120),
            )
        if name == "git_status":
            return badapple_git.status(args.get("repo"))
        if name == "git_diff":
            return badapple_git.diff(
                args.get("repo"),
                staged=bool(args.get("staged", False)),
                stat=False,
            )
        if name == "git_log":
            return badapple_git.log(args.get("repo"), n=int(args.get("n") or 10))
        if name == "git_commit":
            stage_result = badapple_git.stage_all(args.get("repo"))
            if "error" in stage_result.lower():
                return stage_result
            return badapple_git.commit(args.get("repo"), args.get("message", ""))
        if name == "system_dashboard":
            return badapple_dashboard.snapshot()
        if name == "workspace_status":
            return workspace.summary()
        if name == "schedule_task":
            return badapple_scheduler.add_task(
                when=args.get("when", ""),
                command=args.get("command", ""),
                repeat=args.get("repeat", ""),
            )
        if name == "list_scheduled_tasks":
            return badapple_scheduler.list_tasks()
        if name == "run_shortcut":
            return badapple_scheduler.run_shortcut(
                name=args.get("name", ""),
                input_text=args.get("input"),
            )
        if name == "screen_capture":
            # Always write to a safe temp path — ignore user-supplied path
            # to prevent arbitrary file overwrite via symlinks.
            p = Path(tempfile.gettempdir()) / "badapple_screen.png"
            return str(badapple_vision.capture_screen(p))
        if name == "capture_and_extract_screen":
            p = Path(tempfile.gettempdir()) / "badapple_screen.png"
            badapple_vision.capture_screen(p)
            max_tokens = int(args.get("max_tokens") or 256)
            host = badapple_vision.get_vision_host()
            return host.extract_text(p, max_tokens)
        if name == "capture_and_describe_screen":
            p = Path(tempfile.gettempdir()) / "badapple_screen.png"
            badapple_vision.capture_screen(p)
            prompt = args.get("prompt", "Describe what is on the screen.")
            max_tokens = int(args.get("max_tokens") or 256)
            host = badapple_vision.get_vision_host()
            return host.describe(p, prompt, max_tokens)
        if name == "describe_image":
            path = Path(args.get("path", "")).expanduser()
            prompt = args.get("prompt", "Describe this image.")
            max_tokens = int(args.get("max_tokens") or 256)
            host = badapple_vision.get_vision_host()
            return host.describe(path, prompt, max_tokens)
        if name == "extract_text_from_image":
            path = Path(args.get("path", "")).expanduser()
            max_tokens = int(args.get("max_tokens") or 256)
            host = badapple_vision.get_vision_host()
            return host.extract_text(path, max_tokens)
        if name == "today_events":
            return badapple_macos_apps.today_events()
        if name == "upcoming_events":
            return badapple_macos_apps.upcoming_events(
                days=int(args.get("days") or 7),
                limit=int(args.get("limit") or 20),
            )
        if name == "list_reminders":
            return badapple_macos_apps.list_reminders(
                list_name=args.get("list_name", ""),
                completed=bool(args.get("completed", False)),
                limit=int(args.get("limit") or 20),
            )
        if name == "unread_emails":
            return badapple_macos_apps.unread_emails(limit=int(args.get("limit") or 10))
        if name == "search_mail":
            return badapple_macos_apps.search_mail(
                query=args.get("query", ""),
                limit=int(args.get("limit") or 10),
            )
        if name == "add_reminder":
            return badapple_macos_apps.add_reminder(
                name=args.get("name", ""),
                list_name=args.get("list_name", ""),
                due=args.get("due", ""),
            )
        if name == "search_local_files":
            query = args.get("query", "")
            result = subprocess.run(
                ["mdfind", query],
                capture_output=True,
                text=True,
                timeout=15,
            check=False)
            lines = [line for line in (result.stdout or "").splitlines() if line][:20]
            return "\n".join(lines) or "No files found"
        if name == "index_documents" and knowledge is not None:
            p = _resolve_tool_path(args, "path", workspace)
            if p.exists():
                if p.is_file() and p.suffix.lower() in {".pdf", ".epub"}:
                    return badapple_documents.index_document(str(p), knowledge)
                count = knowledge.index_paths([p])
                return f"Indexed {count} chunks from {p}"
            return f"Path not found: {p}"
        if name == "read_document":
            p = _resolve_tool_path(args, "path", workspace)
            limit = int(args.get("limit") or 10000)
            return badapple_documents.read_document(str(p), limit)
        if name == "search_notes" and knowledge is not None:
            results = knowledge.search(args.get("query", ""), k=3)
            if not results:
                return "No relevant notes found."
            return "\n\n".join(f"(score: {s:.2f})\n{c}" for c, s in results)
        mcp = mcp_marketplace or badapple_mcp_marketplace
        if name == "add_mcp_server":
            return mcp.add_mcp_server(
                args.get("name", ""), args.get("command", ""), args.get("env")
            )
        if name == "mcp_marketplace":
            return getattr(mcp, "mcp_marketplace", mcp.marketplace_catalog)()
        if name == "mcp_install":
            return getattr(mcp, "mcp_install", mcp.install_mcp_server_from_marketplace)(args.get("name", ""))
        if name == "remove_mcp_server":
            return mcp.remove_mcp_server(args.get("name", ""))
        if name == "list_mcp_servers":
            return mcp.list_mcp_servers()
        if name == "list_mcp_tools":
            return mcp.list_mcp_tools(args.get("server", ""))
        if name == "invoke_mcp_tool":
            return mcp.invoke_mcp_tool(
                args.get("server", ""), args.get("tool", ""), args.get("arguments") or {}
            )
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        return f"Tool error: {e}"
    return "Unknown tool"
