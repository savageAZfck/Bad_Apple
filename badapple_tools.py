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

from badapple_knowledge import BadAppleKnowledge
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
    "git", "swift", "cargo", "rustc", "python3", "python",
}
SHELL_DANGEROUS_CHARS = set(";|&$`\"'\n\r<>{}[]*?")


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
    # Allow commands either by name or by absolute path to an allowed tool.
    if base.startswith("/"):
        name = os.path.basename(base)
    else:
        name = base
    if name not in SHELL_ALLOWED_COMMANDS:
        return f"Error: '{name}' is not in the allowed command list"
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


def run_tool(name: str, args: dict, knowledge: BadAppleKnowledge | None = None, approval: Any | None = None, policy: Any | None = None, workspace: Any | None = None) -> str:
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
            proposal_id = approval.propose("invoke_mcp_tool", args)
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
            if not p.is_dir():
                return f"Error: {p} is not a directory"
            items = sorted(p.iterdir())[:50]
            return "\n".join(str(i.name) for i in items)
        if name == "read_file":
            p = _resolve_tool_path(args, "path", workspace)
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
            return "UI action failed (Aqua helper not available)"
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
            if action == "type":
                script = f'tell application "{target}" to activate\ntell application "System Events" to keystroke "{value}"'
            elif action == "key":
                script = f'tell application "System Events" to key code {value}'
            elif action == "menu":
                parts = value.split(">")
                script = f'tell application "{target}" to activate\ntell application "System Events" to tell process "{target}" to click menu item "{parts[-1]}" of menu "{parts[0]}" of menu bar 1'
            elif action == "click":
                script = f'tell application "{target}" to activate\ntell application "System Events" to tell process "{target}" to click UI element "{value}"'
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
            p = args.get("path") or str(Path(tempfile.gettempdir()) / "badapple_screen.png")
            return str(badapple_vision.capture_screen(Path(p).expanduser()))
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
        if name == "add_mcp_server":
            return badapple_mcp_marketplace.add_mcp_server(
                args.get("name", ""), args.get("command", ""), args.get("env")
            )
        if name == "mcp_marketplace":
            return badapple_mcp_marketplace.marketplace_catalog()
        if name == "mcp_install":
            return badapple_mcp_marketplace.install_mcp_server_from_marketplace(args.get("name", ""))
        if name == "remove_mcp_server":
            return badapple_mcp_marketplace.remove_mcp_server(args.get("name", ""))
        if name == "list_mcp_servers":
            return badapple_mcp_marketplace.list_mcp_servers()
        if name == "list_mcp_tools":
            return badapple_mcp_marketplace.list_mcp_tools(args.get("server", ""))
        if name == "invoke_mcp_tool":
            return badapple_mcp_marketplace.invoke_mcp_tool(
                args.get("server", ""), args.get("tool", ""), args.get("arguments") or {}
            )
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        return f"Tool error: {e}"
    return "Unknown tool"
