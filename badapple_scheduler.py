#!/usr/bin/env python3
"""Local task scheduler and macOS Shortcuts runner for Bad Apple.

Tasks are stored in a JSONL file and executed by a background thread in the
MLX server. `run_shortcut` calls the local `shortcuts` CLI directly.
"""

import json
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Keep in sync with badapple_tools.SHELL_ALLOWED_COMMANDS / SHELL_DANGEROUS_CHARS.
# Duplicated here (rather than imported) to avoid a circular import --
# badapple_tools already imports this module to dispatch schedule_task.
# schedule_task requires human approval before a task is even queued (see
# policy.yaml), but the approved command still runs unattended, later, with
# no one present to notice something unexpected -- so its deferred execution
# must not be *more* permissive than run_shell's immediate, approved
# execution. Without this, schedule_task would be a complete bypass of
# run_shell's shell-metacharacter and command-allowlist hardening.
_SHELL_ALLOWED_COMMANDS = {
    "ls", "cat", "head", "tail", "find", "grep", "wc", "file",
    "pwd", "mdfind", "ps", "df", "du", "echo", "whoami", "id",
    "git", "swift", "cargo", "rustc", "python3", "python",
}
_SHELL_DANGEROUS_CHARS = set(";|&$`\"'\n\r<>{}[]*?")


def _schedule_file() -> Path:
    for d in (
        Path("/var/lib/bad_apple"),
        Path.home() / ".bad_apple",
        Path(tempfile.gettempdir()),
    ):
        try:
            d.mkdir(parents=True, exist_ok=True)
            test = d / ".write_test"
            test.write_text("x")
            test.unlink()
            return d / "schedule.jsonl"
        except (PermissionError, OSError):
            continue
    return Path.home() / ".bad_apple" / "schedule.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_when(when: str) -> float | None:
    """Parse when as an ISO timestamp or a delay in seconds."""
    try:
        # Delay in seconds
        return time.time() + float(when)
    except ValueError:
        pass
    try:
        from datetime import datetime
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"):
            try:
                dt = datetime.strptime(when, fmt)
                return dt.timestamp()
            except ValueError:
                continue
        # Try ISO parse
        dt = datetime.fromisoformat(when)
        return dt.timestamp()
    except Exception:  # noqa: BLE001 - catch-all wrapper
        return None


def list_tasks() -> str:
    f = _schedule_file()
    if not f.is_file():
        return "No scheduled tasks."
    out = []
    try:
        with f.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    status = t.get("status", "pending")
                    when = t.get("when_ts", 0)
                    out.append(
                        f"{i}. [{status}] {datetime.fromtimestamp(when, tz=timezone.utc).astimezone().isoformat()} — {t.get('command','')[:60]}"
                    )
                except json.JSONDecodeError:
                    continue
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        return f"Error reading schedule: {e}"
    if not out:
        return "No scheduled tasks."
    return "\n".join(out)


def add_task(when: str, command: str, repeat: str = "") -> str:
    """Add a task. `when` is seconds from now or an ISO timestamp."""
    when_ts = _parse_when(when)
    if when_ts is None:
        return f"Error: could not parse time '{when}'"
    task = {
        "when_ts": when_ts,
        "command": command,
        "repeat": repeat,
        "status": "pending",
        "created": _now_iso(),
    }
    try:
        with _schedule_file().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(task) + "\n")
        return f"Scheduled task for {datetime.fromtimestamp(when_ts, tz=timezone.utc).astimezone().isoformat()}"
    except (TypeError, ValueError, OSError) as e:
        return f"Error scheduling task: {e}"


def _validated_argv(command: str) -> list[str] | None:
    """Tokenize `command` and check it against the same hardening run_shell
    uses. Returns the argv list if allowed, or None if rejected.
    """
    if any(c in command for c in _SHELL_DANGEROUS_CHARS):
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens:
        return None
    base = tokens[0]
    name = Path(base).name if base.startswith("/") else base
    if name not in _SHELL_ALLOWED_COMMANDS:
        return None
    return tokens


def _run_command(command: str) -> str:
    try:
        args = json.loads(command) if command.startswith("[") else command
        if isinstance(args, list):
            tokens = args
            if not tokens:
                return "Task error: empty command"
            base = tokens[0]
            name = Path(base).name if isinstance(base, str) and base.startswith("/") else base
            if name not in _SHELL_ALLOWED_COMMANDS:
                return f"Task error: '{name}' is not in the allowed command list"
        else:
            tokens = _validated_argv(args)
            if tokens is None:
                return "Task error: command contains dangerous characters, is empty, or is not in the allowed command list"
        # Never shell=True: scheduled commands run as an argv list, exactly
        # like run_shell, so no shell metacharacter in an already-validated
        # command string can be reinterpreted at execution time.
        result = subprocess.run(tokens, capture_output=True, text=True, timeout=60, check=False)
        return (result.stdout or "") + (result.stderr or "")
    except (subprocess.SubprocessError, OSError, ValueError, json.JSONDecodeError, TypeError, AttributeError) as e:
        return f"Task error: {e}"


def run_due_tasks() -> str:
    """Execute any due tasks and mark them done."""
    f = _schedule_file()
    if not f.is_file():
        return "No tasks due."
    now = time.time()
    results = []
    updated = []
    try:
        with f.open("r+", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    updated.append(line)
                    continue
                try:
                    t = json.loads(line)
                    if t.get("status") == "pending" and t.get("when_ts", 0) <= now:
                        t["status"] = "running"
                        output = _run_command(t.get("command", ""))
                        t["status"] = "done"
                        t["completed"] = _now_iso()
                        t["output"] = output[:2000]
                        results.append(f"Ran '{t.get('command','')[:50]}...' -> {output[:120]}")
                        if t.get("repeat"):
                            # Re-add with repeat interval in seconds
                            try:
                                interval = float(t["repeat"])
                                t["when_ts"] = now + interval
                                t["status"] = "pending"
                                t.pop("completed", None)
                                t.pop("output", None)
                            except ValueError:
                                pass
                    updated.append(json.dumps(t))
                except json.JSONDecodeError:
                    updated.append(line)
            fh.seek(0)
            fh.truncate()
            fh.write("\n".join(updated) + "\n")
    except (OSError, ValueError) as e:
        return f"Scheduler error: {e}"
    if not results:
        return "No tasks due."
    return "\n".join(results)


def run_shortcut(name: str, input_text: str | None = None) -> str:
    """Run a macOS Shortcuts shortcut by name."""
    if not shutil.which("shortcuts"):
        return "Error: `shortcuts` CLI not available on this system."
    cmd = ["shortcuts", "run", name]
    try:
        if input_text:
            result = subprocess.run(
                cmd,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=120,
            check=False)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        if result.returncode != 0:
            return f"Shortcut error ({result.returncode}): {result.stderr or result.stdout}"
        return result.stdout or "Shortcut ran."
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        return f"Shortcut error: {e}"


def _scheduler_loop(interval: int = 60):
    """Background thread that executes due tasks every `interval` seconds."""
    while True:
        try:
            run_due_tasks()
        except Exception as e:  # noqa: BLE001 - logged
            print(f"[scheduler] run_due_tasks failed: {e}", flush=True)
        time.sleep(interval)


def start_background_scheduler(interval: int = 60):
    t = threading.Thread(target=_scheduler_loop, args=(interval,), daemon=True)
    t.start()
    return t
