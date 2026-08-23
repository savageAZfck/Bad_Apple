#!/usr/bin/env python3
"""Local Git copilot for Bad Apple.

Provides read-only repo inspection and, with approval, staged commits.
Everything stays on device.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _git(repo: str) -> list[str]:
    return [shutil.which("git") or "git", "-C", repo]


def _run(args: list[str], timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            return f"Git error ({result.returncode}):\n{out.strip()}"
        return out
    except subprocess.TimeoutExpired:
        return "Git command timed out"
    except Exception as e:
        return f"Git command error: {e}"


def _find_repo(path: Optional[str] = None) -> str:
    if not path:
        path = os.getcwd()
    p = Path(path).expanduser().resolve()
    # Walk up until .git found
    while p != p.parent:
        if (p / ".git").is_dir():
            return str(p)
        p = p.parent
    return str(Path(path).expanduser().resolve())


def status(path: Optional[str] = None) -> str:
    repo = _find_repo(path)
    return _run(_git(repo) + ["status", "--short"])


def diff(path: Optional[str] = None, staged: bool = False, stat: bool = False, max_lines: int = 200) -> str:
    repo = _find_repo(path)
    cmd = _git(repo) + ["diff", "--stat"]
    stat_text = _run(cmd + (["--staged"] if staged else []))
    cmd = _git(repo) + ["diff"]
    if staged:
        cmd.append("--staged")
    if stat:
        return stat_text
    out = _run(cmd)
    lines = out.splitlines()
    if len(lines) > max_lines:
        out = "\n".join(lines[:max_lines])
        out += f"\n\n... diff truncated ({len(lines)} lines total)"
    return f"{stat_text}\n{out}".strip()


def branch(path: Optional[str] = None) -> str:
    repo = _find_repo(path)
    return _run(_git(repo) + ["branch", "--show-current"]).strip()


def log(path: Optional[str] = None, n: int = 10) -> str:
    repo = _find_repo(path)
    return _run(_git(repo) + ["log", "-n", str(n), "--oneline"])


def commit(path: Optional[str] = None, message: str = "") -> str:
    repo = _find_repo(path)
    if not message:
        return "Error: commit message is required"
    return _run(_git(repo) + ["commit", "-m", message], timeout=60)


def stage_all(path: Optional[str] = None) -> str:
    repo = _find_repo(path)
    return _run(_git(repo) + ["add", "-A"], timeout=60)
