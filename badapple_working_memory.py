#!/usr/bin/env python3
"""Working memory / scratchpad for Bad Apple.

A lightweight, persistent scratchpad that the model can read and write during a
conversation.  It lives in the data directory and is surfaced through the menu
bar and dashboard so the user can see the assistant's scratch work.
"""

import os
from pathlib import Path

DEFAULT_WORK_FILE = "/var/lib/bad_apple/working_memory.txt"


def _path() -> Path:
    return Path(os.environ.get("BADAPPLE_WORK_FILE") or DEFAULT_WORK_FILE).expanduser()


def _ensure() -> Path:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.is_file():
        p.write_text("", encoding="utf-8")
    return p


def read_memory(limit: int = 5000) -> str:
    p = _ensure()
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, ValueError) as e:
        return f"Error reading working memory: {e}"
    return text[:limit] if len(text) > limit else text


def write_memory(content: str, mode: str = "replace") -> str:
    p = _ensure()
    try:
        if mode == "append":
            with p.open("a", encoding="utf-8") as f:
                f.write(content)
                if not content.endswith("\n"):
                    f.write("\n")
            return "Working memory appended."
        if mode == "prepend":
            existing = p.read_text(encoding="utf-8") if p.is_file() else ""
            p.write_text(content + "\n" + existing, encoding="utf-8")
            return "Working memory prepended."
        p.write_text(content, encoding="utf-8")
        return "Working memory replaced."
    except (OSError, ValueError) as e:
        return f"Error writing working memory: {e}"


def clear_memory() -> str:
    p = _ensure()
    try:
        p.write_text("", encoding="utf-8")
        return "Working memory cleared."
    except (OSError, ValueError) as e:
        return f"Error clearing working memory: {e}"
