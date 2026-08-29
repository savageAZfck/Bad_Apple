#!/usr/bin/env python3
"""Turn ambient app/window/screen snapshots into long-term memory.

Every time badapple_ambient captures a snapshot, this module rewrites it into
facts and episodes in the MemoryGraph. The OS then semantically remembers what
the user was doing without being asked.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import time
from pathlib import Path
from typing import Any

import badapple_vision
from badapple_extras import MemoryGraph

# Window-title patterns that reveal projects, files, or platforms.
_AMBIENT_PATTERNS = [
    # Editor: "README.md - Bad Apple - Code" or "main.swift — Bad Apple — Xcode"
    (
        re.compile(r"^(.+?)\s*[-—]\s*(.+?)\s*[-—]\s*(Code|Xcode|Nova|Fleet|Cursor|Zed|IntelliJ).*", re.IGNORECASE),
        ["User is editing {g0} in project {g1} with {g2}.", "User's current project is {g1}."],
    ),
    # Browser-ish tab: "GitHub - savageAZfck/Bad_Apple" or "Bad Apple - Google Search"
    (
        re.compile(r"^(.+?)\s*[-—]\s*(GitHub|Google Search|Gmail|YouTube|Notion|Linear|Figma|Slack|Discord|X|Twitter).*", re.IGNORECASE),
        ["User has {g1} open for {g0}.", "User is looking at {g0} on {g1}."],
    ),
    # Mail: "Inbox (1) - foo@gmail.com - Mail"
    (
        re.compile(r"^(Inbox|Drafts|Sent).*?[-—]\s*(.+?)\s*[-—]\s*Mail", re.IGNORECASE),
        ["User is using Mail for {g1}."],
    ),
    # Xcode default: "Bad Apple — main.swift"
    (
        re.compile(r"^(.+?)\s*[-—]\s*(.+?\.swift|.+?\.m|.+?\.mm|.+?\.py|.+?\.rs|.+?\.js)$", re.IGNORECASE),
        ["User is working on {g1} in {g0}.", "User's current project is {g0}."],
    ),
]


def _clean(s: str) -> str:
    return s.strip().rstrip(" -—")


def extract_ambient_facts(app: str, window: str) -> list[str]:
    """Pull facts out of app/window titles."""
    facts: list[str] = []
    seen = set()

    if app and app != "unknown":
        facts.append(f"User is using {app}.")
        if app.lower() == "xcode":
            facts.append("User is developing software.")
        if app.lower() in ("mail", "gmail"):
            facts.append("User is checking email.")
        if app.lower() in ("safari", "chrome", "firefox", "brave", "arc", "zen"):
            facts.append("User is browsing the web.")

    for pattern, templates in _AMBIENT_PATTERNS:
        m = pattern.search(window)
        if not m:
            continue
        for tpl in templates:
            rendered = tpl
            for i, g in enumerate(m.groups(), 1):
                rendered = rendered.replace(f"{{g{i - 1}}}", _clean(g))
            rendered = re.sub(r"\{\w+\}", "", rendered)
            if rendered and rendered.lower() not in seen:
                facts.append(rendered)
                seen.add(rendered.lower())

    # If the app is an editor and the window looks like a file path, store a project fact.
    if app.lower() in ("code", "xcode", "nova", "cursor", "zed", "intellij") and window and " - " not in window and " — " not in window:
        facts.append(f"User is editing {window} in {app}.")

    return [f for f in facts if f and "{g" not in f]


class AmbientMemory:
    """Subscribes to badapple_ambient snapshots and writes them to memory."""

    def __init__(self, memory: MemoryGraph, data_dir: Path) -> None:
        self.memory = memory
        self.data_dir = data_dir
        self._last_vlm: float = 0.0
        self._vlm_cooldown = float(os.environ.get("BADAPPLE_AMBIENT_VLM_COOLDOWN", "120"))
        self._vlm_enabled = os.environ.get("BADAPPLE_AMBIENT_VLM", "0") == "1"
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ambient_mem")
        import badapple_ambient

        badapple_ambient.register_snapshot_callback(self._on_snapshot)

    def _on_snapshot(self, context: dict[str, Any]) -> None:
        app = context.get("app", "unknown")
        window = context.get("window", "unknown")
        timestamp = context.get("timestamp", "")
        screen_path = context.get("screen_path")

        facts = extract_ambient_facts(app, window)
        for fact in facts:
            self.memory.remember(fact, source="ambient")

        self.memory.add_episode(
            f"Ambient snapshot at {timestamp}: {app} / {window}",
            "",
            context={
                "app": app,
                "window": window,
                "screen_path": screen_path,
                "source": "ambient",
            },
        )

        if self._vlm_enabled and screen_path:
            now = time.time()
            if now - self._last_vlm >= self._vlm_cooldown:
                self._last_vlm = now
                self._executor.submit(self._describe_and_remember, Path(screen_path), timestamp)

    def _describe_and_remember(self, image_path: Path, timestamp: str) -> None:
        try:
            if not image_path.is_file():
                return
            description = badapple_vision.describe_screen(
                prompt="Describe what the user is doing in one sentence. Mention apps, files, and any visible task.",
                max_tokens=128,
            )
            if description:
                self.memory.remember(f"Ambient screen at {timestamp}: {description}", source="ambient")
        except Exception as e:  # noqa: BLE001
            print(f"[ambient_memory] VLM extraction failed: {e}", flush=True)
