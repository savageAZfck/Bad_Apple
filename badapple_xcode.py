#!/usr/bin/env python3
"""Xcode coding assistant for Bad Apple.

Indexes an Xcode/Swift project into the local RAG pipeline so the assistant
can search, explain, and navigate code without cloud.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

CODE_SUFFIXES = {".swift", ".m", ".mm", ".h", ".c", ".cpp", ".metal", ".glsl", ".py", ".rs", ".go", ".java"}


def _find_source_files(project_path: Path) -> list[Path]:
    files: list[Path] = []
    for root, _dirs, names in os.walk(project_path):
        for name in names:
            p = Path(root) / name
            if p.suffix.lower() in CODE_SUFFIXES:
                files.append(p)
    return sorted(files)


def index_project(project_path: str, knowledge: Any) -> str:
    """Index all source files in an Xcode project."""
    p = Path(project_path).expanduser()
    if not p.is_dir():
        return f"Error: {p} is not a directory"
    files = _find_source_files(p)
    if not files:
        return "No recognized source files found."
    try:
        indexed = knowledge.index_paths(files)
        return f"Indexed {len(files)} files from {p.name} ({indexed} chunks) with per-file source paths."
    except Exception as e:
        return f"Xcode index error: {e}"


def search_project(query: str, knowledge: Any, max_results: int = 10) -> str:
    """Search the indexed Xcode project for relevant code."""
    try:
        results = knowledge.search(query, k=max_results)
        if not results:
            return "No matching code found."
        lines = [f"Xcode search results for '{query}':"]
        for i, (text, score) in enumerate(results, 1):
            lines.append(f"\n{i}. score={score:.3f}\n{text[:500]}")
        return "\n".join(lines)
    except Exception as e:
        return f"Xcode search error: {e}"


def project_info(project_path: str) -> str:
    path = Path(project_path).expanduser().resolve()
    projects = [path] if path.suffix == ".xcodeproj" else sorted(path.glob("*.xcodeproj"))
    if not projects:
        return f"No .xcodeproj found under {path}"
    try:
        result = subprocess.run(
            ["xcodebuild", "-list", "-json", "-project", str(projects[0])],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return f"xcodebuild error: {result.stderr or result.stdout}"
        return json.dumps(json.loads(result.stdout), indent=2)
    except Exception as e:
        return f"Xcode project info error: {e}"


def build_diagnostics(project_path: str, scheme: str, configuration: str = "Debug") -> str:
    path = Path(project_path).expanduser().resolve()
    projects = [path] if path.suffix == ".xcodeproj" else sorted(path.glob("*.xcodeproj"))
    if not projects:
        return f"No .xcodeproj found under {path}"
    try:
        result = subprocess.run(
            [
                "xcodebuild",
                "-project",
                str(projects[0]),
                "-scheme",
                scheme,
                "-configuration",
                configuration,
                "build",
                "CODE_SIGNING_ALLOWED=NO",
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        output = (result.stdout or "") + (result.stderr or "")
        relevant = [line for line in output.splitlines() if any(token in line for token in ("error:", "warning:", "BUILD "))]
        return "\n".join(relevant[-300:]) or output[-12000:]
    except subprocess.TimeoutExpired:
        return "Xcode build diagnostics timed out."
    except Exception as e:
        return f"Xcode build diagnostics error: {e}"
