#!/usr/bin/env python3
"""Xcode coding assistant for Bad Apple.

Indexes an Xcode/Swift project into the local RAG pipeline so the assistant
can search, explain, and navigate code without cloud.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Set


CODE_SUFFIXES = {".swift", ".m", ".mm", ".h", ".c", ".cpp", ".metal", ".glsl", ".py", ".rs", ".go", ".java"}


def _find_source_files(project_path: Path) -> List[Path]:
    files: List[Path] = []
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
    # Aggregate into one temp file with headers per source file.
    parts: List[str] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            parts.append(f"--- FILE: {f} ---\n{text}")
        except Exception:
            continue
    if not parts:
        return "Could not read any source files."
    try:
        tmp = Path(tempfile.gettempdir()) / f"badapple_xcode_{p.name}.txt"
        tmp.write_text("\n\n".join(parts), encoding="utf-8")
        indexed = knowledge.index_paths([tmp])
        return f"Indexed {len(files)} files from {p.name} ({indexed} chunks)."
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
