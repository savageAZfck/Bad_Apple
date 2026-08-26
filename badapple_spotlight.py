#!/usr/bin/env python3
"""Universal local Spotlight-style search for Bad Apple.

Searches the macOS Spotlight index (Notes, Mail, local files) and the Bad
Apple history/ledger in one query. No cloud.
"""

import json
import shutil
import subprocess
from pathlib import Path


def _run_mdfind(query: str, max_results: int = 20) -> list[str]:
    if not shutil.which("mdfind"):
        return []
    try:
        result = subprocess.run(
            ["mdfind", query],
            capture_output=True,
            text=True,
            timeout=15,
        check=False)
        if result.returncode != 0:
            return []
        return result.stdout.strip().splitlines()[:max_results]
    except (subprocess.SubprocessError, OSError, ValueError):
        return []


def _search_notes(query: str, max_results: int = 10) -> list[str]:
    """Search macOS Notes via Spotlight."""
    try:
        q = f"(kMDItemTextContent == '*{query}*'c || kMDItemTitle == '*{query}*'c) && kMDItemContentType == 'com.apple.notes.note'"
        result = subprocess.run(
            ["mdfind", q],
            capture_output=True,
            text=True,
            timeout=15,
        check=False)
        return result.stdout.strip().splitlines()[:max_results]
    except (subprocess.SubprocessError, OSError, ValueError):
        return []


def _search_mail(query: str, max_results: int = 10) -> list[str]:
    """Search macOS Mail via Spotlight."""
    try:
        # Try common Mail content types
        results: list[str] = []
        for ct in ("com.apple.mail.email", "com.apple.mail.emlx", "com.apple.mail.message"):
            q = f"(kMDItemTextContent == '*{query}*'c || kMDItemTitle == '*{query}*'c) && kMDItemContentType == '{ct}'"
            result = subprocess.run(
                ["mdfind", q],
                capture_output=True,
                text=True,
                timeout=10,
            check=False)
            results.extend(result.stdout.strip().splitlines())
        return results[:max_results]
    except (subprocess.SubprocessError, OSError, ValueError, LookupError, TypeError):
        return []


def _search_history(query: str, max_results: int = 10) -> list[str]:
    ledger = Path("/var/lib/bad_apple/ledger.jsonl")
    if not ledger.is_file():
        return []
    out = []
    try:
        for line in ledger.read_text(encoding="utf-8", errors="ignore").splitlines():
            if query.lower() in line.lower():
                try:
                    rec = json.loads(line)
                    text = rec.get("prompt", rec.get("text", line[:200]))
                    out.append(f"[history] {rec.get('ts','')} {text[:160]}")
                except json.JSONDecodeError:
                    out.append(line[:160])
            if len(out) >= max_results:
                break
    except (OSError, ValueError) as e:
        print(f"[spotlight] splitlines failed: {e}", flush=True)
    return out


def search(query: str, max_results: int = 20) -> str:
    """Run a universal local search across Mail, Notes, files, and Bad Apple history."""
    if not query:
        return "Error: query is required"
    notes = _search_notes(query, max_results=max_results // 3)
    mail = _search_mail(query, max_results=max_results // 3)
    files = _run_mdfind(query, max_results=max_results // 3)
    history = _search_history(query, max_results=max_results // 3)
    results = {
        "query": query,
        "notes": notes,
        "mail": mail,
        "files": files,
        "bad_apple_history": history,
    }
    # Build concise summary
    lines = [f"Spotlight results for '{query}':"]
    for key, values in results.items():
        if key == "query":
            continue
        if values:
            lines.append(f"\n{key.upper()} ({len(values)}):")
            for v in values[:max_results]:
                lines.append(f"  - {v}")
    if len(lines) == 1:
        lines.append("Nothing found locally.")
    return "\n".join(lines)
