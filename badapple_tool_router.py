#!/usr/bin/env python3
"""Natural-language tool router for Bad Apple.

Maps free-form user prompts to concrete tool calls so the 9B brain is only
woken when the request is genuinely ambiguous.  Uses a fast keyword/regex
pass, then a BGE-small semantic fallback against a bank of example phrases.
"""

import json
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# Tool-name -> list of example natural-language prompts.
# These are the "ground-truth" training phrases the router compares against.
TOOL_EXAMPLES: dict[str, list[str]] = {
    "list_directory": [
        "list the files in my documents",
        "show me what's in /tmp",
        "what files are in this directory",
        "list files in my home folder",
        "show the contents of /Users/savag3",
    ],
    "read_file": [
        "read the file /etc/hosts",
        "show me the contents of my todo list",
        "open the note about ideas",
        "what does the readme say",
        "read /var/log/bad_apple_mlx_server.log",
    ],
    "write_file": [
        "write a file called notes.txt with hello world",
        "save this to my scratchpad",
        "create a file named todo.txt",
        "write the following to a file",
    ],
    "run_shell": [
        "run shell ls /tmp",
        "execute df -h",
        "run ps aux",
        "show me the output of uname -a",
    ],
    "run_applescript": [
        "run applescript to tell application finder to activate",
        "execute applescript display notification hello",
    ],
    "search_content": [
        "search for todo in my notes",
        'grep for "error" in /var/log',
        "find files containing the word password",
    ],
    "search_local_files": [
        "find my resume on the mac",
        "spotlight search for budget",
        "mdfind the file called screenshot",
    ],
    "index_documents": [
        "index the documents in ~/Documents",
        "add this folder to the knowledge base",
        "index /Users/savag9/bad_apple",
    ],
    "git_status": [
        "what is the git status of the current repo",
        "show me git status",
    ],
    "git_diff": [
        "show me the git diff",
        "what changed in the last commit",
    ],
    "git_log": [
        "show me the git log",
        "recent commits",
    ],
    "git_commit": [
        "commit these changes with message update",
        "git commit -m 'fix things'",
    ],
    "read_working_memory": [
        "read my working memory",
        "what is in my scratchpad",
    ],
    "write_working_memory": [
        "write to working memory: focus on feature x",
        "add this to my scratchpad",
    ],
    "clear_working_memory": [
        "clear my working memory",
        "erase the scratchpad",
    ],
    "screen_capture": [
        "capture my screen",
        "take a screenshot",
    ],
    "capture_and_describe_screen": [
        "what is on my screen",
        "describe my screen",
        "what do you see on the display",
    ],
    "capture_and_extract_screen": [
        "extract text from my screen",
        "read the text on my screen",
    ],
    "describe_image": [
        "describe this image /path/to/photo.png",
        "what is in the picture",
    ],
    "extract_text_from_image": [
        "extract text from /path/to/receipt.png",
        "ocr this image",
    ],
    "workspace_status": [
        "what is the project status",
        "workspace status",
        "tell me about the active project",
    ],
    "today_events": [
        "what meetings do I have today",
        "what is on my calendar today",
        "today's events",
    ],
    "upcoming_events": [
        "what meetings do I have this week",
        "upcoming events",
    ],
    "list_reminders": [
        "list my reminders",
        "what are my reminders",
    ],
    "add_reminder": [
        "add a reminder: call mom",
        "remind me to pick up milk",
    ],
    "unread_emails": [
        "do I have unread emails",
        "check my mail",
    ],
    "search_mail": [
        "search my email for invoices",
        "find emails from adam",
    ],
    "consolidate_memory": [
        "consolidate my memory",
        "run dream mode",
        "offline consolidation",
    ],
    "p2p_sync": [
        "p2p sync",
        "sync my memory to peers",
    ],
    "p2p_peers": [
        "list p2p peers",
        "who are my peers",
    ],
    "list_mcp_servers": [
        "list mcp servers",
        "what mcp servers do I have",
    ],
    "add_mcp_server": [
        "add mcp server time python3 -m mcp_server_time",
        "register mcp filesystem npx -y @modelcontextprotocol/server-filesystem",
    ],
    "remove_mcp_server": [
        "remove mcp server test",
        "delete mcp server time",
    ],
    "list_mcp_tools": [
        "list tools from mcp server time",
        "what tools does mcp server time have",
    ],
    "invoke_mcp_tool": [
        "invoke mcp tool get_current_time on server time",
        "use mcp server time tool get_current_time",
    ],
    "generate_image": [
        "generate an image of a cat",
        "draw a picture of a sunset",
        "make an image of a robot",
        "create an image of a mountain landscape",
        "image generation for a logo",
    ],
}


# Keyword regex patterns.  These win immediately, no embedding needed.
TOOL_KEYWORDS: dict[str, list[re.Pattern]] = {
    "list_directory": [re.compile(r"\b(list|show)\b.*\bfiles\b|\blist files in\b|\blist the files\b", re.IGNORECASE)],
    "read_file": [re.compile(r"\b(read|open)\b.*\bfile\b|\bread\s+[~./]\S+", re.IGNORECASE)],
    "write_file": [re.compile(r"\b(write|create)\b.*\bfile\b|\bfile\s+named?\s+\S+", re.IGNORECASE)],
    "run_shell": [re.compile(r"\b(run|execute)\b.*\bshell\b|\brun\s+(?:command|ls|cat|ps|df|du|find|grep|mdfind)\b", re.IGNORECASE)],
    "run_applescript": [re.compile(r"\brun\b.*\bapplescript\b", re.IGNORECASE)],
    "search_content": [re.compile(r"\bsearch\b.*\bfor\b.*\bin\b|\bgrep\b|\bfind.*containing\b", re.IGNORECASE)],
    "search_local_files": [re.compile(r"\bmdfind\b|\bspotlight\b|\bfind my\b|\bfind (?:this|the) file\b", re.IGNORECASE)],
    "index_documents": [re.compile(r"\bindex\b.*\bdocument|\bindex\s+[~./]", re.IGNORECASE)],
    "git_status": [re.compile(r"\bgit\s+status\b", re.IGNORECASE)],
    "git_diff": [re.compile(r"\bgit\s+diff\b", re.IGNORECASE)],
    "git_log": [re.compile(r"\bgit\s+log\b", re.IGNORECASE)],
    "git_commit": [re.compile(r"\bgit\s+commit\b", re.IGNORECASE)],
    "read_working_memory": [re.compile(r"\bread\b.*\bworking memory\b|\bworking memory\b", re.IGNORECASE)],
    "write_working_memory": [re.compile(r"\bwrite\b.*\bworking memory\b|\badd.*scratchpad\b", re.IGNORECASE)],
    "clear_working_memory": [re.compile(r"\bclear\b.*\bworking memory\b|\bclear\s+scratchpad\b", re.IGNORECASE)],
    "screen_capture": [re.compile(r"\bcapture\b.*\bscreen\b|\bscreenshot\b", re.IGNORECASE)],
    "capture_and_describe_screen": [re.compile(r"\bwhat\b.*\bon my screen\b|\bdescribe my screen\b", re.IGNORECASE)],
    "capture_and_extract_screen": [re.compile(r"\bextract\b.*\btext\b.*\bscreen\b", re.IGNORECASE)],
    "describe_image": [re.compile(r"\bdescribe\b.*\bimage\b|\bdescribe\s+[~./]\S+\.(?:png|jpg|jpeg)\b", re.IGNORECASE)],
    "extract_text_from_image": [re.compile(r"\bextract\b.*\btext\b.*\bimage\b|\bocr\b", re.IGNORECASE)],
    "workspace_status": [re.compile(r"\bworkspace status\b|\bproject status\b", re.IGNORECASE)],
    "today_events": [re.compile(r"\b(today'?s? events?|events? today|meetings today|what.*on my calendar today)\b", re.IGNORECASE)],
    "upcoming_events": [re.compile(r"\bupcoming\b.*\bevents?\b|\bnext week\b.*\bcalendar\b", re.IGNORECASE)],
    "list_reminders": [re.compile(r"\b(list|what are)\b.*\breminders\b", re.IGNORECASE)],
    "add_reminder": [re.compile(r"\b(add|create)\b.*\breminder\b|\bremind me\b", re.IGNORECASE)],
    "unread_emails": [re.compile(r"\bunread\b.*\b(email|mail)\b|\bcheck my mail\b", re.IGNORECASE)],
    "search_mail": [re.compile(r"\bsearch\b.*\bmail\b|\bsearch\b.*\bemail\b", re.IGNORECASE)],
    "consolidate_memory": [re.compile(r"\bconsolidate\b.*\bmemory\b|\bdream\b|\boffline consolidation\b", re.IGNORECASE)],
    "p2p_sync": [re.compile(r"\bp2p\s+sync\b|\bsync\b.*\bpeers\b", re.IGNORECASE)],
    "p2p_peers": [re.compile(r"\bp2p\s+peers\b|\blist\s+peers\b", re.IGNORECASE)],
    "generate_image": [re.compile(r"\b(generate|draw|make|create)\b.*\bimage\b|\bimage generation\b", re.IGNORECASE)],
    "list_mcp_servers": [re.compile(r"\b(list|show)\b.*\bmcp\s+servers?\b", re.IGNORECASE)],
    "add_mcp_server": [re.compile(r"\b(add|register)\b.*\bmcp\s+server\b", re.IGNORECASE)],
    "remove_mcp_server": [re.compile(r"\b(remove|delete)\b.*\bmcp\s+server\b", re.IGNORECASE)],
    "list_mcp_tools": [re.compile(r"\b(list|show)\b.*\bmcp\s+tools?\b", re.IGNORECASE)],
    "invoke_mcp_tool": [re.compile(r"\b(invoke|use|call)\b.*\bmcp\s+tool\b", re.IGNORECASE)],
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return float(np.max(a @ b.T))


class ToolRouter:
    """Maps natural-language prompts to tool calls."""

    SEMANTIC_THRESHOLD = float(os.environ.get("BADAPPLE_TOOL_ROUTER_THRESHOLD", "0.88"))
    MAX_EXAMPLES = 32

    def __init__(
        self,
        data_dir: Path,
        encoder: Callable[[list[str]], Any] | None = None,
    ):
        self.data_dir = data_dir
        self.encoder = encoder
        self._lock = threading.RLock()
        self._learned_path = data_dir / "tool_router.json"
        self._examples: dict[str, list[str]] = {k: list(v) for k, v in TOOL_EXAMPLES.items()}
        self._embeddings: dict[str, np.ndarray] = {}
        self._example_texts: dict[str, list[str]] = {}
        self._load_learned()
        self._rebuild_embeddings()

    def _load_learned(self) -> None:
        if not self._learned_path.is_file():
            return
        try:
            with self._learned_path.open("r", encoding="utf-8") as f:
                learned = json.load(f)
            for name, phrases in learned.items():
                if name in self._examples:
                    for p in phrases:
                        if p not in self._examples[name]:
                            self._examples[name].append(p)
        except Exception as e:
            print(f"[tool_router] could not load learned examples: {e}", flush=True)

    def _save_learned(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with self._learned_path.open("w", encoding="utf-8") as f:
                json.dump(self._examples, f, indent=2)
        except Exception as e:
            print(f"[tool_router] could not save learned examples: {e}", flush=True)

    def _rebuild_embeddings(self) -> None:
        """Pre-compute normalized example embeddings."""
        if self.encoder is None:
            return
        with self._lock:
            self._embeddings = {}
            self._example_texts = {}
            for name, phrases in self._examples.items():
                texts = phrases[-self.MAX_EXAMPLES:]
                self._example_texts[name] = texts
                try:
                    emb = np.asarray(self.encoder(texts))
                    self._embeddings[name] = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
                except Exception as e:
                    print(f"[tool_router] failed to embed {name}: {e}", flush=True)

    def record_success(self, prompt: str, tool_name: str) -> None:
        """Learn that this prompt maps to a tool."""
        if not prompt or not tool_name or tool_name not in self._examples:
            return
        with self._lock:
            if prompt not in self._examples[tool_name]:
                self._examples[tool_name].append(prompt)
                if len(self._examples[tool_name]) > self.MAX_EXAMPLES:
                    self._examples[tool_name].pop(0)
                self._save_learned()
                self._rebuild_embeddings()

    def _keyword_match(self, prompt: str) -> tuple[str, float] | None:
        low = prompt.lower()
        for name, patterns in TOOL_KEYWORDS.items():
            for pat in patterns:
                if pat.search(low):
                    return name, 1.0
        return None

    def _semantic_match(self, prompt: str) -> tuple[str, float] | None:
        if self.encoder is None or not self._embeddings:
            return None
        try:
            q = np.asarray(self.encoder([prompt]))
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)
            best_name: str | None = None
            best_score = -1.0
            for name, emb in self._embeddings.items():
                sim = float(np.max(q @ emb.T))
                if sim > best_score:
                    best_score = sim
                    best_name = name
            if best_name and best_score >= self.SEMANTIC_THRESHOLD:
                return best_name, best_score
        except Exception as e:
            print(f"[tool_router] semantic match failed: {e}", flush=True)
        return None

    def _extract_args(self, tool_name: str, prompt: str) -> dict[str, Any]:
        low = prompt.lower().strip()

        # Path extraction helper.
        def _path_arg() -> str | None:
            # Look for an absolute or tilde path, or a simple filename.
            m = re.search(r"(?:file|path|directory|folder|in|from|to|at)\s+['\"]?([~./]?[\w\-./]+(?:/[\w\-./]+)*)['\"]?", low)
            if m:
                return _resolve_common_path(m.group(1))
            m = re.search(r"[~./][\w\-./]+(?:/[\w\-./]+)*", low)
            if m:
                return _resolve_common_path(m.group(0))
            return None

        if tool_name == "list_directory":
            p = _path_arg()
            return {"path": p if p else "."}

        if tool_name == "read_file":
            p = _path_arg()
            return {"path": p if p else ""}

        if tool_name == "write_file":
            m = re.search(r"(?:file|named?|called)\s+['\"]?([\w\-./]+)['\"]?", low)
            filename = m.group(1) if m else "note.txt"
            m = re.search(r"(?:with|containing|that says|:\s*)(.+)$", low)
            content = m.group(1).strip("\"'").strip() if m else ""
            return {"filename": filename, "content": content}

        if tool_name in ("run_shell", "run_applescript"):
            m = re.search(r"\b(?:run|execute)\s+(?:shell|applescript)\s+(.+)$", low, re.IGNORECASE)
            if m:
                return {"command": m.group(1).strip(), "timeout": 15}
            # Fallback: if prompt has a bare command pattern, use the rest.
            m = re.search(r"\b(?:run|execute)\s+(.+)$", low, re.IGNORECASE)
            if m:
                return {"command": m.group(1).strip(), "timeout": 15}
            return {"command": "", "timeout": 15}

        if tool_name == "search_content":
            m = re.search(r'["\']([^"\']+)["\']', prompt)
            query = m.group(1) if m else ""
            p = _path_arg()
            return {"query": query, "path": p if p else "."}

        if tool_name == "search_local_files":
            m = re.search(r'["\']([^"\']+)["\']', prompt)
            return {"query": m.group(1) if m else prompt}

        if tool_name == "index_documents":
            p = _path_arg()
            return {"path": p if p else "."}

        if tool_name == "git_commit":
            m = re.search(r'["\']([^"\']+)["\']', prompt)
            return {"message": m.group(1) if m else "Update"}

        if tool_name == "write_working_memory":
            m = re.search(r"(?:working memory|scratchpad)\s*[:-]?\s*['\"]?(.+)['\"]?$", low, re.IGNORECASE)
            return {"content": m.group(1).strip("\"'").strip() if m else ""}

        if tool_name == "add_reminder":
            m = re.search(r"(?:remind me to|reminder:?|add a reminder:?|remind me:)\s*(.+?)(?:\s+(?:at|on|in)\s+|$)", low, re.IGNORECASE)
            return {"title": m.group(1).strip("\"'").strip() if m else prompt}

        if tool_name in ("describe_image", "extract_text_from_image"):
            p = _path_arg()
            return {"path": p if p else ""}

        if tool_name == "today_events":
            m = re.search(r"\b(\d+)\s*days?\b", low)
            return {"days": int(m.group(1)) if m else 0}

        if tool_name == "upcoming_events":
            m = re.search(r"\b(\d+)\s*days?\b", low)
            return {"days": int(m.group(1)) if m else 7}

        if tool_name == "search_mail":
            m = re.search(r'["\']([^"\']+)["\']', prompt)
            return {"query": m.group(1) if m else prompt}

        if tool_name == "generate_image":
            # Extract the prompt phrase after 'image of', 'image:', 'draw', etc.
            m = re.search(r"\b(?:image\s+(?:of|with)?|generate|draw|make|create)\b[\s:]*(an?\s+)?(?:image\b[\s:]*)?(.+?)(?:\.|$)", low, re.IGNORECASE)
            if not m:
                m = re.search(r"\b(?:image\s*[:\-]?\s*)?(.+?)(?:\.|$)", low, re.IGNORECASE)
            prompt_text = m.group(2).strip() if m else low
            prompt_text = re.sub(r"^(of|with|a|an)\s+", "", prompt_text, flags=re.IGNORECASE)
            return {"prompt": prompt_text}

        if tool_name == "add_mcp_server":
            m = re.search(r"\bmcp\s+server\s+(\S+)\s+(.+)$", low, re.IGNORECASE)
            if m:
                return {"name": m.group(1).strip(), "command": m.group(2).strip()}
            m = re.search(r"(?:add|register)\s+mcp\s+server\s+(\S+)\s+(.+)$", low, re.IGNORECASE)
            return {"name": m.group(1).strip() if m else "", "command": m.group(2).strip().rstrip(".!?,;:") if m else ""}

        if tool_name == "remove_mcp_server":
            m = re.search(r"\bmcp\s+server\s+(\S+)(?:\s*\.|$)", low, re.IGNORECASE)
            if m:
                return {"name": m.group(1).strip()}
            m = re.search(r"(?:remove|delete)\s+mcp\s+server\s+(\S+)(?:\s*\.|$)", low, re.IGNORECASE)
            return {"name": m.group(1).strip() if m else ""}

        if tool_name == "list_mcp_tools":
            m = re.search(r"\btools?\s+(?:from\s+server\s+|on\s+server\s+|of\s+server\s+)?(\S+)(?:\s+server\b)?(?:\.|$)", low, re.IGNORECASE)
            if m:
                return {"server": m.group(1).strip()}
            m = re.search(r"\bserver\s+(\S+)\b", low, re.IGNORECASE)
            return {"server": m.group(1).strip() if m else ""}

        if tool_name == "invoke_mcp_tool":
            m = re.search(r"\btool\s+(\S+)\s+on\s+server\s+(\S+)\s*(?:with|using|args:?)?\s*(.+?)?(?:\s*\.|$)", low, re.IGNORECASE)
            if m:
                args_text = (m.group(3) or "").strip()
                arguments = {}
                if args_text:
                    kv = re.search(r"(\S+)\s+(.+)$", args_text, re.IGNORECASE)
                    if kv:
                        arguments[kv.group(1)] = kv.group(2).strip('"\'')
                    else:
                        arguments["text"] = args_text
                return {"tool": m.group(1).strip(), "server": m.group(2).strip(), "arguments": arguments}
            m = re.search(r"\bserver\s+(\S+)\s+tool\s+(\S+)\s*(?:with|using|args:?)?\s*(.+?)?(?:\s*\.|$)", low, re.IGNORECASE)
            if m:
                args_text = (m.group(3) or "").strip()
                arguments = {}
                if args_text:
                    kv = re.search(r"(\S+)\s+(.+)$", args_text, re.IGNORECASE)
                    if kv:
                        arguments[kv.group(1)] = kv.group(2).strip('"\'')
                    else:
                        arguments["text"] = args_text
                return {"server": m.group(1).strip(), "tool": m.group(2).strip(), "arguments": arguments}
            return {"server": "", "tool": ""}

        return {}

    def resolve(self, prompt: str) -> tuple[str, dict[str, Any], float] | None:
        """Return (tool_name, args, confidence) or None to fall through to 9B."""
        kw = self._keyword_match(prompt)
        if kw:
            name, conf = kw
            return name, self._extract_args(name, prompt), conf

        sem = self._semantic_match(prompt)
        if sem:
            name, conf = sem
            return name, self._extract_args(name, prompt), conf

        return None


def _resolve_common_path(raw: str) -> str:
    low = raw.lower().strip().rstrip(".!?,;:\"'")
    if low in ("my home", "my home directory", "home", "home directory"):
        return "~"
    if low in ("this directory", "current directory", "here", "."):
        return "."
    if low in ("my documents", "documents"):
        return "~/Documents"
    if low in ("my downloads", "downloads"):
        return "~/Downloads"
    if low in ("my desktop", "desktop"):
        return "~/Desktop"
    return raw.strip()
