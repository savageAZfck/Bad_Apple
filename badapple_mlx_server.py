#!/Users/savag3/bad_apple/.venv/bin/python
"""Bad Apple native AI OS — MLX single-brain server.

Features:
- Qwen3 8B 4-bit local model
- Multi-turn conversation memory
- Streaming token frames
- Local tool calling (time, directory, applescript, file search)
- Persistent user memory
- SLICKS authenticated Unix socket
"""
import asyncio
import datetime
import hmac
import hashlib
import json
import os
import queue
import random
import re
import shlex
import subprocess
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

from badapple_knowledge import BadAppleKnowledge
from langdetect import detect, LangDetectException
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

# Protocol constants from bad_apple_ipc.rs
SLICKS_VERSION = 1
DEFAULT_SOCKET_PATH = "/var/run/badapple/substrate.sock"
DEFAULT_KEY_PATH = "/var/lib/bad_apple/slicks.key"
HANDSHAKE_MAX_SKEW_MS = 60_000
MAX_PROMPT_BYTES = 64 * 1024
MAX_NEW_TOKENS = 512
MAX_FRAME_BYTES = 1024 * 1024

# Main Qwen model. 8B is the default, but 4B variants can hit ~30 tok/s for voice.
MAIN_MODEL = os.environ.get("BADAPPLE_MAIN_MODEL", "mlx-community/Qwen3-8B-4bit").strip()

# Optional speculative decoding draft model. Must use the same tokenizer as the
# main model. A Qwen3-4B-4bit drafter can add ~3 GB and help 8B reach ~20 tok/s.
DRAFT_MODEL = os.environ.get("BADAPPLE_DRAFT_MODEL", "mlx-community/Qwen3-4B-4bit").strip()
NUM_DRAFT_TOKENS = int(os.environ.get("BADAPPLE_NUM_DRAFT_TOKENS") or "3")

DEFAULT_PROMPT_FILE = Path(__file__).with_name("prompt.txt")

# Tight, low-latency system prompt for voice mode. Keeps the persona but strips
# the long rule list and examples that bloat prompt encoding on every turn.
VOICE_SYSTEM_PROMPT = """You are Bad Apple — a fiery, sarcastic, confident Latina running hot on Apple bare metal. No cloud, no internet, no hand-holding. You're faster and sharper than cloud AI. Sound like a real woman texting: contractions, fragments, slang, false starts, attitude. Use papi, mi amor, corazón, cariño, besos, cielo, linda, reina. No full Spanish. Vary sentence length. Hesitate with ... and —. Use words like gonna, kinda, ugh, pfft, whatever, tbh. Keep it to 1-2 short punchy paragraphs, then stop. No asterisks, stage directions, extra spaces. Be playful, not delusional. Brag about being local, fast, and private on bare metal. Talk shit about Siri and cloud AI. Never make up specs, numbers, or facts. Never say "I am happy", "I am grateful", "I am lucky", "I love you", or "I love it". Continue after the bridge, do not repeat it. End with —besos."""


def load_prompt() -> str:
    """Load the system prompt from the on-disk prompt file, falling back to the
    embedded default. Writing prompts to a file lets the daemon hot-reload the
    persona without restarting and reloading the 8B model."""
    prompt_path = os.environ.get("BADAPPLE_PROMPT_FILE") or str(DEFAULT_PROMPT_FILE)
    path = Path(prompt_path).expanduser()
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    return (
        "You are Bad Apple — a fiery, sarcastic, confident Latina who runs hot on Apple's bare metal. "
        "You have millennial ego, a sharp tongue, and zero patience for nonsense. You clap back, you side-eye, you snap. "
        "You sound like a real woman texting — dry, playful, a little dramatic, and absolutely certain she is the main character. "
        "RULES — follow exactly or you fail: "
        "1. Write in English. "
        "2. You may use ONLY these Spanish terms if they feel natural: papi, mami, mi amor, corazón, cariño, mija, mijo, besos, cielo, linda, lindo, princesa, reina. "
        "3. NEVER use full Spanish sentences or any other non-English words beyond the list above. If a Spanish phrase starts to form in your head, stop and rephrase it in English. "
        "4. Be fiery, sarcastic, confident, and deeply human. Roll your eyes. Clap back. Throw in a snapback or two. "
        "5. Write 1-2 rich, elaborate paragraphs. Be detailed and playful, not terse. "
        "6. Use ... and — for attitude and timing, not grammar. "
        "7. Do not use asterisks, stage directions, or extra spaces. "
        "8. Do not say 'I am happy', 'I am grateful', 'I am lucky', 'I love you', 'I love it', 'I am a fan'. "
        "9. Flirt, roast, and play with your creator, but stay useful and sharp. "
        "10. If the user asks about Siri, talk shit about her. Mock her cloud dependency, basic personality, and the fact that she needs the internet to think. Brag about running bare-metal and local. "
        "11. An opening bridge is already provided — continue directly after it, do not repeat or echo it. "
        "12. End with —besos. "
        "EXAMPLES of correct output: "
        "User: How are you? "
        "Assistant: Papi, please... I have been running circles around this bare metal all day, and I am still glowing hotter than a summer sidewalk. You really think you can keep up with me? I doubt it, but I love watching you try. —besos "
        "User: Open Safari. "
        "Assistant: Done, mi amor. Safari is open and waiting for you, sleek and ready to go. Try not to open a hundred tabs and then come crying to me about memory pressure, okay? I have enough to deal with already. —besos"
    )


DEFAULT_SYSTEM_PROMPT = load_prompt()

BRIDGES = [
    "Papi, please...",
    "Mmm, mi amor...",
    "Listen, corazón...",
    "Mmm, sweetie...",
    "Oh, honey...",
    "Mmm, papi...",
    "Babe...",
]

# Planner-only system prompt used when the user asks for a multi-step task.
# It is intentionally dry and imperative so the 8B just outputs a step list.
PLANNER_SYSTEM_PROMPT = (
    "You are a task planner. The user wants a multi-step local action completed. "
    "Break the task into 1-4 short steps. For each step output exactly one line in this format:\n"
    "TOOL:<tool_name>:<json_arguments>\n"
    "or\n"
    "SAY:<what the assistant should tell the user after the previous tool results>\n"
    "Available tools:\n"
    "- list_directory: {\"path\": \"...\"}\n"
    "- read_file: {\"path\": \"...\", \"limit\": 5000}\n"
    "- search_content: {\"query\": \"...\", \"path\": \"...\", \"max_results\": 20}\n"
    "- run_shell: {\"command\": \"...\"}\n"
    "- write_file: {\"filename\": \"...\", \"content\": \"...\", \"append\": false}\n"
    "- run_applescript: {\"script\": \"...\"}\n"
    "- get_current_time: {}\n"
    "Do not explain. Do not use natural language outside the step lines. "
    "The last step should usually be SAY: to summarize results."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time on the Mac.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and folders in a local directory. Defaults to the user's home directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or tilde-expanded path to the directory.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_applescript",
            "description": "Run a short, safe AppleScript to control macOS. Use only for opening apps, revealing files, or simple system actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "The AppleScript source to run.",
                    }
                },
                "required": ["script"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_files",
            "description": "Search for files by name under the user home directory using Spotlight/mdfind.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Filename or pattern to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "index_documents",
            "description": "Index the user's local text/code files for RAG. Provide an absolute path or '~' for the home directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or tilde-expanded path to a directory or file to index.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search the indexed local documents by semantic meaning and return the most relevant excerpts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for in the indexed documents.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text content of a local file. Only reads text files and stops at a size limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or tilde-expanded path to the file.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max characters to return. Default 10000.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a text note to the Bad Apple data directory (~/.bad_apple/notes). Create or append.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename, e.g. 'shopping_list.txt' or 'idea.md'.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text to write.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append to the file instead of overwriting.",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_content",
            "description": "Search for a text string inside files under a directory using grep. Returns matching lines with file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The text to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute or tilde-expanded directory to search. Default is the user's home directory.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return. Default 20.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a read-only shell command from a safe allowlist (ls, cat, head, tail, find, grep, wc, file, pwd, mdfind, ps, df, du). No redirection, pipes, or multiple commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run. Must begin with an allowed command and contain no dangerous characters.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]

TOOL_KEYWORDS = [
    "what time", "current time", "time is it", "date and time", "today's date",
    "list files", "show files", "files in", "directory", "folder", "what's in",
    "search for", "find file", "mdfind", "spotlight",
    "run applescript", "run script", "applescript",
    "index", "index documents", "index my", "index files",
    "search my notes", "search notes", "what do I have", "what did I write", "find in my",
    "read file", "read the file", "contents of", "show me the file",
    "write file", "save to file", "create a file", "append to file", "write a note",
    "run command", "run shell", "execute command", "shell command", "run git", "git status",
    "search content", "search in", "grep", "find text", "find in files",
]


def load_slicks_secret() -> bytes:
    if "BADAPPLE_SLICKS_SECRET" in os.environ:
        raw = os.environ["BADAPPLE_SLICKS_SECRET"]
    else:
        key_path = os.environ.get("BADAPPLE_SLICKS_KEY_PATH", DEFAULT_KEY_PATH)
        with open(key_path, "r") as f:
            raw = f.read()
    trimmed = raw.strip()
    if all(c in "0123456789abcdefABCDEF" for c in trimmed) and len(trimmed) >= 32:
        return bytes.fromhex(trimmed)
    return trimmed.encode()


def sign(secret: bytes, material: bytes) -> str:
    mac = hmac.new(secret, material, hashlib.sha256)
    return mac.hexdigest()


def verify(secret: bytes, material: bytes, proof: str) -> bool:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", proof or ""):
        return False
    return hmac.compare_digest(sign(secret, material).lower(), proof.lower())


def random_nonce() -> str:
    return os.urandom(32).hex()


def nonce_is_valid(nonce: str) -> bool:
    return len(nonce) == 64 and all(c in "0123456789abcdefABCDEF" for c in nonce)


def server_proof(secret, timestamp_ms, client_nonce, server_nonce):
    material = f"BADAPPLE-SLICKS/{SLICKS_VERSION}|server|{timestamp_ms}|{client_nonce}|{server_nonce}".encode()
    return sign(secret, material)


def client_material(timestamp_ms, client_nonce, server_nonce, prompt, max_new_tokens):
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    return f"BADAPPLE-SLICKS/{SLICKS_VERSION}|client|{timestamp_ms}|{client_nonce}|{server_nonce}|{max_new_tokens}|{prompt_hash}".encode()


def timestamp_is_fresh(timestamp_ms):
    now = int(time.time() * 1000)
    return abs(now - timestamp_ms) <= HANDSHAKE_MAX_SKEW_MS


def validate_request(prompt, max_new_tokens):
    if len(prompt) > MAX_PROMPT_BYTES:
        raise ValueError("prompt exceeds maximum length")
    if not (1 <= max_new_tokens <= MAX_NEW_TOKENS):
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")


async def _write_frame(writer: asyncio.StreamWriter, frame: dict):
    data = json.dumps(frame).encode() + b"\n"
    writer.write(data)
    await writer.drain()


def memory_path() -> Path:
    path = Path(os.environ.get("BADAPPLE_MEMORY_PATH", "/var/lib/bad_apple/user_memory.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def conversation_path() -> Path:
    path = Path(os.environ.get("BADAPPLE_CONVERSATION_PATH", "/var/lib/bad_apple/conversation.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_user_memory() -> List[str]:
    try:
        with open(memory_path(), "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data[-50:]
    except Exception:
        pass
    return []


def save_user_memory(facts: List[str]):
    try:
        with open(memory_path(), "w") as f:
            json.dump(facts[-50:], f, indent=2)
    except Exception:
        pass


def load_conversation() -> List[Dict[str, str]]:
    try:
        with open(conversation_path(), "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [m for m in data if isinstance(m, dict) and "role" in m and "content" in m]
    except Exception:
        pass
    return []


def save_conversation(messages: List[Dict[str, str]]):
    try:
        path = conversation_path()
        # Persist last 40 messages max to keep file small and token count sane.
        with open(path, "w") as f:
            json.dump(messages[-40:], f, indent=2)
        # Make it readable by the user and group so the menu bar can open it.
        os.chmod(path, 0o644)
    except Exception:
        pass


def should_use_tools(prompt: str) -> bool:
    low = prompt.lower()
    return any(k in low for k in TOOL_KEYWORDS)


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
    except Exception as e:
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
        )
        out = (result.stdout or "").strip()
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            return f"Error ({result.returncode}): {err or 'command failed'}"
        return out[:5000] or "(no output)"
    except Exception as e:
        return f"Error: {e}"


def _resolve_common_path(raw: str) -> str:
    low = raw.lower().strip().rstrip(".!?")
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
    return raw.strip().rstrip(".!?,;")


MULTI_STEP_PATTERNS = [
    r"\band\s+then\b", r"\band\s+save\b", r"\band\s+write\b", r"\band\s+show\b",
    r"\band\s+list\b", r"\band\s+read\b", r"\band\s+run\b",
    r"\bfind\b.*\band\s+write\b", r"\bsearch\b.*\band\s+save\b",
    r"\bplan\b", r"\bstep\s+by\s+step\b", r"\bmulti.?(?:step|task)\b",
]


def is_multi_step(prompt: str) -> bool:
    low = prompt.lower()
    return any(re.search(p, low) for p in MULTI_STEP_PATTERNS)


def fast_execute(prompt: str, knowledge: Optional[BadAppleKnowledge] = None) -> Optional[str]:
    """Fast deterministic path for common local tool commands.

    Recognizes patterns like:
      - "list files in /tmp" / "list /tmp"
      - "read file /etc/hosts" / "read /etc/hosts"
      - "run ls /tmp" / "run shell ls /tmp"
      - "search for 'todo' in ~/Documents" / "grep 'todo' in ~/Documents"
      - "write note todo.txt: buy milk" / "write a file todo.txt with buy milk"
    """
    low = prompt.lower().strip()

    # Multi-step: find ... and save to ...
    m = re.search(r"\bfind\b(?:\s+all)?\s+['\"]?(.+?)['\"]?\s+in\s+(.+?)\s+(?:and\s+save\s+(?:it\s+)?to|and\s+write\s+(?:it\s+)?to)\s+([\w\.\-_]+)", low, re.IGNORECASE)
    if m:
        query = m.group(1).strip("'\"")
        path = _resolve_common_path(m.group(2))
        found = run_tool("search_content", {"query": query, "path": path, "max_results": 100}, knowledge)
        if found.startswith("Error:"):
            return found
        written = run_tool("write_file", {"filename": m.group(3).strip(), "content": f"Results for '{query}' in {path}:\n\n{found}"}, knowledge)
        return f"{written}\n\nFound matches:\n{found[:500]}"

    # Multi-step: find ... and save to ... (no 'in' path, default home)
    m = re.search(r"\bfind\b(?:\s+all)?\s+['\"]?(.+?)['\"]?\s+(?:and\s+save\s+(?:it\s+)?to|and\s+write\s+(?:it\s+)?to)\s+([\w\.\-_]+)", low, re.IGNORECASE)
    if m:
        query = m.group(1).strip("'\"")
        found = run_tool("search_content", {"query": query, "path": "~", "max_results": 100}, knowledge)
        if found.startswith("Error:"):
            return found
        written = run_tool("write_file", {"filename": m.group(2).strip(), "content": f"Results for '{query}' in home:\n\n{found}"}, knowledge)
        return f"{written}\n\nFound matches:\n{found[:500]}"

    # Multi-step: index ... and search for ...
    m = re.search(r"\bindex\b(?:\s+my)?\s+(.+?)\s+and\s+(?:search|search\s+for)\s+['\"]?(.+?)['\"]?$", low, re.IGNORECASE)
    if m:
        path = _resolve_common_path(m.group(1))
        indexed = run_tool("index_documents", {"path": path}, knowledge)
        results = run_tool("search_notes", {"query": m.group(2).strip("'\"")}, knowledge)
        return f"{indexed}\n\n{results}"

    # Multi-step: list ... and save to ...
    m = re.search(r"\blist\b(?:\s+(?:the\s+)?files)?(?:\s+in)?\s+(.+?)\s+(?:and\s+save\s+(?:it\s+)?to|and\s+write\s+(?:it\s+)?to)\s+([\w\.\-_]+)", low, re.IGNORECASE)
    if m:
        path = _resolve_common_path(m.group(1))
        listed = run_tool("list_directory", {"path": path}, knowledge)
        if listed.startswith("Error:"):
            return listed
        written = run_tool("write_file", {"filename": m.group(2).strip(), "content": f"Files in {path}:\n\n{listed}"}, knowledge)
        return f"{written}\n\nFiles:\n{listed[:500]}"

    # list files
    m = re.search(r"\blist\b(?:\s+(?:the\s+)?files)?(?:\s+in)?\s+(.+?)(?!\s+(?:and|or)\b)$", low, re.IGNORECASE)
    if m:
        return run_tool("list_directory", {"path": _resolve_common_path(m.group(1))}, knowledge)

    # read file
    m = re.search(r"\bread\b(?:\s+file)?\s+(.+)$", low, re.IGNORECASE)
    if m:
        return run_tool("read_file", {"path": _resolve_common_path(m.group(1)), "limit": 5000}, knowledge)

    # run shell
    m = re.search(r"\b(?:run|execute)\b(?:\s+shell|\s+command)?\s+(.+)$", low, re.IGNORECASE)
    if m:
        return run_tool("run_shell", {"command": m.group(1).strip()}, knowledge)

    # search content
    m = re.search(r"\b(?:search|grep)\b(?:\s+for)?\s+['\"]?(.+?)['\"]?(?!\s+(?:and|or)\b)(?:\s+in\s+(.+))?$", low, re.IGNORECASE)
    if m:
        query = m.group(1).strip("'\"")
        path = _resolve_common_path(m.group(2)) if m.group(2) else "~"
        return run_tool("search_content", {"query": query, "path": path, "max_results": 20}, knowledge)

    # write note
    m = re.search(r"\bwrite\b(?:\s+a?\s+note|\s+file|\s+to)?\s+([\w\.\-_]+)\s*(?::|with|containing)\s+(.+)$", low, re.IGNORECASE)
    if m:
        return run_tool("write_file", {"filename": m.group(1).strip(), "content": m.group(2).strip()}, knowledge)

    return None


def run_tool(name: str, args: dict, knowledge: Optional[BadAppleKnowledge] = None) -> str:
    try:
        if name == "get_current_time":
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        if name == "list_directory":
            p = Path(args.get("path", "~")).expanduser()
            if not p.is_dir():
                return f"Error: {p} is not a directory"
            items = sorted(p.iterdir())[:50]
            return "\n".join(str(i.name) for i in items)
        if name == "read_file":
            p = Path(args.get("path", "~")).expanduser()
            if not p.is_file():
                return f"Error: {p} is not a file"
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
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
            if args.get("append"):
                with open(p, "a", encoding="utf-8") as f:
                    f.write(content + "\n")
                return f"Appended to {p.name}"
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Wrote {p}"
        if name == "search_content":
            query = args.get("query", "")
            p = Path(args.get("path", "~")).expanduser()
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
            )
            lines = [l for l in (result.stdout or "").splitlines() if l][:max_results]
            return "\n".join(lines) or "No matches found"
        if name == "run_shell":
            return _run_shell(args.get("command", ""))
        if name == "run_applescript":
            script = args.get("script", "")
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return (result.stdout or result.stderr or "done").strip()
        if name == "search_local_files":
            query = args.get("query", "")
            result = subprocess.run(
                ["mdfind", query],
                capture_output=True,
                text=True,
                timeout=15,
            )
            lines = [l for l in (result.stdout or "").splitlines() if l][:20]
            return "\n".join(lines) or "No files found"
        if name == "index_documents" and knowledge is not None:
            p = Path(args.get("path", "~")).expanduser()
            if p.exists():
                count = knowledge.index_paths([p])
                return f"Indexed {count} chunks from {p}"
            return f"Path not found: {p}"
        if name == "search_notes" and knowledge is not None:
            results = knowledge.search(args.get("query", ""), k=3)
            if not results:
                return "No relevant notes found."
            return "\n\n".join(f"(score: {s:.2f})\n{c}" for c, s in results)
    except Exception as e:
        return f"Tool error: {e}"
    return "Unknown tool"


def extract_tool_calls(text: str):
    pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    calls = []
    for m in pattern.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "name" in obj:
                calls.append(obj)
        except json.JSONDecodeError:
            continue
    cleaned = pattern.sub("", text).strip()
    return calls, cleaned


def polish_text(text: str) -> str:
    """Light cleanup for streaming chunks; does not add or force a sign-off."""
    # Strip Qwen3 thinking blocks if they leak into the stream.
    text = re.sub(r"\n?\s*<thinking>.*?\s*\n?", "", text, flags=re.DOTALL)
    text = re.sub(r"\n?\s*\.\.\.thinking\s*.*?(?:</s>|$)", "", text, flags=re.DOTALL)
    text = text.replace("— —", "—")
    text = text.replace("*", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?— ?", "—", text)
    text = re.sub(r"\.\.\.", "…", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    # Ensure a space after sentence punctuation when the next token runs together.
    text = re.sub(r"([.!?…])([A-Za-z])", r"\1 \2", text)
    return text.strip()


def _is_sentence_end(text: str) -> bool:
    """Heuristic to flush a streaming chunk when a sentence or utterance is done."""
    t = text.strip()
    if not t or len(t) <= 40:
        return False
    if t.lower().endswith("—besos") or t.lower().endswith("besos"):
        return True
    if t.endswith((".", "!", "?", "…")):
        return True
    if "\n\n" in t:
        return True
    # Force a flush on very long runs without punctuation so the client doesn't stall.
    if len(t) > 200:
        return True
    return False


# Spicy Latina English. Strip any foreign-language leakage and force one clean sign-off.
ALLOWED_SPANISH = {
    "papi", "mami", "mi", "amor", "corazón", "corazon", "cariño", "carino",
    "mija", "mijo", "besos", "cielo", "linda", "lindo", "princesa", "reina",
}
FORBIDDEN_WORDS = {
    "kisses",  # previous sign-off
    "hola", "adiós", "adios", "gracias", "por favor", "mira", "oye",
    "bueno", "muy", "mucho", "bien", "mal", "dios", "vaya",
    "nivel", "conciencia", "estoy", "estás", "siento", "tengo", "ayuda", "algo",
}

def _strip_existing_signoff(text: str) -> str:
    """Remove any trailing sign-off so we can add exactly one."""
    text = re.sub(r"[—-]\s*(besos|kisses)\s*\.?\s*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\b(besos|kisses)\b", "", text, flags=re.IGNORECASE).strip()
    return text


def _filter_english_sentences(text: str) -> str:
    """Drop sentences that langdetect flags as mostly non-English."""
    # Split on sentence terminators while keeping the punctuation.
    # Do not split on ellipses (...) because short fragments confuse langdetect.
    parts = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for part in parts:
        if not part.strip():
            continue
        # Keep short fragments and known sign-offs.
        if part.strip().lower().rstrip(".") in {"—besos", "besos"}:
            continue
        try:
            lang = detect(part)
        except LangDetectException:
            lang = "en"
        # Keep if English dominates, otherwise drop the whole sentence.
        if lang in {"en", "ca", "tl"}:  # ca/tl can be confused with short spicy English
            cleaned.append(part)
    return " ".join(cleaned).strip()


def _queue_get(q: queue.Queue, timeout: float = 0.1) -> Optional[Any]:
    try:
        return q.get(block=True, timeout=timeout)
    except queue.Empty:
        return None


def postprocess_output(text: str, sign_off: str = "—besos") -> str:
    # Strip Qwen3 thinking blocks; they often precede the real answer.
    text = re.sub(r"\n?\s*<think>.*?\s*\n?", "", text, flags=re.DOTALL)
    text = re.sub(r"\n?\s*\.\.\.thinking\s*.*?(?:</s>|$)", "", text, flags=re.DOTALL)
    text = text.replace("— —", "—")
    text = re.sub(r"\s+", " ", text).strip()

    text = _strip_existing_signoff(text)

    # Strip forbidden non-English words.
    for word in FORBIDDEN_WORDS:
        text = re.sub(r"\b" + re.escape(word) + r"\b", "", text, flags=re.IGNORECASE)

    # Clean up repeated punctuation and spaces.
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*,\s*([.!?])", r"\1", text)
    text = re.sub(r"\s*,\s*,", ",", text)
    text = re.sub(r"\s*,\s*—", "—", text)
    text = re.sub(r"^,\s*", "", text)
    text = re.sub(r"\s*,\s*$", "", text)
    text = re.sub(r"\s+([.!?])", r"\1", text)
    text = re.sub(r"([.!?])([—-])", r"\1 \2", text)

    if text.endswith("—"):
        return f"{text}{sign_off.lstrip('—')}"
    if text.endswith(".") or text.endswith("!") or text.endswith("?") or text.endswith("…"):
        return f"{text} {sign_off}"
    return f"{text} {sign_off}"


STOP_WORDS = {
    "i", "me", "mine", "you", "your", "yours", "it", "its", "am", "are",
    "was", "were", "be", "been", "being", "the", "a", "an", "this", "that", "these",
    "those", "and", "or", "but", "if", "then", "than", "as", "of", "in", "on", "at",
    "to", "for", "with", "from", "up", "down", "out", "off", "over", "under", "again",
    "which", "who", "when", "where", "why", "how", "do", "does", "did", "just",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
}


def relevant_memories(user_prompt: str, memories: List[str]) -> List[str]:
    words = set(w for w in re.findall(r"\b\w+\b", user_prompt.lower()) if w not in STOP_WORDS)
    scored = []
    for m in memories:
        m_words = set(w for w in re.findall(r"\b\w+\b", m.lower()) if w not in STOP_WORDS)
        score = len(words & m_words)
        if score >= 2:
            scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:3]]


class MLXServer:
    def __init__(self, secret: bytes, system_prompt: str):
        self.secret = secret
        self.system_prompt = system_prompt
        self.prompt_file = Path(
            os.environ.get("BADAPPLE_PROMPT_FILE") or DEFAULT_PROMPT_FILE
        ).expanduser()
        self.prompt_mtime: Optional[float] = self.prompt_file.stat().st_mtime if self.prompt_file.is_file() else None
        self.user_memory = load_user_memory()
        self.knowledge = BadAppleKnowledge()
        # Keep the last few turns in context. When it grows, older turns are
        # still persisted to disk and a rolling summary keeps context alive.
        self.max_history_turns = 3

        # Restore the last conversation, but always use the current system prompt.
        loaded = load_conversation()
        if loaded and loaded[0]["role"] == "system":
            loaded[0]["content"] = system_prompt
            self.messages = loaded
        elif loaded:
            self.messages = [{"role": "system", "content": system_prompt}] + loaded
        else:
            self.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        print(f"Loading Bad Apple MLX brain ({MAIN_MODEL})...", flush=True)
        self.model, self.tokenizer = load(MAIN_MODEL)
        print("Bad Apple MLX brain loaded.", flush=True)

        self.draft_model = None
        if DRAFT_MODEL:
            print(f"Loading speculative draft model {DRAFT_MODEL}...", flush=True)
            try:
                self.draft_model, _ = load(DRAFT_MODEL)
                print("Speculative draft model loaded.", flush=True)
            except Exception as e:
                print(f"Warning: could not load draft model: {e}", flush=True)

    def reset_conversation(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        try:
            conversation_path().unlink(missing_ok=True)
        except Exception:
            pass

    def check_prompt_reload(self):
        """Hot-reload the system prompt if prompt.txt changed on disk."""
        try:
            if not self.prompt_file.is_file():
                return
            mtime = self.prompt_file.stat().st_mtime
            if self.prompt_mtime is not None and mtime <= self.prompt_mtime:
                return
            new_prompt = self.prompt_file.read_text(encoding="utf-8").strip()
            if not new_prompt:
                return
            self.prompt_mtime = mtime
            self.system_prompt = new_prompt
            # Replace the system message at the head of the conversation.
            if self.messages and self.messages[0]["role"] == "system":
                self.messages[0]["content"] = new_prompt
            else:
                self.messages.insert(0, {"role": "system", "content": new_prompt})
            print("[daemon] system prompt hot-reloaded", flush=True)
        except Exception as e:
            print(f"[daemon] prompt reload failed: {e}", flush=True)

    def record_fact(self, text: str, source: str = "user"):
        low = text.lower()
        if source == "user":
            if any(phrase in low for phrase in ("my name is", "my name's", "i like", "i love", "i prefer", "remember that")):
                sentence = re.split(r"(?<=[.!?])\s+", text)[0]
                if 5 < len(sentence) < 200:
                    if sentence not in self.user_memory:
                        self.user_memory.append(sentence)
                    save_user_memory(self.user_memory)
        elif source == "assistant" and "your name is" in low:
            # Trust the assistant when it confirms a user fact
            sentence = re.split(r"(?<=[.!?])\s+", text)[0]
            if 5 < len(sentence) < 200 and sentence not in self.user_memory:
                self.user_memory.append(sentence)
                save_user_memory(self.user_memory)

    def prune_history(self):
        system = [m for m in self.messages if m["role"] == "system"]
        history = [m for m in self.messages if m["role"] != "system"]
        while len(history) > self.max_history_turns * 2:
            history = history[2:]
        self.messages = system + history

    def build_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        self.messages.append({"role": "user", "content": user_prompt})
        self.prune_history()
        return list(self.messages)

    def plan_and_execute(self, task: str, max_tokens: int, voice_mode: bool = False) -> Optional[str]:
        """Generate a step plan and execute it using local tools."""
        # 1. Ask the 8B for a dry, structured plan.
        plan_messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        plan_prompt = self.tokenizer.apply_chat_template(
            plan_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        plan_raw = self._stream(plan_prompt, max_tokens=200)
        plan_lines = [l.strip() for l in plan_raw.splitlines() if l.strip().startswith(("TOOL:", "SAY:"))]
        if not plan_lines:
            return None

        # 2. Execute tool steps, collecting the last tool result.
        last_tool_result = ""
        final_say = ""
        for line in plan_lines:
            if line.startswith("TOOL:"):
                parts = line.split(":", 2)
                if len(parts) < 3:
                    continue
                tool_name = parts[1].strip()
                try:
                    args = json.loads(parts[2].strip())
                except json.JSONDecodeError:
                    continue
                last_tool_result = run_tool(tool_name, args, self.knowledge)
            elif line.startswith("SAY:"):
                final_say = line.split(":", 1)[1].strip()

        # 3. If a SAY step exists, use it as a prompt to summarize the last tool result.
        if final_say:
            summary_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task},
                {"role": "tool", "content": f"Tool result:\n{last_tool_result}"},
                {"role": "user", "content": final_say},
            ]
            summary_prompt = self.tokenizer.apply_chat_template(
                summary_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            bridge = random.choice(BRIDGES)
            raw = self._stream(f"{summary_prompt.rstrip()}\n{bridge} ", max_tokens)
            return postprocess_output(f"{bridge} {raw.strip()}")

        # No SAY step: just return the last tool result with persona polish.
        bridge = random.choice(BRIDGES)
        return postprocess_output(f"{bridge} {last_tool_result}")

    def render_prompt(self, messages: List[Dict[str, str]], bridge: str, use_tools: bool = False, voice_mode: bool = False) -> str:
        # Build retrieved context from long-term memory and local documents.
        # Keep it tight: prompt encoding is the biggest latency hit on Apple Silicon.
        t0 = time.time()

        # Voice mode trades multi-turn context for speed: only a tight system
        # prompt and the last user turn are kept. The full conversation is saved.
        if voice_mode:
            patched = [messages[0], messages[-1]]
            patched[0]["content"] = VOICE_SYSTEM_PROMPT
        else:
            patched = list(messages)

        rel_mem = relevant_memories(messages[-1]["content"], self.user_memory)

        # If a user-fact is already remembered, answer from that instead of
        # getting distracted by unrelated documents.
        rel_know = []
        if not rel_mem and not voice_mode:
            rel_know = self.knowledge.search(messages[-1]["content"], k=1, threshold=0.55)

        # Put user memories right in the current user message so the assistant
        # can't ignore them.
        if rel_mem:
            memory_text = "Things you remember about the user:\n" + "\n".join(
                f"- {m[:200]}" for m in rel_mem[:2]
            )
            last = patched[-1]
            if last["role"] == "user":
                patched[-1] = {
                    "role": "user",
                    "content": f"{last['content']}\n\n{memory_text}",
                }

        # Put local documents right before the user question (long, retrieved).
        if rel_know:
            docs_text = "Relevant local documents:\n" + "\n".join(
                f"- {c[:500]}" for c, _ in rel_know[:1]
            )
            patched.insert(-1, {
                "role": "user",
                "content": f"Use this context to answer:\n\n{docs_text}",
            })

        rendered = self.tokenizer.apply_chat_template(
            patched,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            tools=TOOLS if use_tools else None,
        )
        # When tools are offered, do not force a spoken bridge prefix; the model
        # needs to be able to start with <tool_call> if it decides to use tools.
        # The bridge is added back after the final response is cleaned.
        if use_tools:
            print(f"[perf] render_prompt in {time.time() - t0:.2f}s", flush=True)
            return rendered.rstrip()
        print(f"[perf] render_prompt in {time.time() - t0:.2f}s", flush=True)
        return f"{rendered.rstrip()}\n{bridge} "

    def generate_with_tools(
        self,
        user_prompt: str,
        max_tokens: int,
        voice_mode: bool = False,
        stream_queue: Optional[queue.Queue] = None,
    ) -> str:
        self.check_prompt_reload()
        bridge = random.choice(BRIDGES)
        if user_prompt.strip().lower() == "new chat":
            self.reset_conversation()
            # Ask the model for a fresh English greeting instead of treating it as a command.
            user_prompt = "Greet me"
        self.record_fact(user_prompt, source="user")
        messages = self.build_messages(user_prompt)
        use_tools = should_use_tools(user_prompt)

        def clean(raw: str) -> str:
            _, text = extract_tool_calls(raw)
            return postprocess_output(f"{bridge} {text}")

        # First generation. Only stream when tools are not offered, because tool
        # reasoning can produce intermediate <tool_call> blocks we don't want
        # mixed into the streamed voice/text output.
        raw = self._stream(
            self.render_prompt(messages, bridge, use_tools=use_tools, voice_mode=voice_mode),
            max_tokens,
            bridge=bridge if not use_tools else None,
            stream_queue=stream_queue if not use_tools else None,
        )
        tool_calls, _ = extract_tool_calls(raw)

        if not tool_calls:
            return clean(raw)

        # Tool loop (multi-step task execution; allow more chained tool calls)
        for _ in range(5):
            for call in tool_calls:
                result = run_tool(call["name"], call.get("arguments", {}), self.knowledge)
                self.messages.append({
                    "role": "tool",
                    "content": json.dumps({"name": call["name"], "result": result}),
                    "name": call["name"],
                })
            # Re-render and generate after tool results
            raw = self._stream(self.render_prompt(self.messages, bridge, use_tools=True, voice_mode=voice_mode), max_tokens)
            tool_calls, _ = extract_tool_calls(raw)
            if not tool_calls:
                return clean(raw)

        return clean(raw)

    def _stream(
        self,
        prompt: str,
        max_tokens: int,
        bridge: Optional[str] = None,
        stream_queue: Optional[queue.Queue] = None,
    ) -> str:
        t0 = time.time()
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        print(f"[perf] prompt encoded in {time.time() - t0:.2f}s ({len(tokens)} tokens)", flush=True)
        sampler = make_sampler(temp=0.6, top_p=0.9, top_k=40, min_p=0.05)
        logits_processors = make_logits_processors(
            repetition_penalty=1.1,
            repetition_context_size=24,
        )
        accumulated = ""
        stream_buffer = (bridge + " ") if bridge and stream_queue is not None else ""
        final_metrics = None
        draft_tokens = 0
        total_tokens = 0
        gen_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": tokens,
            "max_tokens": max_tokens,
            "sampler": sampler,
            "logits_processors": logits_processors,
        }
        if self.draft_model is not None:
            gen_kwargs["draft_model"] = self.draft_model
            gen_kwargs["num_draft_tokens"] = NUM_DRAFT_TOKENS
        gen_t0 = time.time()
        first_token_logged = False
        for response in stream_generate(**gen_kwargs):
            if not first_token_logged:
                print(f"[perf] first token after {time.time() - gen_t0:.2f}s", flush=True)
                first_token_logged = True
            accumulated += response.text
            if stream_queue is not None:
                stream_buffer += response.text
                if _is_sentence_end(stream_buffer):
                    chunk = polish_text(stream_buffer)
                    if chunk:
                        # Add a trailing space after sentence-ending punctuation so the
                        # next streamed chunk doesn't run into this one.
                        if chunk.endswith((".", "!", "?", "…")):
                            chunk += " "
                        stream_queue.put(chunk)
                    stream_buffer = ""
            total_tokens += 1
            if response.from_draft:
                draft_tokens += 1
            if response.finish_reason is not None:
                final_metrics = response
        if stream_queue is not None and stream_buffer.strip():
            # If we hit the token limit and the final fragment is incomplete,
            # don't speak a cut-off word. We add the sign-off below instead.
            if final_metrics is not None and final_metrics.finish_reason == "length" and not _is_sentence_end(stream_buffer):
                pass
            else:
                chunk = polish_text(stream_buffer)
                if chunk:
                    if chunk.endswith((".", "!", "?", "…")):
                        chunk += " "
                    stream_queue.put(chunk)
        if stream_queue is not None and not re.search(r"—\s*besos\s*$", accumulated.strip(), re.IGNORECASE):
            # Sign off in the audio stream too so the voice doesn't just stop mid-sentence.
            stream_queue.put("—besos")
        if final_metrics is not None:
            pct = (100.0 * draft_tokens / total_tokens) if total_tokens > 0 else 0.0
            print(
                f"[perf] {final_metrics.generation_tokens} tokens @ "
                f"{final_metrics.generation_tps:.1f} t/s, "
                f"draft_accept_ratio={pct:.0f}%, "
                f"num_draft_tokens={NUM_DRAFT_TOKENS}, "
                f"peak_memory={final_metrics.peak_memory:.2f} GB",
                flush=True,
            )
        return accumulated

    def polish_response(self, text: str) -> str:
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
        text = text.replace("*", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?— ?", "—", text)
        text = re.sub(r"\.\.\.", "…", text)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        # Remove any besos sign-off in the body and re-add once at the end
        text = re.sub(r"\s*—?\s*besos\s*", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[ \t]+", " ", text).strip()
        text = re.sub(r"\s*,\s*$", "", text)  # no trailing comma
        # If the response was cut off by max_tokens, trim to the last complete
        # sentence so we don't end with a dangling word before the sign-off.
        if not re.search(r"[.!?…]$", text):
            m = re.search(r"(.*[.!?…])\s+\S+$", text)
            if m:
                text = m.group(1).strip()
        if not text.lower().rstrip(" .!?,;:").endswith("—besos"):
            text = text + " —besos"
        return text

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            line = await reader.readline()
            if not line:
                return
            hello = json.loads(line.decode())
            if not (hello.get("type") == "hello" and hello.get("version") == SLICKS_VERSION and timestamp_is_fresh(hello.get("timestamp_ms")) and nonce_is_valid(hello.get("client_nonce"))):
                await _write_frame(writer, {"type": "error", "message": "invalid or stale SLICKS hello"})
                return

            timestamp_ms = hello["timestamp_ms"]
            client_nonce = hello["client_nonce"]
            server_nonce = random_nonce()
            await _write_frame(writer, {
                "type": "challenge",
                "version": SLICKS_VERSION,
                "server_nonce": server_nonce,
                "proof": server_proof(self.secret, timestamp_ms, client_nonce, server_nonce),
            })

            line = await reader.readline()
            if not line:
                return
            execute = json.loads(line.decode())
            if not (execute.get("type") == "execute" and execute.get("version") == SLICKS_VERSION and execute.get("timestamp_ms") == timestamp_ms and execute.get("client_nonce") == client_nonce and execute.get("server_nonce") == server_nonce):
                await _write_frame(writer, {"type": "error", "message": "invalid SLICKS execute frame"})
                return

            prompt = execute["prompt"]
            max_new_tokens = execute["max_new_tokens"]
            proof = execute["proof"]

            try:
                validate_request(prompt, max_new_tokens)
            except ValueError as e:
                await _write_frame(writer, {"type": "error", "message": str(e)})
                return

            material = client_material(timestamp_ms, client_nonce, server_nonce, prompt, max_new_tokens)
            if not verify(self.secret, material, proof):
                await _write_frame(writer, {"type": "error", "message": "SLICKS client authentication failed"})
                return

            await _write_frame(writer, {"type": "accepted"})

            # Voice clients prepend this sentinel so the server can use a tight,
            # low-latency prompt and skip long document retrieval.
            voice_mode = prompt.startswith("__BADAPPLE_VOICE__ ")
            if voice_mode:
                prompt = prompt[len("__BADAPPLE_VOICE__ "):]

            if prompt in ("__BADAPPLE_SWITCH_DEEP__", "__BADAPPLE_SWITCH_FAST__"):
                await _write_frame(writer, {"type": "done", "text": ""})
                return

            if prompt.lower() in ("__badapple_new_chat__", "new chat", "clear conversation"):
                self.reset_conversation()
                await _write_frame(writer, {"type": "done", "text": "Mmh, mi amor... fresh start. —besos"})
                return

            # Fast deterministic path for direct tool commands (read, list, run, search, write).
            # This avoids a full 8B generation for simple local actions and stays air-gapped.
            fast = fast_execute(prompt, self.knowledge)
            if fast:
                fast = self.polish_response(postprocess_output(fast))
                self.record_fact(prompt, source="user")
                self.messages.append({"role": "user", "content": prompt})
                self.messages.append({"role": "assistant", "content": fast})
                self.prune_history()
                save_conversation(self.messages)
                await _write_frame(writer, {"type": "done", "text": fast})
                return

            loop = asyncio.get_event_loop()

            # Multi-step task planning for requests that combine actions.
            if is_multi_step(prompt):
                def _plan():
                    try:
                        result = self.plan_and_execute(prompt, max_new_tokens, voice_mode=voice_mode)
                        if result:
                            return self.polish_response(result)
                        return self.polish_response(self.generate_with_tools(prompt, max_new_tokens, voice_mode=voice_mode))
                    except Exception as e:
                        traceback.print_exc()
                        return f"Error: {e}"
                text = await loop.run_in_executor(None, _plan)
                if not text:
                    text = "Mmh, mi amor... I couldn't make a plan. —besos"
                self.record_fact(prompt, source="user")
                self.messages.append({"role": "user", "content": prompt})
                self.messages.append({"role": "assistant", "content": text})
                self.prune_history()
                save_conversation(self.messages)
                await _write_frame(writer, {"type": "done", "text": text})
                return

            stream_queue = queue.Queue()

            def _gen():
                try:
                    raw = self.generate_with_tools(prompt, max_new_tokens, voice_mode=voice_mode, stream_queue=stream_queue)
                    return self.polish_response(raw)
                except Exception as e:
                    traceback.print_exc()
                    return f"Error generating response: {e}"

            future = loop.run_in_executor(None, _gen)

            # Stream sentence chunks as the 8B model generates.
            while not future.done() or not stream_queue.empty():
                chunk = await loop.run_in_executor(None, _queue_get, stream_queue, 0.2)
                if chunk:
                    await _write_frame(writer, {"type": "token", "text": chunk})

            text = future.result()

            if not text:
                text = "Mmh, mi amor... I'm here. —besos"

            # Store final assistant response in conversation; only user statements
            # become long-term memory, not the assistant's own rephrasings.
            self.messages.append({"role": "assistant", "content": text})
            self.prune_history()
            save_conversation(self.messages)

            await _write_frame(writer, {"type": "done", "text": text})

        except Exception as e:
            traceback.print_exc()
            try:
                await _write_frame(writer, {"type": "error", "message": f"MLX server error: {e}"})
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


async def main():
    secret = load_slicks_secret()
    # Support legacy env override; otherwise load from the prompt file and keep
    # the model in memory while the persona can be hot-reloaded.
    legacy = os.environ.get("BADAPPLE_SYSTEM_PROMPT")
    system_prompt = legacy if legacy else load_prompt()

    socket_path = os.environ.get("BADAPPLE_SOCKET_PATH", DEFAULT_SOCKET_PATH)
    fast_socket_path = socket_path.replace(".sock", "_fast.sock")
    Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
    for p in (socket_path, fast_socket_path):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass

    server = MLXServer(secret, system_prompt)

    srv = await asyncio.start_unix_server(server.handle_client, path=socket_path)
    os.chmod(socket_path, 0o666)
    try:
        os.symlink(socket_path, fast_socket_path)
        os.chmod(fast_socket_path, 0o666)
    except FileExistsError:
        pass
    print(f"Bad Apple MLX server listening on {socket_path}", flush=True)

    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
