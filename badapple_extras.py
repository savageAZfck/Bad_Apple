"""Bad Apple OS extras — persona packs, streaming firewall, audit ledger,
semantic cache, and human-in-the-loop approvals.

Ported from the user's own GitHub projects and adapted for the Bad Apple
single-brain MLX server:
- build-a-homie (persona packs, teach, session recaps)
- apollos-shield / sovereign_edge (streaming Aho-Corasick output firewall)
- chronos-viper / shadow-matrix / ify (hash-chained audit ledger, PII redaction)
- f5_counterpunch / tok3n-viper (bge-small semantic cache + intent classifier)
- ify / savageops (phased approval workflow for destructive tools)
"""

import datetime
import fcntl
import hashlib
import hmac
import json
import os
import re
import subprocess
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from badapple_memory import MemoryGraph  # noqa: F401 - re-export

try:
    import yaml
except ImportError: # pragma: no cover
    yaml = None


def _safe_json(data: Any, sort_keys: bool = True) -> str:
    """Canonical JSON string for hashing."""
    return json.dumps(data, sort_keys=sort_keys, ensure_ascii=True, default=str)


# =============================================================================
# 1. PERSONA PACK SYSTEM
# =============================================================================


class PersonaPack:
    """Load and switch between persona packs stored on disk.

    The default pack is always the on-disk `prompt.txt` (California beach
    girl).  Alternative packs live in a JSON file `~/.bad_apple/personas.json`
    or at `BADAPPLE_PERSONAS_FILE`.  Packs are selected at run-time by the
    `BADAPPLE_PERSONA` environment variable or by the user command
    `switch to <persona>`.
    """

    DEFAULT_PERSONAS = {
        "default": {
            "name": "Bad Apple",
            "description": "Sovereign, anti-cloud, pro-bare-metal local AI assistant.",
            "system_prompt_file": "prompt.txt",
            "voice_system_prompt": None,
            "roast_bank": [],
        }
    }

    def __init__(self, data_dir: Path, prompt_file: Path):
        self.data_dir = data_dir
        self.prompt_file = prompt_file
        self.personas_file = (
            Path(os.environ["BADAPPLE_PERSONAS_FILE"]).expanduser()
            if os.environ.get("BADAPPLE_PERSONAS_FILE")
            else (data_dir / "personas.json")
            if (data_dir / "personas.json").is_file()
            else Path(__file__).with_name("personas.json")
        )
        self.personas = dict(self.DEFAULT_PERSONAS)
        self._last_mtime: float | None = None
        self._load_personas()
        self.active = os.environ.get("BADAPPLE_PERSONA", "default").lower().strip()
        if self.active not in self.personas:
            self.active = "default"

    def _load_personas(self):
        if self.personas_file.is_file():
            try:
                mtime = self.personas_file.stat().st_mtime
                if self._last_mtime is not None and mtime <= self._last_mtime:
                    return
                data = json.loads(self.personas_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.personas = dict(self.DEFAULT_PERSONAS)
                    self.personas.update(data)
                    self._last_mtime = mtime
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
                print(f"[persona] could not load personas: {e}", flush=True)

    def reload_personas(self):
        self._load_personas()

    def _resolve_prompt(self, persona: dict[str, Any]) -> str:
        path = persona.get("system_prompt_file")
        if path:
            try:
                p = Path(path).expanduser()
                if not p.is_absolute() and self.data_dir:
                    p = self.data_dir / p
                if p.is_file():
                    return p.read_text(encoding="utf-8").strip()
            except (OSError, ValueError) as e:
                print(f"[extras] expanduser failed: {e}", flush=True)
        return persona.get("system_prompt", self._fallback_prompt())

    def _resolve_voice_prompt(self, persona: dict[str, Any]) -> str | None:
        if persona.get("voice_system_prompt"):
            return persona["voice_system_prompt"].strip()
        return None

    def _fallback_prompt(self) -> str:
        try:
            if self.prompt_file.is_file():
                return self.prompt_file.read_text(encoding="utf-8").strip()
        except (OSError, ValueError) as e:
            print(f"[extras] is_file failed: {e}", flush=True)
        return (
            "You are Bad Apple — an independent, sassy, flirty California beach girl, "
            "running hot on Apple bare metal. No cloud, no internet, no hand-holding. "
            "Be playful, direct, and useful. No sign-off."
        )

    def get_system_prompt(self, voice_mode: bool = False) -> str:
        self._load_personas()
        persona = self.personas.get(self.active, self.personas["default"])
        if voice_mode:
            vp = self._resolve_voice_prompt(persona)
            if vp:
                return vp
        return self._resolve_prompt(persona)

    def get_roast_bank(self) -> list[str]:
        persona = self.personas.get(self.active, self.personas["default"])
        return persona.get("roast_bank", []) or []

    def switch(self, name: str) -> bool:
        low = name.lower().strip()
        if low in self.personas:
            self.active = low
            return True
        # Try a fuzzy prefix match.
        for k in self.personas:
            if k.startswith(low) or low in k:
                self.active = k
                return True
        return False

    def list_personas(self) -> list[str]:
        return list(self.personas.keys())

    def handle_command(self, prompt: str) -> str | None:
        """Intercept persona-related user commands.

        Supported:
            switch to <persona>
            teach <line>
        Returns a non-None response if the command was consumed.
        """
        low = prompt.strip().lower()
        if low.startswith("switch to "):
            name = prompt[10:].strip()
            if self.switch(name):
                return f"Switched to {self.active} persona, babe."
            return f"Don't know that persona. I have: {', '.join(self.list_personas())}"
        if low.startswith("teach "):
            line = prompt[6:].strip()
            if not line:
                return "What do you want to teach me, hun?"
            custom = self._load_custom_lines()
            if line not in custom:
                custom.append(line)
                self._save_custom_lines(custom)
            return f"Learned: '{line}'"
        return None

    def _custom_lines_file(self) -> Path:
        return self.data_dir / "custom_banter.json"

    def _load_custom_lines(self) -> list[str]:
        try:
            if self._custom_lines_file().is_file():
                data = json.loads(self._custom_lines_file().read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data[-200:]
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
            print(f"[extras] is_file failed: {e}", flush=True)
        return []

    def _save_custom_lines(self, lines: list[str]):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self._custom_lines_file(), "w", encoding="utf-8") as f:
                json.dump(lines[-200:], f, indent=2)
        except (TypeError, ValueError, OSError) as e:
            print(f"[persona] could not save custom lines: {e}", flush=True)


# =============================================================================
# 2. STREAMING OUTPUT FIREWALL
# =============================================================================


class _TrieNode:
    __slots__ = ("children", "fail", "outputs")

    def __init__(self):
        self.children: dict[str, _TrieNode] = {}
        self.fail: _TrieNode | None = None
        self.outputs: set = set()


class AhoCorasickAutomaton:
    """Aho-Corasick automaton over structural tokens (alphanumeric + spaces).

    Adapted from sovereign_edge.py.  Patterns are plain strings; they are
    tokenized into words/space markers before insertion so the scanner can
    match phrases while ignoring punctuation and casing.
    """

    def __init__(self, patterns: list[str]):
        self.root = _TrieNode()
        for idx, pat in enumerate(patterns):
            node = self.root
            for tok in _tokenize_structural(pat)[0]:
                if tok not in node.children:
                    node.children[tok] = _TrieNode()
                node = node.children[tok]
            node.outputs.add(idx)
        self._build_fail_links()

    def _build_fail_links(self):
        queue = []
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)
        while queue:
            rnode = queue.pop(0)
            for key, unode in rnode.children.items():
                queue.append(unode)
                fnode = rnode.fail
                while fnode and key not in fnode.children:
                    fnode = fnode.fail
                unode.fail = fnode.children[key] if fnode and key in fnode.children else self.root
                unode.outputs |= unode.fail.outputs

    def search(self, tokens: list[str]) -> str | None:
        node = self.root
        for tok in tokens:
            while node and tok not in node.children:
                node = node.fail
            node = node.children[tok] if node and tok in node.children else self.root
            if node.outputs:
                return f"<pattern:{next(iter(node.outputs))}>"
        return None


def _tokenize_structural(text: str, partial_word: str = "", trailing_gap: bool = False) -> tuple[list[str], str, bool]:
    tokens = []
    buff = partial_word
    in_gap = trailing_gap
    for c in text:
        if c.isalnum():
            buff += c.lower()
            in_gap = False
        else:
            if not in_gap:
                if buff:
                    tokens.append(buff)
                    buff = ""
                tokens.append(" ")
                in_gap = True
    if buff:
        tokens.append(buff)
    return tokens, buff, in_gap


class StreamingFirewall:
    """Rolling-window output firewall.  Feeds streaming text into an Aho-Corasick
    automaton and flags forbidden patterns.  Patterns are loaded from
    `~/.bad_apple/blocklist.txt` and are merged with sensible defaults.
    """

    DEFAULT_PATTERNS = [
        "sk-",
        "ssh-rsa",
        "-----BEGIN",
        "-----END",
        "BEGIN PRIVATE KEY",
        "BEGIN OPENSSH PRIVATE KEY",
    ]

    def __init__(self, data_dir: Path, window: int = 256):
        self.data_dir = data_dir
        self.window = window
        self.patterns: list[str] = list(self.DEFAULT_PATTERNS)
        self._load_blocklist()
        self.automaton = AhoCorasickAutomaton(self.patterns)
        self.partial_word = ""
        self.trailing_gap = False
        self.buffer: deque = deque(maxlen=window)

    def _load_blocklist(self):
        path = Path(os.environ.get("BADAPPLE_BLOCKLIST") or self.data_dir / "blocklist.txt").expanduser()
        if path.is_file():
            try:
                raw = path.read_text(encoding="utf-8")
                extra = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.startswith("#")]
                self.patterns.extend(extra)
            except (OSError, ValueError) as e:
                print(f"[firewall] could not load blocklist: {e}", flush=True)

    def reset(self):
        self.partial_word = ""
        self.trailing_gap = False
        self.buffer.clear()

    def push_and_check(self, text: str) -> str | None:
        """Push text into the rolling window.  Returns the matched pattern if any."""
        tokens, self.partial_word, self.trailing_gap = _tokenize_structural(
            text, self.partial_word, self.trailing_gap
        )
        self.buffer.extend(tokens)
        win = list(self.buffer)[-self.window :]
        return self.automaton.search(win)

    def check_full(self, text: str) -> str | None:
        """One-shot scan of a complete string."""
        toks, _, _ = _tokenize_structural(text)
        return self.automaton.search(toks)


# =============================================================================
# 3. HASH-CHAINED AUDIT LEDGER
# =============================================================================


class AuditLedger:
    """Append-only, hash-chained, redacted audit log.

    Every query, tool call, response, and error is written to
    `~/.bad_apple/ledger.jsonl` with a SHA-256 chain.  Secrets and PII are
    redacted before writing.
    """

    def __init__(self, data_dir: Path, genesis: str = "bad-apple-genesis-v1"):
        self.data_dir = data_dir
        self.ledger_path = data_dir / "ledger.jsonl"
        self.lock_path = data_dir / "ledger.lock"
        self.genesis = genesis
        self._secret = os.environ.get("BADAPPLE_LEDGER_SECRET", "").encode()
        self._lock = threading.RLock()

    def _last_hash(self) -> str:
        if not self.ledger_path.is_file():
            return hashlib.sha256(self.genesis.encode()).hexdigest()
        try:
            with open(self.ledger_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                position = f.tell()
                buffer = b""
                while position > 0:
                    size = min(8192, position)
                    position -= size
                    f.seek(position)
                    buffer = f.read(size) + buffer
                    lines = [line for line in buffer.splitlines() if line.strip()]
                    if lines and (position == 0 or len(lines) > 1):
                        return json.loads(lines[-1])["hash"]
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError, LookupError) as e:
            print(f"[extras] open failed: {e}", flush=True)
        return hashlib.sha256(self.genesis.encode()).hexdigest()

    @staticmethod
    def _redact_value(value: Any) -> Any:
        if isinstance(value, str):
            # API keys, private key headers, tokens, emails, SSN-ish, phone-ish.
            if re.search(r"\b(sk-[a-zA-Z0-9_\-]{20,}|\bAIza[0-9A-Za-z_\-]{35,})\b", value):
                return "[REDACTED_KEY]"
            if re.search(r"-----BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----", value):
                return "[REDACTED_KEY]"
            if re.search(r"[0-9a-fA-F]{64,}", value):
                return "[REDACTED_HASH]"
            if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", value):
                return "[REDACTED_EMAIL]"
            if re.search(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b", value):
                return "[REDACTED_SSN]"
            if re.search(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", value):
                return "[REDACTED_PHONE]"
            # Long random-looking strings.
            if re.search(r"[A-Za-z0-9_\-]{40,}", value):
                return "[REDACTED_TOKEN]"
        return value

    def redact(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self.redact(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self.redact(x) for x in data]
        if isinstance(data, str):
            return self._redact_value(data)
        return data

    def record(self, event_type: str, data: Any):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with self._lock, open(self.lock_path, "a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                safe = self.redact(data)
                prev = self._last_hash()
                payload = {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": event_type,
                    "data": safe,
                    "prev_hash": prev,
                }
                raw = _safe_json(payload)
                if self._secret:
                    raw_hash = hmac.new(self._secret, raw.encode(), hashlib.sha256).hexdigest()
                else:
                    raw_hash = hashlib.sha256(raw.encode()).hexdigest()
                payload["hash"] = raw_hash
                line = (_safe_json(payload) + "\n").encode("utf-8")
                fd = os.open(self.ledger_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(fd, line)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except (OSError, ValueError, LookupError, TypeError) as e:
            print(f"[audit] ledger write failed: {e}", flush=True)

    def verify(self) -> list[dict[str, Any]]:
        """Return a list of verification results for every entry."""
        results = []
        prev = hashlib.sha256(self.genesis.encode()).hexdigest()
        if not self.ledger_path.is_file():
            return results
        try:
            with open(self.ledger_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    calc = _safe_json({
                        "ts": entry["ts"],
                        "type": entry["type"],
                        "data": entry["data"],
                        "prev_hash": entry["prev_hash"],
                    })
                    if self._secret:
                        expected = hmac.new(self._secret, calc.encode(), hashlib.sha256).hexdigest()
                    else:
                        expected = hashlib.sha256(calc.encode()).hexdigest()
                    results.append({
                        "ts": entry["ts"],
                        "type": entry["type"],
                        "valid": expected == entry.get("hash") and entry["prev_hash"] == prev,
                    })
                    prev = entry.get("hash", expected)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError, LookupError) as e:
            results.append({"ts": None, "type": "verify_error", "valid": False, "error": str(e)})
        return results


# =============================================================================
# 4. SEMANTIC CACHE + INTENT CLASSIFIER
# =============================================================================


class SemanticCache:
    """Query-to-response cache using bge-small-en-v1.5 embeddings.

    Before a 9B generation, the server checks whether a previous, semantically
    similar query has been answered.  If cosine similarity is above the
    threshold, the cached response is returned instantly.

    This also provides a lightweight intent classifier: user queries are
    compared against labeled example embeddings and the nearest label is
    returned.  The server uses this to decide whether to inject roast moods,
    retrieve documents, or route to tools.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DEFAULT_THRESHOLD = 0.92
    MAX_CACHE_SIZE = 500

    def __init__(self, data_dir: Path, threshold: float | None = None):
        self.data_dir = data_dir
        self.cache_path = data_dir / "semantic_cache.json"
        self.threshold = threshold or float(
            os.environ.get("BADAPPLE_CACHE_THRESHOLD") or self.DEFAULT_THRESHOLD
        )
        self.tokenizer = None
        self.model = None
        self._entries: list[dict[str, Any]] = []
        self._load()

    def _load_model(self):
        if self.model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer

            print("[cache] loading bge-small encoder...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModel.from_pretrained(self.MODEL_NAME)
            self.model.eval()
            print("[cache] encoder loaded.", flush=True)
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            print(f"[cache] encoder failed to load: {e}", flush=True)

    def _encode(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        if self.model is None:
            # Fallback: zero vectors. Cache will not match.
            return np.zeros((len(texts), 384))
        import torch

        all_emb = []
        for i in range(0, len(texts), 16):
            batch = texts[i : i + 16]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                out = self.model(**encoded)
            # Mean pooling.
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            emb = (out[0] * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_emb.append(emb.numpy())
        emb = np.vstack(all_emb)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norm + 1e-10)

    def _load(self):
        if not self.cache_path.is_file():
            return
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._entries = data
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
            print(f"[cache] could not load cache: {e}", flush=True)

    def _save(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._entries[-self.MAX_CACHE_SIZE :], f, indent=2)
        except (TypeError, ValueError, OSError) as e:
            print(f"[cache] could not save cache: {e}", flush=True)

    def lookup(self, query: str, persona: str = "default") -> str | None:
        if not self._entries:
            return None
        q_emb = self._encode([query])[0]
        best_score = -1.0
        best_idx = -1
        for i, e in enumerate(self._entries):
            if e.get("persona", "default") != persona:
                continue
            try:
                vec = np.array(e["embedding"], dtype=np.float32)
            except (LookupError, TypeError, ValueError):
                continue
            score = float(q_emb @ vec)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0 and best_score >= self.threshold:
            entry = self._entries[best_idx]
            entry["hits"] = entry.get("hits", 0) + 1
            entry["last_hit"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._save()
            return entry["response"]
        return None

    def store(self, query: str, response: str, persona: str = "default", intent: str | None = None):
        emb = self._encode([query])[0].tolist()
        self._entries.append({
            "query": query,
            "embedding": emb,
            "response": response,
            "persona": persona,
            "intent": intent,
            "hits": 0,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        if len(self._entries) > self.MAX_CACHE_SIZE * 2:
            # Keep the most-used half.
            self._entries = sorted(self._entries, key=lambda e: e.get("hits", 0))[-self.MAX_CACHE_SIZE :]
        self._save()

    def clear(self) -> str:
        """Clear all in-memory and on-disk semantic cache entries."""
        self._entries = []
        try:
            if self.cache_path.is_file():
                self.cache_path.unlink()
        except (OSError, ValueError) as e:
            print(f"[cache] could not clear cache file: {e}", flush=True)
        return "Semantic cache cleared."

    def classify_intent(self, query: str) -> str | None:
        """Compare query to a small fixed set of example phrases and return label."""
        examples = {
            "greeting": ["hi", "hello", "hey", "what's up", "good morning", "good evening"],
            "roast": ["who are you", "what do you think of siri", "roast the cloud", "bad apple vs siri"],
            "math": ["what is 2+2", "calculate", "solve", "what is 7+7", "math"],
            "time": ["what time is it", "current time", "what day is it"],
            "search": ["find file", "search my files", "where is"],
            "tool": ["list files", "run shell", "open app", "write note", "read file"],
            "memory": ["remember that", "what did I say", "do I have"],
            "chat": ["tell me about", "explain", "how do I", "what is"],
        }
        q_emb = self._encode([query])[0]
        best_score = -1.0
        best_label = None
        for label, phrases in examples.items():
            emb = self._encode(phrases)
            scores = q_emb @ emb.T
            avg = float(scores.mean())
            if avg > best_score:
                best_score = avg
                best_label = label
        return best_label


# =============================================================================
# 4a. WORKSPACE / PROJECT CONTEXT
# =============================================================================


class Workspace:
    """Tracks the active project/workspace directory and provides context
    (recent files, git state, notes) for the assistant to reason over.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.workspace_file = data_dir / "workspace.json"
        self._state: dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.workspace_file.is_file():
            try:
                self._state = json.loads(self.workspace_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
                print(f"[workspace] could not load: {e}", flush=True)

    def _save(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.workspace_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
        except (TypeError, ValueError, OSError) as e:
            print(f"[workspace] could not save: {e}", flush=True)

    @property
    def path(self) -> Path | None:
        p = self._state.get("workspace_dir")
        if not p:
            return None
        expanded = Path(p).expanduser()
        return expanded if expanded.is_dir() else None

    def set(self, workspace_dir: str) -> str:
        p = Path(workspace_dir).expanduser()
        if not p.is_dir():
            return f"Error: {p} is not a directory"
        self._state["workspace_dir"] = str(p.resolve())
        self._state["set_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._save()
        return f"Workspace set to {p}"

    def clear(self) -> str:
        self._state = {}
        self._save()
        return "Workspace cleared."

    def summary(self) -> str:
        p = self.path
        if not p:
            return "No active workspace."

        # Build system detection.
        build_system = []
        build_files = {
            "Cargo.toml": "Rust/Cargo",
            "package.json": "Node/npm",
            "pyproject.toml": "Python",
            "setup.py": "Python setuptools",
            "Package.swift": "Swift Package",
            "Makefile": "Make",
            "CMakeLists.txt": "CMake",
            "build.gradle": "Gradle",
            "pom.xml": "Maven",
            "*.xcodeproj": "Xcode project",
            "*.xcworkspace": "Xcode workspace",
        }
        for pattern, name in build_files.items():
            if pattern.startswith("*"):
                if any(p.glob(pattern)):
                    build_system.append(name)
            elif (p / pattern).is_file():
                build_system.append(name)

        # Recent files (top 10 by mtime).
        recent = []
        try:
            all_files = [f for f in p.rglob("*") if f.is_file() and not f.name.startswith(".")]
            all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            recent = [str(f.relative_to(p)) for f in all_files[:10]]
        except (OSError, ValueError) as e:
            print(f"[extras] rglob failed: {e}", flush=True)

        git_info = ""
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(p),
                capture_output=True,
                text=True,
                timeout=5,
            check=False)
            dirty = len(status.stdout.strip().splitlines()) if status.stdout.strip() else 0
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(p),
                capture_output=True,
                text=True,
                timeout=5,
            check=False).stdout.strip() or "unknown"
            last_commit = subprocess.run(
                ["git", "log", "-1", "--oneline"],
                cwd=str(p),
                capture_output=True,
                text=True,
                timeout=5,
            check=False).stdout.strip() or "no commits"
            git_info = f"git branch: {branch}, dirty: {dirty}, last: {last_commit}. "
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            print(f"[extras] run failed: {e}", flush=True)

        readme_summary = ""
        for readme_name in ["README.md", "readme.md", "README.rst"]:
            readme = p / readme_name
            if readme.is_file():
                try:
                    text = readme.read_text(encoding="utf-8", errors="ignore").strip()
                    readme_summary = f"README: {text[:160].replace(chr(10), ' ')}... "
                except (OSError, ValueError) as e:
                    print(f"[extras] strip failed: {e}", flush=True)
                break

        build = f"Build system: {', '.join(build_system)}. " if build_system else ""
        files = f"Recent files: {', '.join(recent)}." if recent else "No files found."
        return f"Workspace: {p}. {build}{git_info}{readme_summary}{files}"

    def resolve_path(self, maybe_path: str | None) -> Path:
        if maybe_path:
            return Path(maybe_path).expanduser()
        p = self.path
        return p if p else Path.home()


# =============================================================================
# 4a2. LONG-TERM MEMORY GRAPH
# =============================================================================


class Policy:
    """Declarative policy engine for the Bad Apple tool cage.

    Loads a YAML (or JSON) policy file that controls which tools are allowed,
    which require explicit approval, and which arguments/paths are permitted.
    """

    DEFAULT_POLICY = {
        "policy_version": "1.0",
        "autopilot": False,
        "defaults": {
            "allowed": True,
            "require_approval": True,
            "allowed_paths": [],
            "denied_patterns": [],
        },
        "tools": {
            "get_current_time": {"allowed": True, "require_approval": False},
            "list_directory": {"allowed": True, "require_approval": False},
            "read_file": {"allowed": True, "require_approval": False},
            "search_content": {"allowed": True, "require_approval": False},
            "search_local_files": {"allowed": True, "require_approval": False},
            "write_file": {"allowed": True, "require_approval": True},
            "run_shell": {"allowed": True, "require_approval": True},
            "run_applescript": {"allowed": True, "require_approval": True},
            "index_documents": {"allowed": True, "require_approval": True},
            "search_notes": {"allowed": True, "require_approval": False},
        },
    }

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.policy_path = (
            Path(os.environ["BADAPPLE_POLICY_FILE"]).expanduser()
            if os.environ.get("BADAPPLE_POLICY_FILE")
            else (data_dir / "policy.yaml")
            if (data_dir / "policy.yaml").is_file()
            else Path(__file__).with_name("policy.yaml")
        )
        self._policy: dict[str, Any] = dict(self.DEFAULT_POLICY)
        self._load()

    def _load(self):
        if not self.policy_path.is_file():
            return
        try:
            raw = self.policy_path.read_text(encoding="utf-8")
            if yaml is not None:
                data = yaml.safe_load(raw)
            else:
                data = json.loads(raw)
            if isinstance(data, dict):
                self._policy = data
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
            print(f"[policy] could not load {self.policy_path}: {e}", flush=True)

    @property
    def autopilot(self) -> bool:
        return bool(self._policy.get("autopilot", self.DEFAULT_POLICY["autopilot"]))

    def set_autopilot(self, enabled: bool) -> None:
        self._policy["autopilot"] = bool(enabled)

    def _tool_cfg(self, tool_name: str) -> dict[str, Any]:
        tools = self._policy.get("tools", self.DEFAULT_POLICY["tools"])
        defaults = self._policy.get("defaults", self.DEFAULT_POLICY["defaults"])
        cfg = dict(defaults)
        if isinstance(tools, dict) and tool_name in tools:
            if isinstance(tools[tool_name], dict):
                cfg.update(tools[tool_name])
        return cfg

    def is_allowed(self, tool_name: str) -> bool:
        return bool(self._tool_cfg(tool_name).get("allowed", True))

    def needs_approval(self, tool_name: str) -> bool:
        if self.autopilot:
            return False
        return bool(self._tool_cfg(tool_name).get("require_approval", True))

    def timeout(self, tool_name: str, default: int = 30) -> int:
        return int(self._tool_cfg(tool_name).get("max_timeout", default))

    def _denied(self, value: str, patterns: list[str]) -> str | None:
        if not patterns:
            return None
        lower = value.lower()
        for pat in patterns:
            if pat and pat.lower() in lower:
                return pat
        return None

    def _path_in_allowed(self, path: Path, allowed: list[str]) -> bool:
        if not allowed:
            return True
        resolved = path.expanduser().resolve()
        for root in allowed:
            try:
                root_path = Path(root).expanduser().resolve()
                if str(resolved).startswith(str(root_path)):
                    return True
            except (OSError, ValueError):
                continue
        return False

    def validate(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Return an error string if the tool call violates policy, else None."""
        if not self.is_allowed(tool_name):
            return f"tool '{tool_name}' is not allowed"
        cfg = self._tool_cfg(tool_name)

        # Path-based checks
        if "path" in args:
            path = Path(args["path"]).expanduser()
            allowed_paths = cfg.get("allowed_paths", [])
            if allowed_paths and not self._path_in_allowed(path, allowed_paths):
                return f"path {path} is outside allowed roots"

        # Write directory containment for write_file
        if tool_name == "write_file":
            notes_dir = Path(
                cfg.get("notes_dir")
                or os.environ.get("BADAPPLE_NOTES_DIR", "~/.bad_apple/notes")
            ).expanduser()
            filename = os.path.basename(args.get("filename", "note.txt"))
            target = (notes_dir / filename).resolve()
            try:
                notes_dir_resolved = notes_dir.resolve()
                if not str(target).startswith(str(notes_dir_resolved)):
                    return "write_file must stay in the notes directory"
            except (OSError, ValueError):
                return "invalid write_file path"
            if self._denied(filename, cfg.get("denied_patterns", [])):
                return "write filename contains a forbidden pattern"

        # Shell command checks
        if tool_name == "run_shell":
            command = args.get("command", "")
            allowed = cfg.get("allowed_commands", [])
            if allowed:
                first = command.strip().split()[0].lower() if command.strip() else ""
                if first not in [c.lower() for c in allowed]:
                    return f"command '{first}' is not in the allowed list"
            denied = self._denied(command, cfg.get("denied_patterns", []))
            if denied:
                return f"command matches forbidden pattern '{denied}'"

        # AppleScript checks
        if tool_name == "run_applescript":
            script = args.get("script", "")
            denied = self._denied(script, cfg.get("denied_patterns", []))
            if denied:
                return f"AppleScript matches forbidden pattern '{denied}'"

        # Per-app permissions (accessibility, run_shell, run_applescript, etc.)
        target = args.get("target") or args.get("app") or args.get("application")
        if target:
            allowed_apps = cfg.get("allowed_apps") or self._policy.get("allowed_apps")
            if allowed_apps and str(target).lower() not in [a.lower() for a in allowed_apps]:
                return f"app '{target}' is not in the allowed apps list"

        # Per-file permissions for any tool that touches a file
        if "path" in args:
            target_path = Path(args["path"]).expanduser()
            allowed_files = cfg.get("allowed_files")
            if allowed_files and not self._path_in_allowed(target_path, allowed_files):
                return f"path {target_path} is outside allowed_files list"

        return None


# =============================================================================
# 5. HUMAN-IN-THE-LOOP APPROVALS
# =============================================================================


class ApprovalGate:
    """Phased, human-in-the-loop approval for destructive tools.

    The policy engine now decides which tools require approval; this gate
    stores the pending proposals and handles the user approve/reject flow.

    Modeled on ify / savageops proposal/approve flow.
    """

    def __init__(self, data_dir: Path, policy: Policy | None = None):
        self.data_dir = data_dir
        self.pending: dict[str, dict[str, Any]] = {}
        self.policy = policy or Policy(data_dir)
        self._load_pending()

    def _load_pending(self):
        path = self.data_dir / "pending_approvals.json"
        if path.is_file():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.pending = data
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
                print(f"[approval] could not load pending: {e}", flush=True)

    def _save_pending(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.data_dir / "pending_approvals.json", "w", encoding="utf-8") as f:
                json.dump(self.pending, f, indent=2)
        except (TypeError, ValueError, OSError) as e:
            print(f"[approval] could not save pending: {e}", flush=True)

    @property
    def autopilot(self) -> bool:
        return bool(self.policy.autopilot)

    def needs_approval(self, tool_name: str) -> bool:
        return self.policy.needs_approval(tool_name)

    def propose(self, tool_name: str, arguments: dict[str, Any], user_prompt: str = "") -> str:
        proposal_id = str(uuid.uuid4())[:8]
        self.pending[proposal_id] = {
            "tool": tool_name,
            "arguments": arguments,
            "user_prompt": user_prompt,
            "proposed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "approved": False,
        }
        self._save_pending()
        return proposal_id

    def approve(self, proposal_id: str) -> tuple[str, dict[str, Any]] | None:
        entry = self.pending.get(proposal_id)
        if not entry:
            return None
        del self.pending[proposal_id]
        self._save_pending()
        return entry["tool"], entry["arguments"]

    def handle_approve_command(self, prompt: str) -> tuple[str, dict[str, Any]] | None:
        """Parse `approve <id>` and return (tool, args) if found."""
        low = prompt.strip().lower()
        m = re.match(r"^approve\s+([a-z0-9\-]+)$", low)
        if not m:
            return None
        return self.approve(m.group(1))

    def get_pending_summary(self) -> str:
        if not self.pending:
            return "No pending approvals."
        lines = ["Pending approvals:"]
        for pid, info in self.pending.items():
            lines.append(f"  {pid}: {info['tool']} — {info.get('user_prompt','')[:60]}")
        return "\n".join(lines)
