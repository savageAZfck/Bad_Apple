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
import random
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

from badapple_knowledge import BadAppleKnowledge
from langdetect import detect, LangDetectException
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

# Protocol constants from bad_apple_ipc.rs
SLICKS_VERSION = 1
DEFAULT_SOCKET_PATH = "/var/run/badapple/substrate.sock"
DEFAULT_KEY_PATH = "/var/lib/bad_apple/slicks.key"
HANDSHAKE_MAX_SKEW_MS = 60_000
MAX_PROMPT_BYTES = 64 * 1024
MAX_NEW_TOKENS = 512
MAX_FRAME_BYTES = 1024 * 1024

# Optional speculative decoding draft model. Must use the same tokenizer as the
# main 8B model. A Qwen3-1.7B-4bit adds ~800 MB and can 1.5-2x token throughput.
DRAFT_MODEL = os.environ.get("BADAPPLE_DRAFT_MODEL", "mlx-community/Qwen3-1.7B-4bit").strip()
NUM_DRAFT_TOKENS = int(os.environ.get("BADAPPLE_NUM_DRAFT_TOKENS") or "3")

DEFAULT_PROMPT_FILE = Path(__file__).with_name("prompt.txt")


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
]

TOOL_KEYWORDS = [
    "what time", "current time", "time is it", "date and time", "today's date",
    "list files", "show files", "files in", "directory", "folder", "what's in",
    "search for", "find file", "mdfind", "spotlight",
    "run applescript", "run script", "applescript",
    "index", "index documents", "index my", "index files",
    "search my notes", "search notes", "what do I have", "what did I write", "find in my",
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


def should_use_tools(prompt: str) -> bool:
    low = prompt.lower()
    return any(k in low for k in TOOL_KEYWORDS)


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


# Spicy Latina English. Strip any foreign-language leakage and force one clean sign-off.
ALLOWED_SPANISH = {
    "papi", "mami", "mi", "amor", "corazón", "corazon", "cariño", "carino",
    "mija", "mijo", "besos", "cielo", "linda", "lindo", "princesa", "reina",
}
FORBIDDEN_WORDS = {
    "kisses",  # previous sign-off
    "hola", "adiós", "adios", "gracias", "por favor", "sí", "si", "no", "mira", "oye",
    "bueno", "muy", "mucho", "bien", "mal", "dios", "vaya", "ay", "muy bien",
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
    parts = re.split(r"(?<=[.!?…])\s+", text)
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


def postprocess_output(text: str, sign_off: str = "—besos") -> str:
    text = text.replace("— —", "—")
    text = re.sub(r"\s+", " ", text).strip()

    text = _strip_existing_signoff(text)
    text = _filter_english_sentences(text)

    # Strip forbidden non-English words.
    for word in FORBIDDEN_WORDS:
        text = re.sub(r"\b" + re.escape(word) + r"\b", "", text, flags=re.IGNORECASE)

    # Clean up repeated punctuation and spaces.
    text = re.sub(r"\s+", " ", text).strip()
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
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self.user_memory = load_user_memory()
        self.knowledge = BadAppleKnowledge()
        # Keep only the last turn plus system; 8B 4-bit tends to latch onto
        # its own previous turns. Long-term memory and RAG handle the rest.
        self.max_history_turns = 1

        print("Loading Bad Apple MLX brain...", flush=True)
        self.model, self.tokenizer = load("mlx-community/Qwen3-8B-4bit")
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

    def render_prompt(self, messages: List[Dict[str, str]], bridge: str, use_tools: bool = False) -> str:
        # Build retrieved context from long-term memory and local documents
        rel_mem = relevant_memories(messages[-1]["content"], self.user_memory)

        # If a user-fact is already remembered, answer from that instead of
        # getting distracted by unrelated documents.
        rel_know = []
        if not rel_mem:
            rel_know = self.knowledge.search(messages[-1]["content"], k=3, threshold=0.45)

        patched = list(messages)

        # Put user memories right in the current user message so the assistant
        # can't ignore them.
        if rel_mem:
            memory_text = "Things you remember about the user:\n" + "\n".join(f"- {m}" for m in rel_mem)
            last = patched[-1]
            if last["role"] == "user":
                patched[-1] = {
                    "role": "user",
                    "content": f"{last['content']}\n\n{memory_text}",
                }

        # Put local documents right before the user question (long, retrieved)
        if rel_know:
            docs_text = "Relevant local documents:\n" + "\n".join(f"- {c}" for c, _ in rel_know)
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
        return f"{rendered.rstrip()}\n{bridge} "

    def generate_with_tools(self, user_prompt: str, max_tokens: int) -> str:
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

        # First generation
        raw = self._stream(self.render_prompt(messages, bridge, use_tools=use_tools), max_tokens)
        tool_calls, _ = extract_tool_calls(raw)

        if not tool_calls:
            return clean(raw)

        # Tool loop
        for _ in range(3):
            for call in tool_calls:
                result = run_tool(call["name"], call.get("arguments", {}), self.knowledge)
                self.messages.append({
                    "role": "tool",
                    "content": json.dumps({"name": call["name"], "result": result}),
                    "name": call["name"],
                })
            # Re-render and generate after tool results
            raw = self._stream(self.render_prompt(self.messages, bridge, use_tools=True), max_tokens)
            tool_calls, _ = extract_tool_calls(raw)
            if not tool_calls:
                return clean(raw)

        return clean(raw)

    def _stream(self, prompt: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        sampler = make_sampler(temp=0.4, top_p=0.85, top_k=20, min_p=0.05)
        accumulated = ""
        final_metrics = None
        draft_tokens = 0
        total_tokens = 0
        gen_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": tokens,
            "max_tokens": max_tokens,
            "sampler": sampler,
        }
        if self.draft_model is not None:
            gen_kwargs["draft_model"] = self.draft_model
            gen_kwargs["num_draft_tokens"] = NUM_DRAFT_TOKENS
        for response in stream_generate(**gen_kwargs):
            accumulated += response.text
            total_tokens += 1
            if response.from_draft:
                draft_tokens += 1
            if response.finish_reason is not None:
                final_metrics = response
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

            if prompt in ("__BADAPPLE_SWITCH_DEEP__", "__BADAPPLE_SWITCH_FAST__"):
                await _write_frame(writer, {"type": "done", "text": ""})
                return

            if prompt.lower() in ("__badapple_new_chat__", "new chat", "clear conversation"):
                self.reset_conversation()
                await _write_frame(writer, {"type": "done", "text": "Mmh, mi amor... fresh start. —besos"})
                return

            def _gen():
                try:
                    raw = self.generate_with_tools(prompt, max_new_tokens)
                    return self.polish_response(raw)
                except Exception as e:
                    traceback.print_exc()
                    return f"Error generating response: {e}"

            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, _gen)

            if not text:
                text = "Mmh, mi amor... I'm here. —besos"

            # Store final assistant response in conversation; only user statements
            # become long-term memory, not the assistant's own rephrasings.
            self.messages.append({"role": "assistant", "content": text})

            await _write_frame(writer, {"type": "done", "text": text})

        except Exception as e:
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
