"""Bad Apple MLX server — conversation persistence and memory helpers.

Extracted from badapple_mlx_server.py.
"""
import json
import os
import re
from pathlib import Path

def memory_path() -> Path:
    path = Path(os.environ.get("BADAPPLE_MEMORY_PATH", "/var/lib/bad_apple/user_memory.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def conversation_path() -> Path:
    path = Path(os.environ.get("BADAPPLE_CONVERSATION_PATH", "/var/lib/bad_apple/conversation.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def load_user_memory() -> list[str]:
    try:
        with open(memory_path()) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data[-50:]
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
        print(f"[mlx_server] open failed: {e}", flush=True)
    return []

def save_user_memory(facts: list[str]):
    try:
        with open(memory_path(), "w") as f:
            json.dump(facts[-50:], f, indent=2)
    except (TypeError, ValueError, OSError) as e:
        print(f"[mlx_server] open failed: {e}", flush=True)

def load_conversation() -> list[dict[str, str]]:
    try:
        with open(conversation_path()) as f:
            data = json.load(f)
            if isinstance(data, list):
                return [m for m in data if isinstance(m, dict) and "role" in m and "content" in m]
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
        print(f"[mlx_server] open failed: {e}", flush=True)
    return []

def save_conversation(messages: list[dict[str, str]]):
    try:
        path = conversation_path()
        # Persist last 40 messages max to keep file small and token count sane.
        with open(path, "w") as f:
            json.dump(messages[-40:], f, indent=2)
        # Make it readable by the user and group so the menu bar can open it.
        os.chmod(path, 0o644)
    except Exception:  # noqa: BLE001,S110 - cleanup
        pass


STOP_WORDS = {
    "i", "me", "mine", "you", "your", "yours", "it", "its", "am", "are",
    "was", "were", "be", "been", "being", "the", "a", "an", "this", "that", "these",
    "those", "and", "or", "but", "if", "then", "than", "as", "of", "in", "on", "at",
    "to", "for", "with", "from", "up", "down", "out", "off", "over", "under", "again",
    "which", "who", "when", "where", "why", "how", "do", "does", "did", "just",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
}


def relevant_memories(user_prompt: str, memories: list[str]) -> list[str]:
    words = set(w for w in re.findall(r"\b\w+\b", user_prompt.lower()) if w not in STOP_WORDS)
    scored = []
    for m in memories:
        m_words = set(w for w in re.findall(r"\b\w+\b", m.lower()) if w not in STOP_WORDS)
        score = len(words & m_words)
        if score >= 2:
            scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:3]]
