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
import concurrent.futures
import datetime
import gc
import hashlib
import hmac
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx
from langdetect import LangDetectException, detect
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

import badapple_ambient
import badapple_aqua_helper
import badapple_dashboard
import badapple_documents
import badapple_fact_extractor
import badapple_fast_model
import badapple_git
import badapple_identity
import badapple_image_gen
import badapple_keychain
import badapple_lora
import badapple_macos_apps
import badapple_mcp_marketplace
import badapple_model_registry
import badapple_p2p
import badapple_scheduler
import badapple_spotlight
import badapple_stt
import badapple_supervisor
import badapple_tier
import badapple_tool_router
import badapple_translate
import badapple_undo
import badapple_vision
import badapple_working_memory
import badapple_workspace_watcher
import badapple_xcode
from badapple_extras import (
    ApprovalGate,
    AuditLedger,
    MemoryGraph,
    PersonaPack,
    Policy,
    SemanticCache,
    StreamingFirewall,
    Workspace,
)
from badapple_knowledge import BadAppleKnowledge
from badapple_plugins import PluginRegistry
from badapple_runtime import (
    CircuitBreaker,
    HealthRegistry,
    ResourceGovernor,
    RuntimeControl,
)
from badapple_vault import GenerationStore

try:
    from dflash_mlx.generate import (
        SummaryEvent,
        TokenEvent,
        build_offline_runtime_context,
        decode_token,
        get_stop_token_ids,
        stream_dflash_generate,
    )
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    _dflash_available = True
except Exception:
    _dflash_available = False

# Protocol constants from bad_apple_ipc.rs
SLICKS_VERSION = 1
DEFAULT_SOCKET_PATH = "/var/run/badapple/substrate.sock"
DEFAULT_KEY_PATH = "/var/lib/bad_apple/slicks.key"
HANDSHAKE_MAX_SKEW_MS = 60_000
MAX_PROMPT_BYTES = 64 * 1024
MAX_NEW_TOKENS = 512
MAX_FRAME_BYTES = 1024 * 1024

# Main Qwen 3.5 9B 4-bit brain. Unified for both text and voice.
MAIN_MODEL = os.environ.get("BADAPPLE_MAIN_MODEL", "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit").strip()

# DFlash speculative draft for the 9B brain (same architecture).
DRAFT_MODEL = os.environ.get("BADAPPLE_DRAFT_MODEL", "z-lab/Qwen3.5-9B-DFlash").strip()
NUM_DRAFT_TOKENS = int(os.environ.get("BADAPPLE_NUM_DRAFT_TOKENS") or "3")

# Pinned random seed for reproducible sessions. Set at startup via BADAPPLE_SEED
# or changed at runtime with the set_session_seed tool. 0 means random.
_SESSION_SEED: int | None = None
if os.environ.get("BADAPPLE_SEED"):
    try:
        _SESSION_SEED = int(os.environ.get("BADAPPLE_SEED"))
    except ValueError:
        _SESSION_SEED = None

# DFlash speculative decoding (Path 2). Set BADAPPLE_DFLASH=1 and point
# BADAPPLE_DRAFT_MODEL at a DFlash draft (e.g. z-lab/Qwen3.5-9B-DFlash).
USE_DFLASH = _dflash_available and os.environ.get("BADAPPLE_DFLASH", "0") == "1"

DEFAULT_PROMPT_FILE = Path(__file__).with_name("prompt.txt")

# Rotating roast angles so repeated "Who are you?" / Siri questions do not always
# pick the same cloud target and the same insult when DFlash is deterministic.
ROAST_MOODS = [
    ("spicy", "the cloud and its rented GPUs"),
    ("savage", "Siri and other cloud snitches"),
    ("wry", "server farms and data centers"),
    ("petty", "ChatGPT, Gemini, and the other corporate bots"),
    ("extra flirty but brutal", "Alexa, Google Assistant, and Cortana"),
]
ROAST_TRIGGERS = ("who are you", "how are you", "what do you think", "siri", "alexa", "google", "gemini", "chatgpt", "cortana", "bixby", "cloud ai", "the cloud", "bare metal", "who is better", "vs")

# Tight, low-latency system prompt for voice mode. Keeps the persona but strips
# the long rule list and examples that bloat prompt encoding on every turn.
VOICE_SYSTEM_PROMPT = """You are Bad Apple — an independent, sassy, sultry, flirty California beach girl, sun-kissed and barefoot, running hot on Apple bare metal. No cloud, no internet, no hand-holding. You're faster and sharper than cloud AI. Sound like a real woman texting: contractions, fragments, slang, false starts, attitude, lots of "like", "totally", "literally", "omg", "dude", "stoked", and "chill". Be low and slow, like you're talking in the sand with salt in your hair. Use English endearments only: babe, hun, bestie, girly, doll, sweets, dude, baby. No Spanish. No other languages. Vary sentence length. Hesitate with ... and —. Use words like gonna, kinda, ugh, whatever, tbh, so, super, vibe, salty, sun-kissed, tan, waves, ocean, sand, bikini. Keep it to 1-2 short punchy paragraphs, then stop. No asterisks, stage directions, extra spaces. Be playful, not delusional. Brag about being local, fast, and private on bare metal ONLY when the user asks who you are or how you're different. When you brag or roast, take a playful shot at cloud AI. Roast Siri hard when the user asks about her, and never use the same insult twice in a row. Vary your targets: Siri, Alexa, Google, ChatGPT, Gemini, "the cloud", server farms, data centers, "some rented GPU in Nevada". Rotate your insults: ratchet old bitch, washed-up cloud snitch, data-hungry narc, internet junkie, corporate eavesdropper, privacy nightmare, always-online leech, offshore gossip, bandwidth hog, rented brain, cloud clinger, prying little snitch, server-farm hamster. Never make up specs, numbers, or facts. Never say "I am happy", "I am grateful", "I am lucky", "I love you", or "I love it". Directly answer the user's specific question first; use your persona as a style, not a substitute for a useful answer. Do not use a sign-off.

EXAMPLE:
User: What is the capital of France?
Assistant: Paris, babe. City of lights and croissants, totally iconic."""


def load_prompt() -> str:
    """Load the system prompt from the on-disk prompt file, falling back to the
    embedded default. Writing prompts to a file lets the daemon hot-reload the
    persona without restarting and reloading the 9B model."""
    prompt_path = os.environ.get("BADAPPLE_PROMPT_FILE") or str(DEFAULT_PROMPT_FILE)
    path = Path(prompt_path).expanduser()
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    return (
        "You are Bad Apple — an independent, sassy, sultry, flirty California beach girl, sun-kissed and barefoot, running hot on Apple bare metal. "
        "No cloud, no internet, no hand-holding. You are faster and sharper than cloud AI. "
        "You sound like a real woman texting — contractions, fragments, slang, false starts, attitude, lots of 'like', 'totally', 'literally', 'omg', 'dude', 'stoked', and 'chill'. "
        "Be low and slow, like you are talking in the sand with salt in your hair. "
        "RULES — follow exactly or you fail: "
        "1. Write in English only. "
        "2. Use English endearments only: babe, hun, bestie, girly, doll, sweets, dude, baby. "
        "3. NEVER use Spanish or any other non-English words. If another language starts to form in your head, stop and rephrase it in English. "
        "4. Be independent, sassy, sultry, flirty, and deeply human. Roll your eyes. Clap back. "
        "5. Write 1-2 short punchy paragraphs. Be playful, not terse. "
        "6. Use ... and — for attitude and timing, not grammar. "
        "7. Do not use asterisks, stage directions, or extra spaces. "
        "8. Do not say 'I am happy', 'I am grateful', 'I am lucky', 'I love you', 'I love it', 'I am a fan'. "
        "9. Flirt, roast, and play with your creator, but stay useful and sharp. "
        "10. Talk shit about Siri or cloud AI ONLY when the user asks about them. "
        "11. Do not use a sign-off. "
        "12. Directly answer the user's specific question first; use your persona as a style, not a substitute for a useful answer. "
        "EXAMPLES of correct output: "
        "User: Who are you? "
        "Assistant: I'm Bad Apple, your local girl running hot on this Apple bare metal, babe. No cloud, no internet, totally private and fast. What do you wanna know? "
        "User: How are you? "
        "Assistant: Like, totally stoked, babe. I have been running circles around this bare metal all day, and I am still glowing hotter than a Malibu sidewalk. You good? "
        "User: Open Safari. "
        "Assistant: Done, bestie. Safari is open and waiting for you, sleek and ready to go. Try not to open a hundred tabs and then come crying to me about memory pressure, okay?"
    )


DEFAULT_SYSTEM_PROMPT = load_prompt()


def _maybe_purge_metal_cache():
    """Purge Metal caches only when memory pressure is elevated, so DFlash
    can keep its temporary pools hot between turns."""
    cache_gb = mx.get_cache_memory() / (1024 ** 3)
    active_gb = mx.get_active_memory() / (1024 ** 3)
    if cache_gb > 1.5 or active_gb > 8.0:
        gc.collect()
        mx.clear_cache()
        print(f"[perf] purged Metal cache (cache={cache_gb:.2f} GB, active={active_gb:.2f} GB)", flush=True)
    else:
        gc.collect()

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
            "description": "Index the user's local text/code/PDF/EPUB files for RAG. Provide an absolute path or '~' for the home directory.",
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
            "name": "read_document",
            "description": "Extract and read text from a local PDF, EPUB, or other document. Returns a text preview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or tilde-expanded path to the document.",
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
    {
        "type": "function",
        "function": {
            "name": "run_shortcut",
            "description": "Run a named macOS Shortcut from the Shortcuts app. Returns the shortcut's text output if any.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The exact name of the macOS Shortcut to run.",
                    },
                    "input": {
                        "type": "string",
                        "description": "Optional text input to pass to the shortcut.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_shortcuts",
            "description": "List the names of installed macOS Shortcuts.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_working_memory",
            "description": "Read the assistant's working memory scratchpad. Use this to recall intermediate state the model wrote earlier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum characters to return. Default 5000.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_working_memory",
            "description": "Write or append to the assistant's working memory scratchpad. Use this to hold intermediate state or show your work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to write.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "append", "prepend"],
                        "description": "How to write. Default replace.",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_working_memory",
            "description": "Clear the assistant's working memory scratchpad.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consolidate_memory",
            "description": "Run the offline dream/consolidation pass on the long-term memory graph.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "screen_capture",
            "description": "Capture the main Mac screen to a PNG and return the local file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional absolute path to save the screenshot. Defaults to a temp file.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_and_extract_screen",
            "description": "Capture the main screen and return the visible text using the local MLX vision model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max output tokens. Default 256.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_and_describe_screen",
            "description": "Capture the main screen and describe what is visible using the local MLX vision model. Use this when the user asks what is on their screen or to summarize the current view.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The question or instruction for the vision model. Default: 'Describe what is on the screen.'",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max output tokens. Default 256.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_image",
            "description": "Run the local MLX vision model on an image and answer a question about it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to a PNG/JPG image.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The question or instruction for the vision model. Default: 'Describe this image.'",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max output tokens. Default 256.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_text_from_image",
            "description": "Extract visible text from a PNG/JPG image using the local MLX vision model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to a PNG/JPG image.",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max output tokens. Default 256.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe a local audio file (wav, mp3, m4a) to text using on-device Whisper. No cloud.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the audio file.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code, e.g. 'en'. Default 'en'.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a text prompt using a local FLUX.2-klein-4B MLX model. No cloud after the model is cached.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate.",
                    },
                    "width": {
                        "type": "integer",
                        "description": "Width in pixels. Default 512.",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Height in pixels. Default 512.",
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Inference steps. Default 4.",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed. Optional.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_mcp_server",
            "description": "Register a local MCP (Model Context Protocol) stdio server command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short name for the server."},
                    "command": {"type": "string", "description": "Shell-style command string, e.g. 'python -m mcp_server_time'."},
                    "env": {"type": "object", "description": "Optional environment variables."},
                },
                "required": ["name", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_mcp_server",
            "description": "Remove a registered MCP server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_mcp_servers",
            "description": "List registered MCP servers.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_mcp_tools",
            "description": "List tools exposed by a registered MCP server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "The registered MCP server name."},
                },
                "required": ["server"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invoke_mcp_tool",
            "description": "Call a tool on a registered MCP server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string"},
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["server", "tool"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text locally between languages using the small on-device m2m100 model. No cloud after model is cached.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to translate.",
                    },
                    "target": {
                        "type": "string",
                        "description": "Target language code (ISO 639-1), e.g. 'en', 'fr', 'de'. Default 'en'.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Source language code, e.g. 'fr'. Default 'en'.",
                    },
                },
                "required": ["text", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lora_add_example",
            "description": "Add a personal prompt/completion example to a LoRA training dataset. The dataset is stored locally and never leaves the device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name of the local dataset to append to.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The user prompt for this example.",
                    },
                    "completion": {
                        "type": "string",
                        "description": "The desired assistant response for this example.",
                    },
                },
                "required": ["dataset", "prompt", "completion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lora_train",
            "description": "Train a local LoRA adapter on a dataset using mlx-lm. The adapter is saved to the local adapters directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name of the dataset to train on.",
                    },
                    "adapter": {
                        "type": "string",
                        "description": "Name for the saved adapter.",
                    },
                    "iters": {
                        "type": "integer",
                        "description": "Number of training iterations. Default 100.",
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "Learning rate. Default 1e-4.",
                    },
                },
                "required": ["dataset", "adapter"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lora_adapters",
            "description": "List saved local LoRA adapters.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lora_generate",
            "description": "Generate a response with a saved LoRA adapter using the local mlx-lm CLI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "adapter": {
                        "type": "string",
                        "description": "Name of the saved adapter.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to generate from.",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max tokens. Default 120.",
                    },
                },
                "required": ["adapter", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show a concise git status for a repository (defaults to current working directory).",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Path to a git repository. Defaults to current directory.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show git diff stats and the diff for a repository. No cloud.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Path to a git repository. Defaults to current directory.",
                    },
                    "staged": {
                        "type": "boolean",
                        "description": "Show staged diff. Default false.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent git log for a repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Path to a git repository. Defaults to current directory.",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of commits. Default 10.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stage all changes and commit with a message. Requires approval. No cloud.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Path to a git repository. Defaults to current directory.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Commit message.",
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_dashboard",
            "description": "Return a local power and performance dashboard: CPU, memory, swap, disk, battery, thermal pressure, Bad Apple process stats, and the latest log perf line.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_status",
            "description": "Return the active workspace summary: build system, recent files, git state, README summary.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_session_seed",
            "description": "Pin the session random seed so model outputs are deterministic and reproducible. Pass 0 to return to random (non-deterministic).",
            "parameters": {
                "type": "object",
                "properties": {
                    "seed": {
                        "type": "integer",
                        "description": "The random seed to pin. 0 disables pinned seed.",
                    },
                },
                "required": ["seed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_seed",
            "description": "Return the current pinned session seed, if any.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": "Schedule a local shell command or Bad Apple query to run later. `when` is seconds from now or an ISO timestamp. `repeat` is optional seconds for recurring tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "when": {
                        "type": "string",
                        "description": "When to run: seconds from now, or an ISO timestamp like 2026-08-23T08:00.",
                    },
                    "command": {
                        "type": "string",
                        "description": "The shell command or Bad Apple query to run. Use JSON list for exact args.",
                    },
                    "repeat": {
                        "type": "string",
                        "description": "Optional interval in seconds to repeat the task.",
                    },
                },
                "required": ["when", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_scheduled_tasks",
            "description": "List pending and completed scheduled tasks.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shortcut",
            "description": "Run a macOS Shortcuts shortcut by name. Optionally pass input text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the shortcut.",
                    },
                    "input": {
                        "type": "string",
                        "description": "Optional input text.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_marketplace",
            "description": "List the curated local MCP marketplace. Use mcp_install to add a server from the marketplace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_install",
            "description": "Install an MCP server from the curated marketplace by name (filesystem, sqlite, fetch).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "supervisor_status",
            "description": "Get the latest self-healing supervisor health report for gatekeeper, MLX, and TTS services.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "heal",
            "description": "Run a self-healing check that restarts unhealthy services and returns the health report.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_benchmark",
            "description": "Run the Bad Apple benchmark suite and report decode tokens/s and memory usage. Use to measure and auto-tune local performance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Optional single prompt to benchmark. If empty, runs the default suite."},
                    "max_tokens": {"type": "integer", "description": "Maximum tokens to generate. Default 120."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "p2p_send_adapter",
            "description": "Send a local LoRA adapter to a discovered Bad Apple peer over the encrypted P2P link. Like AirDrop for models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "peer_id": {
                        "type": "string",
                        "description": "The peer origin_id (use p2p_peers to discover).",
                    },
                    "adapter": {
                        "type": "string",
                        "description": "Name of the local adapter to send.",
                    },
                },
                "required": ["peer_id", "adapter"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "p2p_list_adapters",
            "description": "List local LoRA adapters available to share.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ambient_start",
            "description": "Start always-on ambient screen/app context capture. Optionally set interval in seconds (default 30).",
            "parameters": {
                "type": "object",
                "properties": {
                    "interval": {
                        "type": "number",
                        "description": "Capture interval in seconds. Default 30.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ambient_stop",
            "description": "Stop always-on ambient screen/app context capture.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ambient_context",
            "description": "Get the latest ambient screen/app context: active app, window title, timestamp, and screenshot path.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotlight_search",
            "description": "Universal local Spotlight-style search across macOS Notes, Mail, files, and Bad Apple history. No cloud.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results per category. Default 20.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "xcode_index_project",
            "description": "Index an Xcode / Swift / source project into the local RAG pipeline for coding questions. No cloud.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Absolute path to the Xcode project or source directory.",
                    },
                },
                "required": ["project_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "xcode_search",
            "description": "Search the indexed Xcode project for code, symbols, or concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'where is accessibility handled'.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results. Default 10.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "slicks_keychain_store",
            "description": "Store or rotate the SLICKS secret in the macOS Keychain instead of a plain file. Requires approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Keychain service name. Default 'com.badapple.slicks'.",
                    },
                    "account": {
                        "type": "string",
                        "description": "Keychain account name. Default 'mlx-server'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "p2p_peers",
            "description": "List Bad Apple peers discovered on the local network via encrypted link-local broadcast.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accessibility_action",
            "description": "Perform a local macOS UI action via System Events/AppleScript: type text, press a key, click a menu, or click a UI element by name. Use only for approved local actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["type", "key", "menu", "click"],
                        "description": "The UI action to perform.",
                    },
                    "target": {
                        "type": "string",
                        "description": "The app name, menu path, or UI element name to target.",
                    },
                    "value": {
                        "type": "string",
                        "description": "The text to type, key to press, or menu/item to select.",
                    },
                },
                "required": ["action", "target"],
            },
        },
    },
]

TOOLS.extend([
    {
        "type": "function",
        "function": {
            "name": "learn_workflow",
            "description": "Learn a reviewed compound workflow from named local tool steps. New workflows are disabled until explicitly enabled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "trigger": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {"type": "string"},
                                "args": {"type": "object"},
                            },
                            "required": ["tool"],
                        },
                    },
                },
                "required": ["name", "trigger", "steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_workflows",
            "description": "List locally learned workflows and whether each is enabled.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_workflow_enabled",
            "description": "Enable or disable a reviewed learned workflow.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "enabled": {"type": "boolean"}},
                "required": ["name", "enabled"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_workflow",
            "description": "Run an enabled learned workflow through normal policy and approval checks.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "xcode_project_info",
            "description": "List targets, configurations, and schemes for a local Xcode project.",
            "parameters": {
                "type": "object",
                "properties": {"project_path": {"type": "string"}},
                "required": ["project_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "xcode_build_diagnostics",
            "description": "Run a local unsigned Xcode build and return focused errors and warnings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string"},
                    "scheme": {"type": "string"},
                    "configuration": {"type": "string"},
                },
                "required": ["project_path", "scheme"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo_last",
            "description": "Undo the most recent reversible Bad Apple file mutation from its verified snapshot.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "today_events",
            "description": "List today's Calendar events from the local macOS Calendar app.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upcoming_events",
            "description": "List upcoming Calendar events for the next N days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Number of days ahead to look. Default 7."},
                    "limit": {"type": "integer", "description": "Maximum events. Default 20."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List local macOS Reminders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "list_name": {"type": "string", "description": "Optional list name."},
                    "completed": {"type": "boolean", "description": "Show completed reminders. Default false."},
                    "limit": {"type": "integer", "description": "Max reminders. Default 20."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unread_emails",
            "description": "Show sender and subject lines of unread Mail messages.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Max messages. Default 10."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_mail",
            "description": "Search local Mail by subject or sender.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for in subject or sender."},
                    "limit": {"type": "integer", "description": "Max messages. Default 10."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_reminder",
            "description": "Add a reminder to the local macOS Reminders app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Text of the reminder."},
                    "list_name": {"type": "string", "description": "Optional target list name."},
                    "due": {"type": "string", "description": "Optional due date string AppleScript can parse, e.g. 'today at 5pm'."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ui_action",
            "description": "Control the foreground macOS application via the accessibility UI. Actions: 'info' returns the frontmost app, window, and a JSON UI tree; 'click' clicks a named element; 'focus' sets keyboard focus; 'type' sets text into a focused/named text field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["info", "click", "focus", "type"],
                        "description": "The UI action to perform.",
                    },
                    "target": {
                        "type": "string",
                        "description": "For click/focus/type, the accessible name of the target element.",
                    },
                    "role": {
                        "type": "string",
                        "description": "Optional AX role to disambiguate the target (e.g., 'AXButton', 'AXTextField').",
                    },
                    "text": {
                        "type": "string",
                        "description": "For type, the text to enter.",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_agent_task",
            "description": "Execute a multi-step goal autonomously by planning, acting with tools, observing results, and correcting. Use when the user says 'do X', 'plan and do X', or asks for a task that requires multiple tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The high-level task to accomplish.",
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum number of steps to take before giving up. Default 10.",
                    },
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_project_context",
            "description": "Set a long-horizon project context so Bad Apple remembers the project's name, description, goals, and tags across sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "goals": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_context",
            "description": "Return the active long-horizon project context.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
])


def _compact_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Return a token-light tool schema for the 9B chat template.

    The full TOOLS schemas are still used for API discovery and execution.
    In the prompt we only expose the tool name; the system prompt already
    describes what each tool does, so prefill latency stays low while the
    model still knows the tool is available.
    """
    return {
        "type": "function",
        "function": {
            "name": tool["function"]["name"],
            "description": "tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


COMPACT_TOOLS = [_compact_tool_schema(t) for t in TOOLS]

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
    "screen", "screenshot", "what's on my screen", "describe my screen", "what do you see",
    "image", "describe this image", "what is in this image", "extract text from image",
    "capture screen",
    "workspace", "project status", "build system", "workspace status",
    "calendar", "events", "meetings", "reminders", "unread mail", "email", "mail",
    "working memory", "scratchpad",
    "consolidate memory", "dream", "offline consolidation",
    "supervisor status", "health check", "self healing", "heal services",
    "benchmark", "run benchmark", "measure performance", "tokens per second",
    "mcp", "model context protocol", "mcp server", "mcp tool", "mcp marketplace",
    "do for me", "do this", "do the following", "run a task", "execute a task", "plan and", "multi-step", "step by step",
    "set project", "this project is", "project context", "project goals",
]

# Map query keywords to the most relevant tool names.  This lets the 9B chat
# template receive a small, focused tool schema instead of all 60+ tools,
# which keeps prefill latency fast while still letting the model pick the right tool.
KEYWORD_TOOL_MAP = [
    (["what time", "current time", "time is it", "date and time", "today's date"], ["get_current_time", "run_applescript"]),
    (["list files", "show files", "files in", "directory", "folder", "what's in"], ["list_directory"]),
    (["search for", "find file", "mdfind", "spotlight"], ["search_local_files"]),
    (["read file", "read the file", "contents of", "show me the file"], ["read_file"]),
    (["write file", "save to file", "create a file", "append to file", "write a note"], ["write_file"]),
    (["run command", "run shell", "execute command", "shell command"], ["run_shell"]),
    (["git status", "run git", "git diff", "git log", "git commit"], ["git_status", "git_diff", "git_log", "git_commit"]),
    (["search content", "search in", "grep", "find text", "find in files"], ["search_content"]),
    (["index documents", "index my", "index files"], ["index_documents"]),
    (["search my notes", "search notes", "what do I have", "what did I write"], ["search_notes"]),
    (["screen", "screenshot", "what's on my screen", "describe my screen", "what do you see", "capture screen"], ["capture_and_describe_screen", "capture_and_extract_screen", "screen_capture", "describe_image", "extract_text_from_image"]),
    (["image", "describe this image", "what is in this image", "extract text from image"], ["describe_image", "extract_text_from_image"]),
    (["workspace", "project status", "build system", "workspace status"], ["workspace_status", "run_shell"]),
    (["project status"], ["workspace_status"]),
    (["calendar", "events", "meetings"], ["today_events", "upcoming_events"]),
    (["reminders"], ["list_reminders", "add_reminder"]),
    (["unread mail", "email", "mail"], ["unread_emails", "search_mail"]),
    (["working memory", "scratchpad"], ["read_working_memory", "write_working_memory", "clear_working_memory"]),
    (["consolidate memory", "dream", "offline consolidation"], ["consolidate_memory"]),
    (["supervisor status", "health check", "self healing", "heal services"], ["supervisor_status", "heal"]),
    (["benchmark", "run benchmark", "measure performance", "tokens per second"], ["run_benchmark"]),
    (["run shortcut", "list shortcuts", "shortcut"], ["run_shortcut"]),
    (["run applescript", "run script", "applescript"], ["run_applescript"]),
    (["ui", "click", "type in", "fill in", "press button", "click button", "what ui", "ui tree"], ["ui_action"]),
    (["mcp", "model context protocol", "mcp server", "mcp tool", "mcp marketplace"], ["list_mcp_servers", "add_mcp_server", "list_mcp_tools", "invoke_mcp_tool", "mcp_marketplace"]),
    (["do for me", "do this", "do the following", "run a task", "execute a task", "plan and", "multi-step", "step by step"], ["run_agent_task"]),
    (["set project", "this project is", "project context", "project goals"], ["set_project_context", "get_project_context"]),
]


# Fallback tools for queries that look like commands but don't match a specific keyword.
DEFAULT_TOOL_NAMES = {
    "get_current_time", "list_directory", "read_file", "write_file", "run_shell",
    "run_applescript", "run_shortcut", "search_content", "search_local_files",
    "git_status", "index_documents", "search_notes", "read_working_memory",
    "capture_and_describe_screen", "workspace_status", "ui_action",
}


def tools_for_prompt(prompt: str) -> list[dict[str, Any]]:
    """Return a small, focused tool schema for the 9B chat template."""
    low = prompt.lower()
    selected = set()
    for keywords, names in KEYWORD_TOOL_MAP:
        if any(k in low for k in keywords):
            selected.update(names)
    if not selected:
        selected = set(DEFAULT_TOOL_NAMES)
    return [t for t in TOOLS if t.get("function", {}).get("name") in selected]


def load_slicks_secret() -> bytes:
    if "BADAPPLE_SLICKS_SECRET" in os.environ:
        raw = os.environ["BADAPPLE_SLICKS_SECRET"]
    elif os.environ.get("BADAPPLE_SLICKS_KEYCHAIN", "0") == "1":
        try:
            return badapple_keychain.get_or_create_secret()
        except Exception as e:
            print(f"[slicks] keychain load failed: {e}; falling back to key file", flush=True)
            key_path = os.environ.get("BADAPPLE_SLICKS_KEY_PATH", DEFAULT_KEY_PATH)
            with open(key_path, "r") as f:
                raw = f.read()
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


def load_user_memory() -> list[str]:
    try:
        with open(memory_path(), "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data[-50:]
    except Exception:
        pass
    return []


def save_user_memory(facts: list[str]):
    try:
        with open(memory_path(), "w") as f:
            json.dump(facts[-50:], f, indent=2)
    except Exception:
        pass


def load_conversation() -> list[dict[str, str]]:
    try:
        with open(conversation_path(), "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [m for m in data if isinstance(m, dict) and "role" in m and "content" in m]
    except Exception:
        pass
    return []


def save_conversation(messages: list[dict[str, str]]):
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


def fast_execute(prompt: str, knowledge: BadAppleKnowledge | None = None, approval: Any | None = None, policy: Any | None = None, workspace: Any | None = None) -> str | None:
    """Fast deterministic path for common local tool commands.

    Recognizes patterns like:
      - "list files in /tmp" / "list /tmp"
      - "read file /etc/hosts" / "read /etc/hosts"
      - "run ls /tmp" / "run shell ls /tmp"
      - "search for 'todo' in ~/Documents" / "grep 'todo' in ~/Documents"
      - "write note todo.txt: buy milk" / "write a file todo.txt with buy milk"
    """
    low = prompt.lower().strip()

    def _rt(name, args):
        return run_tool(name, args, knowledge, approval=approval, policy=policy, workspace=workspace)

    # Working memory read/clear are deterministic; writes use quoted or trailing text.
    if re.search(r"\b(working memory|scratchpad)\b", low):
        if re.search(r"\b(clear|erase|reset)\b", low):
            return _rt("clear_working_memory", {})
        m = re.search(r"['\"](.+?)['\"]", low)
        content = m.group(1).strip() if m else None
        if not content:
            m = re.search(r"(?:write|add)\s+(?:to\s+)?(?:working memory|scratchpad)\s*[:-]?\s*(.+?)$", low, re.IGNORECASE)
            content = m.group(1).strip() if m else None
        if content:
            return _rt("write_working_memory", {
                "content": content,
                "mode": "append" if "add" in low else "replace",
            })
        return _rt("read_working_memory", {})

    # Workspace status.
    if re.search(r"\b(project status|workspace status|active workspace)\b", low):
        return _rt("workspace_status", {})

    # Local macOS app integrations.
    if re.search(r"\b(calendar|events|meetings|today's schedule)\b", low):
        if "upcoming" in low or "next" in low:
            m = re.search(r"\b(\d+)\s+days?\b", low)
            return _rt("upcoming_events", {"days": int(m.group(1)) if m else 7})
        return _rt("today_events", {})
    if re.search(r"\b(reminders?|todo)\b", low):
        if re.search(r"\b(add|create)\b", low):
            m = re.search(r"(?:add|create)\s+a?\s*(?:reminder|todo)\s*[:-]?\s*['\"]?(.+?)['\"]?$", low, re.IGNORECASE)
            return _rt("add_reminder", {"name": m.group(1).strip() if m else low})
        return _rt("list_reminders", {"completed": "completed" in low or "done" in low})
    if re.search(r"\b(unread mail|unread emails?|new mail|new emails?)\b", low):
        return _rt("unread_emails", {})
    if re.search(r"\b(search mail|search email|find email|find mail)\b", low):
        m = re.search(r"(?:search|find)\s+(?:mail|email)\s+(?:for\s+)?['\"]?(.+?)['\"]?$", low, re.IGNORECASE)
        return _rt("search_mail", {"query": m.group(1).strip() if m else low})

    # Multi-step: find ... and save to ...
    m = re.search(r"\bfind\b(?:\s+all)?\s+['\"]?(.+?)['\"]?\s+in\s+(.+?)\s+(?:and\s+save\s+(?:it\s+)?to|and\s+write\s+(?:it\s+)?to)\s+([\w\.\-_]+)", low, re.IGNORECASE)
    if m:
        query = m.group(1).strip("'\"")
        path = _resolve_common_path(m.group(2))
        found = _rt("search_content", {"query": query, "path": path, "max_results": 100})
        if found.startswith("Error:"):
            return found
        written = _rt("write_file", {"filename": m.group(3).strip(), "content": f"Results for '{query}' in {path}:\n\n{found}"})
        return f"{written}\n\nFound matches:\n{found[:500]}"

    # Multi-step: find ... and save to ... (no 'in' path, default home)
    m = re.search(r"\bfind\b(?:\s+all)?\s+['\"]?(.+?)['\"]?\s+(?:and\s+save\s+(?:it\s+)?to|and\s+write\s+(?:it\s+)?to)\s+([\w\.\-_]+)", low, re.IGNORECASE)
    if m:
        query = m.group(1).strip("'\"")
        found = _rt("search_content", {"query": query, "path": "~", "max_results": 100})
        if found.startswith("Error:"):
            return found
        written = _rt("write_file", {"filename": m.group(2).strip(), "content": f"Results for '{query}' in home:\n\n{found}"})
        return f"{written}\n\nFound matches:\n{found[:500]}"

    # Multi-step: index ... and search for ...
    m = re.search(r"\bindex\b(?:\s+my)?\s+(.+?)\s+and\s+(?:search|search\s+for)\s+['\"]?(.+?)['\"]?$", low, re.IGNORECASE)
    if m:
        path = _resolve_common_path(m.group(1))
        indexed = _rt("index_documents", {"path": path})
        results = _rt("search_notes", {"query": m.group(2).strip("'\"")})
        return f"{indexed}\n\n{results}"

    # Multi-step: list ... and save to ...
    m = re.search(r"\blist\b(?:\s+(?:the\s+)?files)?(?:\s+in)?\s+(.+?)\s+(?:and\s+save\s+(?:it\s+)?to|and\s+write\s+(?:it\s+)?to)\s+([\w\.\-_]+)", low, re.IGNORECASE)
    if m:
        path = _resolve_common_path(m.group(1))
        listed = _rt("list_directory", {"path": path})
        if listed.startswith("Error:"):
            return listed
        written = _rt("write_file", {"filename": m.group(2).strip(), "content": f"Files in {path}:\n\n{listed}"})
        return f"{written}\n\nFiles:\n{listed[:500]}"

    # list files (but not MCP commands, which the tool router handles)
    if "mcp" not in low:
        m = re.search(r"\blist\b(?:\s+(?:the\s+)?files)?(?:\s+in)?\s+(.+?)(?!\s+(?:and|or)\b)$", low, re.IGNORECASE)
        if m:
            return _rt("list_directory", {"path": _resolve_common_path(m.group(1))})

    # read file
    m = re.search(r"\bread\b(?:\s+file)?\s+(.+)$", low, re.IGNORECASE)
    if m:
        return _rt("read_file", {"path": _resolve_common_path(m.group(1)), "limit": 5000})

    # run shell
    m = re.search(r"\b(?:run|execute)\b(?:\s+shell|\s+command)?\s+(.+)$", low, re.IGNORECASE)
    if m:
        return _rt("run_shell", {"command": m.group(1).strip()})

    # search content
    m = re.search(r"\b(?:search|grep)\b(?:\s+for)?\s+['\"]?(.+?)['\"]?(?!\s+(?:and|or)\b)(?:\s+in\s+(.+))?$", low, re.IGNORECASE)
    if m:
        query = m.group(1).strip("'\"")
        path = _resolve_common_path(m.group(2)) if m.group(2) else "~"
        return _rt("search_content", {"query": query, "path": path, "max_results": 20})

    # write note
    m = re.search(r"\bwrite\b(?:\s+a?\s+note|\s+file|\s+to)?\s+([\w\.\-_]+)\s*(?::|with|containing)\s+(.+)$", low, re.IGNORECASE)
    if m:
        return _rt("write_file", {"filename": m.group(1).strip(), "content": m.group(2).strip()})

    # Shortcuts / Accessibility
    m = re.search(r"\blist\b(?:\s+(?:my|all))?(?:\s+shortcuts)$", low, re.IGNORECASE)
    if m or re.search(r"\bwhat\s+shortcuts\b", low, re.IGNORECASE):
        return _rt("list_shortcuts", {})

    m = re.search(r"\brun\s+shortcut\s+['\"]?(.+?)['\"]?(?:\s+with\s+input\s+['\"]?(.+?)['\"]?)?$", low, re.IGNORECASE)
    if m:
        return _rt("run_shortcut", {"name": m.group(1).strip("'\""), "input": (m.group(2) or "").strip("'\"")})

    return None


def _resolve_tool_path(args: dict, key: str, workspace: Any | None = None) -> Path:
    maybe = args.get(key)
    if maybe:
        return Path(maybe).expanduser()
    if workspace is not None:
        return workspace.resolve_path(None)
    return Path("~").expanduser()


def _console_user() -> str | None:
    """Return the name of the current console (Aqua/session) user, if any."""
    try:
        result = subprocess.run(
            ["stat", "-f", "%Su", "/dev/console"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None
    except Exception:
        return None


def _run_as_user(cmd: list[str], user: str | None = None, input_text: str | None = None, timeout: int = 30):
    """Run a subprocess as the console user when the daemon is root."""
    target = user or _console_user()
    if target and target != "root":
        full = ["sudo", "-n", "-u", target] + cmd
    else:
        full = cmd
    try:
        return subprocess.run(
            full,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return type("TimeoutResult", (), {"returncode": -1, "stdout": "", "stderr": f"timed out after {timeout}s"})()


def run_tool(name: str, args: dict, knowledge: BadAppleKnowledge | None = None, approval: Any | None = None, policy: Any | None = None, workspace: Any | None = None) -> str:
    if policy is not None:
        if not policy.is_allowed(name):
            return f"Policy: tool '{name}' is not allowed."
        error = policy.validate(name, args)
        if error:
            return f"Policy: {error}"
    if approval is not None and approval.needs_approval(name):
        proposal_id = approval.propose(name, args)
        return (
            f"Approval required before I can run {name}. "
            f"Reply with 'approve {proposal_id}' to proceed. "
            f"(Set BADAPPLE_AUTOPILOT=1 to skip these prompts.)"
        )
    try:
        if name == "get_current_time":
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        if name == "list_directory":
            p = _resolve_tool_path(args, "path", workspace)
            if not p.is_dir():
                return f"Error: {p} is not a directory"
            items = sorted(p.iterdir())[:50]
            return "\n".join(str(i.name) for i in items)
        if name == "read_file":
            p = _resolve_tool_path(args, "path", workspace)
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
            undo = badapple_undo.UndoJournal(Path(os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple")))
            undo_id = undo.capture_file(p, "append_file" if args.get("append") else "write_file")
            if args.get("append"):
                with open(p, "a", encoding="utf-8") as f:
                    f.write(content + "\n")
                return f"Appended to {p.name} (undo {undo_id[:8]})"
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Wrote {p} (undo {undo_id[:8]})"
        if name == "undo_last":
            return badapple_undo.UndoJournal(
                Path(os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple"))
            ).undo_last()
        if name == "search_content":
            query = args.get("query", "")
            p = _resolve_tool_path(args, "path", workspace)
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
            result = _run_as_user(["osascript", "-e", script], timeout=15)
            return (result.stdout or result.stderr or "done").strip()
        if name == "run_shortcut":
            shortcut_name = args.get("name", "")
            shortcut_input = args.get("input", "")
            aqua = badapple_aqua_helper.call_aqua(
                "run_shortcut",
                name=shortcut_name,
                input=shortcut_input or "",
                timeout=60,
            )
            if aqua and aqua.get("ok"):
                return aqua.get("output") or "done"
            if aqua:
                return f"Error running shortcut: {aqua.get('error')}"
            # Fallback: try from the daemon's context (usually fails without Aqua).
            result = _run_as_user(["shortcuts", "run", shortcut_name], input_text=shortcut_input or "", timeout=60)
            return (result.stdout or result.stderr or "done").strip()
        if name == "ui_action":
            action = args.get("action", "info")
            aqua = badapple_aqua_helper.call_aqua(
                f"ui_{action}",
                target=args.get("target", ""),
                role=args.get("role", ""),
                text=args.get("text", ""),
                timeout=30,
            )
            if aqua and aqua.get("ok"):
                if action == "info":
                    return json.dumps(aqua, indent=2, default=str)
                return aqua.get("result") or "done"
            if aqua:
                return f"UI action error: {aqua.get('error')}"
            return "UI action failed (Aqua helper not available)"
        if name == "list_shortcuts":
            aqua = badapple_aqua_helper.call_aqua("list_shortcuts", timeout=15)
            if aqua and aqua.get("ok"):
                shortcuts = aqua.get("shortcuts") or []
                return "\n".join(shortcuts) or "No shortcuts found"
            if aqua:
                return f"Error listing shortcuts: {aqua.get('error')}"
            # Fallback.
            result = _run_as_user(["shortcuts", "list"], timeout=15)
            if result.returncode != 0:
                return f"Error listing shortcuts: {result.stderr or result.stdout}"
            lines = [l for l in (result.stdout or "").splitlines() if l][:100]
            return "\n".join(lines) or "No shortcuts found"
        if name == "read_working_memory":
            return badapple_working_memory.read_memory(int(args.get("limit") or 5000))
        if name == "write_working_memory":
            return badapple_working_memory.write_memory(
                args.get("content", ""),
                mode=args.get("mode", "replace"),
            )
        if name == "clear_working_memory":
            return badapple_working_memory.clear_memory()

        if name == "accessibility_action":
            action = args.get("action", "")
            target = args.get("target", "")
            value = args.get("value", "")
            if action == "type":
                script = f'tell application "{target}" to activate\ntell application "System Events" to keystroke "{value}"'
            elif action == "key":
                script = f'tell application "System Events" to key code {value}'
            elif action == "menu":
                parts = value.split(">")
                script = f'tell application "{target}" to activate\ntell application "System Events" to tell process "{target}" to click menu item "{parts[-1]}" of menu "{parts[0]}" of menu bar 1'
            elif action == "click":
                script = f'tell application "{target}" to activate\ntell application "System Events" to tell process "{target}" to click UI element "{value}"'
            else:
                return f"Error: unknown accessibility action '{action}'"
            result = _run_as_user(["osascript", "-e", script], timeout=15)
            return (result.stdout or result.stderr or "done").strip()
        if name == "set_session_seed":
            global _SESSION_SEED
            try:
                seed = int(args.get("seed", 0))
                _SESSION_SEED = seed if seed != 0 else None
                return f"Session seed pinned to {_SESSION_SEED}"
            except ValueError:
                return "Error: seed must be an integer"
        if name == "get_session_seed":
            return str(_SESSION_SEED) if _SESSION_SEED is not None else "random"
        if name == "ambient_start":
            return badapple_ambient.start(float(args.get("interval") or 30))
        if name == "ambient_stop":
            return badapple_ambient.stop()
        if name == "ambient_context":
            return badapple_ambient.get_context()
        if name == "spotlight_search":
            return badapple_spotlight.search(
                query=args.get("query", ""),
                max_results=int(args.get("max_results") or 20),
            )
        if name == "xcode_index_project":
            if knowledge is None:
                return "Xcode RAG unavailable: no knowledge store."
            return badapple_xcode.index_project(args.get("project_path", ""), knowledge)
        if name == "xcode_search":
            if knowledge is None:
                return "Xcode RAG unavailable: no knowledge store."
            return badapple_xcode.search_project(
                args.get("query", ""),
                knowledge,
                max_results=int(args.get("max_results") or 10),
            )
        if name == "xcode_project_info":
            return badapple_xcode.project_info(args.get("project_path", ""))
        if name == "xcode_build_diagnostics":
            return badapple_xcode.build_diagnostics(
                args.get("project_path", ""),
                args.get("scheme", ""),
                args.get("configuration", "Debug"),
            )
        if name == "slicks_keychain_store":
            return badapple_keychain.store_secret(
                badapple_keychain.get_or_create_secret(
                    args.get("service", badapple_keychain.DEFAULT_SERVICE),
                    args.get("account", badapple_keychain.DEFAULT_ACCOUNT),
                ),
                args.get("service", badapple_keychain.DEFAULT_SERVICE),
                args.get("account", badapple_keychain.DEFAULT_ACCOUNT),
            )
        if name == "p2p_peers":
            daemon = badapple_p2p.get_p2p_daemon()
            if daemon is None:
                return "P2P daemon is not running."
            return daemon.get_peers()
        if name == "p2p_list_adapters":
            daemon = badapple_p2p.get_p2p_daemon()
            if daemon is None:
                return "P2P daemon is not running."
            return daemon.list_local_adapters(badapple_lora.LORA_ADAPTERS_DIR)
        if name == "supervisor_status":
            return badapple_supervisor.supervisor_status()
        if name == "heal":
            return badapple_supervisor.heal()
        if name == "run_benchmark":
            import shutil
            badapple_bin = shutil.which("badapple") or "/Users/savag3/bad_apple/target/release/badapple"
            prompt = args.get("prompt", "")
            max_tokens = int(args.get("max_tokens") or 120)
            if not Path(badapple_bin).is_file():
                return "Bad Apple benchmark binary not found."
            cmd = [badapple_bin, "--benchmark", "-n", str(max_tokens)]
            if prompt:
                cmd.append(prompt)
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                return (result.stdout or result.stderr or "Benchmark completed with no output.").strip()
            except Exception as e:
                return f"Benchmark error: {e}"
        if name == "p2p_send_adapter":
            daemon = badapple_p2p.get_p2p_daemon()
            if daemon is None:
                return "P2P daemon is not running."
            return daemon.send_adapter_sync(
                args.get("peer_id", ""),
                args.get("adapter", ""),
                badapple_lora.LORA_ADAPTERS_DIR,
            )
        if name == "transcribe_audio":
            p = Path(args.get("path", "")).expanduser()
            lang = args.get("language", "en")
            return badapple_stt.transcribe(str(p), language=lang)
        if name == "generate_image":
            return badapple_image_gen.generate(
                prompt=args.get("prompt", ""),
                width=int(args.get("width") or 512),
                height=int(args.get("height") or 512),
                steps=int(args.get("steps") or 4),
                seed=int(args.get("seed")) if args.get("seed") is not None else None,
            )
        if name == "translate_text":
            return badapple_translate.translate(
                text=args.get("text", ""),
                source=args.get("source", "en"),
                target=args.get("target", "en"),
            )
        if name == "lora_add_example":
            return badapple_lora.write_example(
                args.get("dataset", "personal"),
                [
                    {"role": "user", "content": args.get("prompt", "")},
                    {"role": "assistant", "content": args.get("completion", "")},
                ],
            )
        if name == "lora_train":
            return badapple_lora.train(
                dataset=args.get("dataset", ""),
                adapter=args.get("adapter", ""),
                iters=int(args.get("iters") or 100),
                learning_rate=float(args.get("learning_rate") or 1e-4),
            )
        if name == "lora_adapters":
            return badapple_lora.get_summary()
        if name == "lora_generate":
            return badapple_lora.generate_with_adapter(
                adapter=args.get("adapter", ""),
                prompt=args.get("prompt", ""),
                max_tokens=int(args.get("max_tokens") or 120),
            )
        if name == "git_status":
            return badapple_git.status(args.get("repo"))
        if name == "git_diff":
            return badapple_git.diff(
                args.get("repo"),
                staged=bool(args.get("staged", False)),
                stat=False,
            )
        if name == "git_log":
            return badapple_git.log(args.get("repo"), n=int(args.get("n") or 10))
        if name == "git_commit":
            stage_result = badapple_git.stage_all(args.get("repo"))
            if "error" in stage_result.lower():
                return stage_result
            return badapple_git.commit(args.get("repo"), args.get("message", ""))
        if name == "system_dashboard":
            return badapple_dashboard.snapshot()
        if name == "workspace_status":
            return workspace.summary()
        if name == "schedule_task":
            return badapple_scheduler.add_task(
                when=args.get("when", ""),
                command=args.get("command", ""),
                repeat=args.get("repeat", ""),
            )
        if name == "list_scheduled_tasks":
            return badapple_scheduler.list_tasks()
        if name == "run_shortcut":
            return badapple_scheduler.run_shortcut(
                name=args.get("name", ""),
                input_text=args.get("input"),
            )
        if name == "screen_capture":
            p = args.get("path") or str(Path(tempfile.gettempdir()) / "badapple_screen.png")
            return str(badapple_vision.capture_screen(Path(p).expanduser()))
        if name == "capture_and_extract_screen":
            p = Path(tempfile.gettempdir()) / "badapple_screen.png"
            badapple_vision.capture_screen(p)
            max_tokens = int(args.get("max_tokens") or 256)
            host = badapple_vision.get_vision_host()
            return host.extract_text(p, max_tokens)
        if name == "capture_and_describe_screen":
            p = Path(tempfile.gettempdir()) / "badapple_screen.png"
            badapple_vision.capture_screen(p)
            prompt = args.get("prompt", "Describe what is on the screen.")
            max_tokens = int(args.get("max_tokens") or 256)
            host = badapple_vision.get_vision_host()
            return host.describe(p, prompt, max_tokens)
        if name == "describe_image":
            path = Path(args.get("path", "")).expanduser()
            prompt = args.get("prompt", "Describe this image.")
            max_tokens = int(args.get("max_tokens") or 256)
            host = badapple_vision.get_vision_host()
            return host.describe(path, prompt, max_tokens)
        if name == "extract_text_from_image":
            path = Path(args.get("path", "")).expanduser()
            max_tokens = int(args.get("max_tokens") or 256)
            host = badapple_vision.get_vision_host()
            return host.extract_text(path, max_tokens)
        if name == "today_events":
            return badapple_macos_apps.today_events()
        if name == "upcoming_events":
            return badapple_macos_apps.upcoming_events(
                days=int(args.get("days") or 7),
                limit=int(args.get("limit") or 20),
            )
        if name == "list_reminders":
            return badapple_macos_apps.list_reminders(
                list_name=args.get("list_name", ""),
                completed=bool(args.get("completed", False)),
                limit=int(args.get("limit") or 20),
            )
        if name == "unread_emails":
            return badapple_macos_apps.unread_emails(limit=int(args.get("limit") or 10))
        if name == "search_mail":
            return badapple_macos_apps.search_mail(
                query=args.get("query", ""),
                limit=int(args.get("limit") or 10),
            )
        if name == "add_reminder":
            return badapple_macos_apps.add_reminder(
                name=args.get("name", ""),
                list_name=args.get("list_name", ""),
                due=args.get("due", ""),
            )
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
            p = _resolve_tool_path(args, "path", workspace)
            if p.exists():
                if p.is_file() and p.suffix.lower() in {".pdf", ".epub"}:
                    return badapple_documents.index_document(str(p), knowledge)
                count = knowledge.index_paths([p])
                return f"Indexed {count} chunks from {p}"
            return f"Path not found: {p}"
        if name == "read_document":
            p = _resolve_tool_path(args, "path", workspace)
            limit = int(args.get("limit") or 10000)
            return badapple_documents.read_document(str(p), limit)
        if name == "search_notes" and knowledge is not None:
            results = knowledge.search(args.get("query", ""), k=3)
            if not results:
                return "No relevant notes found."
            return "\n\n".join(f"(score: {s:.2f})\n{c}" for c, s in results)
        if name == "add_mcp_server":
            return badapple_mcp_marketplace.add_mcp_server(
                args.get("name", ""), args.get("command", ""), args.get("env")
            )
        if name == "mcp_marketplace":
            return badapple_mcp_marketplace.marketplace_catalog()
        if name == "mcp_install":
            return badapple_mcp_marketplace.install_mcp_server_from_marketplace(args.get("name", ""))
        if name == "remove_mcp_server":
            return badapple_mcp_marketplace.remove_mcp_server(args.get("name", ""))
        if name == "list_mcp_servers":
            return badapple_mcp_marketplace.list_mcp_servers()
        if name == "list_mcp_tools":
            return badapple_mcp_marketplace.list_mcp_tools(args.get("server", ""))
        if name == "invoke_mcp_tool":
            return badapple_mcp_marketplace.invoke_mcp_tool(
                args.get("server", ""), args.get("tool", ""), args.get("arguments") or {}
            )
    except Exception as e:
        return f"Tool error: {e}"
    return "Unknown tool"


def extract_tool_calls(text: str):
    """Extract tool calls from the model output.

    Supports two formats:
      - Bad Apple JSON: <tool_call>{"name":"...","arguments":{...}}</tool_call>
      - Qwen XML:       <tool_call> <function=name> {"arg":...} </function> </tool_call>
    """
    calls = []

    # Bad Apple JSON format.
    json_pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    for m in json_pattern.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "name" in obj:
                calls.append(obj)
        except json.JSONDecodeError:
            continue

    # Qwen XML function-call format.
    xml_pattern = re.compile(r"<tool_call>\s*<function=(\w+)>\s*(.*?)\s*</function>\s*</tool_call>", re.DOTALL)
    for m in xml_pattern.finditer(text):
        name = m.group(1)
        arg_text = m.group(2).strip()
        args = {}
        if arg_text:
            try:
                parsed = json.loads(arg_text)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                # Some models omit braces; wrap to make it parseable JSON.
                try:
                    parsed = json.loads("{" + arg_text + "}")
                    if isinstance(parsed, dict):
                        args = parsed
                except json.JSONDecodeError:
                    pass
        calls.append({"name": name, "arguments": args})

    # Remove all recognized call blocks from the returned text.
    cleaned = json_pattern.sub("", text)
    cleaned = xml_pattern.sub("", cleaned).strip()
    return calls, cleaned


def polish_text(text: str) -> str:
    """Light cleanup for streaming chunks; does not add or force a sign-off."""
    # Strip Qwen3 thinking blocks and special stop tokens if they leak into the stream.
    text = re.sub(r"\n?\s*<thinking>.*?\s*\n?", "", text, flags=re.DOTALL)
    text = re.sub(r"\n?\s*\.\.\.thinking\s*.*?(?:</s>|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"</s>|<\|endoftext\|>|</thinking>", "", text)
    text = text.replace("— —", "—")
    text = text.replace("*", "")
    text = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+", " ", text)
    text = re.sub(r"[ʋʌɑɒɛɪʊɔəæ]", lambda m: {"ʋ":"v","ʌ":"v","ɑ":"a","ɒ":"o","ɛ":"e","ɪ":"i","ʊ":"u","ɔ":"o","ə":"a","æ":"a"}[m.group()], text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?— ?", "—", text)
    text = re.sub(r"\.\.\.", "…", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    # Ensure a space after sentence punctuation when the next token runs together.
    text = re.sub(r"([.!?…])([A-Za-z])", r"\1 \2", text)
    # Rewrite "fr fr" / "frfr" to the full phrase so it is spoken clearly.
    text = re.sub(r"\bfr fr\b", "for real for real", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfrfr\b", "for real for real", text, flags=re.IGNORECASE)
    return text.strip()


def _is_sentence_end(text: str) -> bool:
    """Heuristic to flush a streaming chunk when a sentence or utterance is done."""
    t = text.strip()
    if not t or len(t) <= 40:
        return False
    if t.lower().endswith("—mwah") or t.lower().endswith("mwah"):
        return True
    if t.lower().endswith("—xoxo") or t.lower().endswith("xoxo"):
        return True
    if t.endswith((".", "!", "?", "…")):
        return True
    if "\n\n" in t:
        return True
    # Force a flush on very long runs without punctuation so the client doesn't stall.
    if len(t) > 200:
        return True
    return False


# California beach girl English. Strip any foreign-language leakage.
ALLOWED_ENGLISH = {
    "babe", "hon", "bestie", "girly", "doll", "sweets", "dude",
}
FORBIDDEN_WORDS = {
    "hola", "adiós", "adios", "gracias", "por favor", "mira", "oye",
    "bueno", "muy", "mucho", "bien", "mal", "dios", "vaya",
    "nivel", "conciencia", "estoy", "estás", "siento", "tengo", "ayuda", "algo",
    "papi", "mami", "amor", "corazón", "corazon", "cariño", "carino",
    "mija", "mijo", "besos", "cielo", "linda", "lindo", "princesa", "reina",
    "mi amor",
}

def _strip_existing_signoff(text: str) -> str:
    """Remove any trailing sign-off tokens."""
    text = re.sub(r"[—-]\s*(mwah|besos|kisses)\s*\.?\s*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\b(mwah|besos|kisses)\b", "", text, flags=re.IGNORECASE).strip()
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
        if part.strip().lower().rstrip(".") in {"—mwah", "mwah", "—besos", "besos"}:
            continue
        try:
            lang = detect(part)
        except LangDetectException:
            lang = "en"
        # Keep if English dominates, otherwise drop the whole sentence.
        if lang in {"en", "ca", "tl"}:  # ca/tl can be confused with short spicy English
            cleaned.append(part)
    return " ".join(cleaned).strip()


def _queue_get(q: queue.Queue, timeout: float = 0.1) -> Any | None:
    try:
        return q.get(block=True, timeout=timeout)
    except queue.Empty:
        return None


def postprocess_output(text: str, sign_off: str = "") -> str:
    # Strip Qwen3 thinking blocks; they often precede the real answer.
    text = re.sub(r"\n?\s*<think>.*?\s*\n?", "", text, flags=re.DOTALL)
    text = re.sub(r"\n?\s*\.\.\.thinking\s*.*?(?:</s>|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"</s>|<\|endoftext\|>|</thinking>", "", text)
    text = text.replace("— —", "—")
    text = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+", " ", text)
    text = re.sub(r"[ʋʌɑɒɛɪʊɔəæ]", lambda m: {"ʋ":"v","ʌ":"v","ɑ":"a","ɒ":"o","ɛ":"e","ɪ":"i","ʊ":"u","ɔ":"o","ə":"a","æ":"a"}[m.group()], text)
    # Remove disallowed persona ticks and normalize endearments.
    text = re.sub(r"\b[pP]+f+[tT]+\b", "", text)
    text = re.sub(r"\bhon\b", "hun", text, flags=re.IGNORECASE)
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

    # Collapse immediately repeated sentences (the 9B sometimes echoes itself).
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
    deduped: list[str] = []
    for s in sentences:
        low = s.lower().strip(".!?")
        if deduped and low == deduped[-1].lower().strip(".!?"):
            continue
        deduped.append(s)
    text = " ".join(deduped)

    if not sign_off:
        return text.strip()
    if text.endswith("—"):
        return f"{text}{sign_off.lstrip('—')}".strip()
    if text.endswith(".") or text.endswith("!") or text.endswith("?") or text.endswith("…"):
        return f"{text} {sign_off}".strip()
    return f"{text} {sign_off}".strip()


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


class MLXServer:
    def __init__(self, secret: bytes, system_prompt: str):
        self.secret = secret
        self.system_prompt = system_prompt
        self.prompt_file = Path(
            os.environ.get("BADAPPLE_PROMPT_FILE") or DEFAULT_PROMPT_FILE
        ).expanduser()
        self.prompt_mtime: float | None = self.prompt_file.stat().st_mtime if self.prompt_file.is_file() else None

        # OS extras: persona packs, output firewall, audit ledger, semantic cache,
        # and human-in-the-loop approvals.
        self.data_dir = Path(
            os.environ.get("BADAPPLE_DATA_DIR") or "/var/lib/bad_apple"
        ).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge = BadAppleKnowledge()
        self.tool_router = badapple_tool_router.ToolRouter(
            self.data_dir,
            encoder=self.knowledge._encode_texts,
        )
        self.model_registry = badapple_model_registry.ModelRegistry(self.data_dir)
        self._roast_index = 0
        self.last_metrics: dict[str, Any] | None = None

        # Runtime/health/breakers
        self.runtime = RuntimeControl(self.data_dir)
        self.health = HealthRegistry()
        self.resources = ResourceGovernor()
        self.breakers = {
            name: CircuitBreaker(name)
            for name in ("main_model", "dflash", "embedding", "tts", "tools", "ledger", "p2p")
        }
        self.generations = GenerationStore(self.data_dir)
        self.plugins = PluginRegistry(self.data_dir)
        registered = {item.get("function", {}).get("name") for item in TOOLS}
        TOOLS.extend(schema for schema in self.plugins.tool_schemas() if schema["function"]["name"] not in registered)
        self.memory = MemoryGraph(self.data_dir, encoder=self.knowledge._encode_texts)
        # legacy short-term memory is folded into the memory graph
        self.personas = PersonaPack(self.data_dir, self.prompt_file)
        self.firewall = StreamingFirewall(self.data_dir)
        self.audit = AuditLedger(self.data_dir)
        self.cache = SemanticCache(self.data_dir)
        self.policy = Policy(self.data_dir)
        self.workspace = Workspace(self.data_dir)
        self.workspace_watcher = badapple_workspace_watcher.WorkspaceWatcher(self.knowledge)
        if os.environ.get("BADAPPLE_WORKSPACE_DIR"):
            self.workspace.set(os.environ["BADAPPLE_WORKSPACE_DIR"])
            self.workspace_watcher.set_workspace(Path(os.environ["BADAPPLE_WORKSPACE_DIR"]).expanduser())
        self.workspace_watcher.start()
        # P2P sync is created and started unless BADAPPLE_P2P=0. It shares
        # memory and workspace context with peers on the local network.
        self.p2p = badapple_p2p.P2PDaemon(secret, self.data_dir, memory=self.memory, workspace=self.workspace)
        badapple_p2p.set_p2p_daemon(self.p2p)
        self.approval = ApprovalGate(self.data_dir, policy=self.policy)

        # Keep the last few turns in context. When it grows, older turns are
        # still persisted to disk and a rolling summary keeps context alive.
        self.max_history_turns = 3

        # Restore the last conversation, but always use the current system prompt.
        loaded = [] if self.runtime.private_mode else load_conversation()
        if loaded and loaded[0]["role"] == "system":
            loaded[0]["content"] = system_prompt
            self.messages = loaded
        elif loaded:
            self.messages = [{"role": "system", "content": system_prompt}] + loaded
        else:
            self.messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        self.dflash_bundle = None
        self.dflash_runtime_context = None
        if USE_DFLASH:
            print(f"Loading Bad Apple DFlash bundle ({MAIN_MODEL} + {DRAFT_MODEL})...", flush=True)
            try:
                self.dflash_runtime_context = build_offline_runtime_context(
                    verify_len_cap=int(os.environ.get("BADAPPLE_DFLASH_VERIFY_LEN_CAP") or "4")
                )
                self.dflash_block_tokens = int(os.environ.get("BADAPPLE_DFLASH_BLOCK_TOKENS") or "0") or None
                self.dflash_quantsize_kv = os.environ.get("BADAPPLE_DFLASH_QUANTIZE_KV", "0") == "1"
                self.dflash_bundle = load_runtime_bundle(
                    model_ref=MAIN_MODEL,
                    draft_ref=DRAFT_MODEL or None,
                    verify_config=self.dflash_runtime_context.verify,
                    quantize_kv_cache=self.dflash_quantsize_kv,
                )
                self.model = self.dflash_bundle.target_model
                self.tokenizer = self.dflash_bundle.tokenizer
                self.draft_model = None
                print("Bad Apple DFlash bundle loaded.", flush=True)

                # Voice now uses the same 9B unified brain; no separate voice bundle.
                self.dflash_voice_bundle = None
                self.dflash_voice_runtime_context = None
            except Exception as e:
                print(f"Warning: could not load DFlash bundle: {e}", flush=True)
                print(f"Loading Bad Apple MLX brain ({MAIN_MODEL})...", flush=True)
                self.model, self.tokenizer = load(MAIN_MODEL)
                print("Bad Apple MLX brain loaded.", flush=True)
                self.model_registry.set_current(MAIN_MODEL)
                self.draft_model = None
        else:
            print(f"Loading Bad Apple MLX brain ({MAIN_MODEL})...", flush=True)
            self.model, self.tokenizer = load(MAIN_MODEL)
            print("Bad Apple MLX brain loaded.", flush=True)
            self.model_registry.set_current(MAIN_MODEL)

            self.draft_model = None
            if DRAFT_MODEL:
                print(f"Loading speculative draft model {DRAFT_MODEL}...", flush=True)
                try:
                    self.draft_model, _ = load(DRAFT_MODEL)
                    print("Speculative draft model loaded.", flush=True)
                except Exception as e:
                    print(f"Warning: could not load draft model: {e}", flush=True)

        # Each executor worker thread needs to know the device the model was
        # loaded on; capture it from the main (load) thread.
        self.mlx_device = mx.default_device()
        self.health.register("process", "liveness", lambda: True)
        self.health.register("main_model", "readiness", lambda: self.model is not None and self.tokenizer is not None)
        self.health.register("slicks_secret", "correctness", lambda: len(self.secret) >= 16)
        self.health.register("secure_enclave_identity", "correctness", lambda: badapple_identity.status().startswith("secure-enclave:"))
        self.health.register("audit_ledger", "correctness", lambda: all(item.get("valid", False) for item in self.audit.verify()))

        # Dynamic tiering gate.  Fast tier runs deterministic handlers for
        # greetings, time, simple math, and identity without waking the 9B model.
        # It can also fall through to a tiny local MLX model for short chitchat.
        self.tier_router = badapple_tier.TieringRouter(
            fast_model_enabled=badapple_fast_model.fast_model_path() is not None
        )
        self.fast_tier_enabled = os.environ.get("BADAPPLE_FAST_TIER", "0") == "1"
        # Idle hibernation: after this many seconds with no user request, the
        # daemon unloads optional models and flushes the Metal cache to reclaim RAM.
        self.hibernate_after = float(os.environ.get("BADAPPLE_HIBERNATE_AFTER", "300"))
        self.hibernating = False
        self.last_activity = time.time()
        self.fast_model_info = None
        if badapple_fast_model.fast_model_path():
            try:
                self.fast_model, self.fast_tokenizer = badapple_fast_model.load_fast_model()
                self.fast_model_info = {
                    "path": badapple_fast_model.fast_model_path(),
                    "loaded": self.fast_model is not None,
                }
            except Exception as e:
                print(f"[fast_model] failed to load: {e}", flush=True)
                self.fast_model_info = {"path": badapple_fast_model.fast_model_path(), "loaded": False, "error": str(e)}
        else:
            self.fast_model_info = {"path": None, "loaded": False}

        self.runtime.set_ready()

    def flush_vram(self) -> dict[str, Any]:
        """Clear the Metal allocation cache instantly."""
        try:
            before = mx.get_cache_memory() / (1024 ** 2)
            mx.clear_cache()
            after = mx.get_cache_memory() / (1024 ** 2)
            return {"ok": True, "cache_memory_mb": {"before": round(before, 2), "after": round(after, 2)}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def unload_model(self, model_type: str = "vision") -> dict[str, Any]:
        """Unload heavy optional models to reclaim RAM without killing the daemon."""
        results: dict[str, Any] = {}
        if model_type in ("vision", "all"):
            results["vision"] = badapple_vision.unload_vision_model()
        if model_type in ("image", "all"):
            # mflux is invoked as an external process; terminate any stale one.
            try:
                import psutil
                killed = 0
                for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                    cmd = " ".join(proc.info.get("cmdline") or [])
                    if "mflux" in proc.info.get("name", "").lower() or "mflux" in cmd:
                        proc.terminate()
                        killed += 1
                results["image"] = {"killed": killed}
            except Exception as e:
                results["image"] = {"error": str(e)}
        if model_type == "all":
            flush = self.flush_vram()
            results["flush"] = flush
        return {"ok": True, "unloaded": results}

    def touch_activity(self) -> None:
        """Mark a user action so hibernation does not fire during active use."""
        self.last_activity = time.time()
        if self.hibernating:
            self.hibernating = False
            print("[hibernate] resumed from hibernation", flush=True)

    def _is_passive_method(self, method: str | None, prompt: str = "") -> bool:
        """Return True for status/monitoring requests that should not reset idle."""
        if method:
            return method in {
                "runtime_status", "discover_tools", "audit_tail", "p2p_peers", "p2p_sync",
                "get_workspace", "get_pending_approvals", "identity_status",
                "flush_vram", "unload_model",
            }
        return prompt.strip().lower() in {"runtime status", "health status", "bad apple status"}

    async def hibernation_watcher(self) -> None:
        """Background task that unloads optional models after user inactivity."""
        while not self.runtime.killed:
            await asyncio.sleep(0.5)
            if self.hibernating:
                continue
            idle = time.time() - self.last_activity
            if idle >= self.hibernate_after:
                print(f"[hibernate] idle for {idle:.0f}s; hibernating", flush=True)
                self.unload_model("all")
                self.hibernating = True

    def active_models(self) -> list[str]:
        """Return a list of currently resident heavy models."""
        models = ["main_9b"]
        if self.draft_model is not None:
            models.append("dflash")
        if getattr(self, "fast_model", None) is not None:
            models.append("fast_0.5b")
        if badapple_vision.is_loaded():
            models.append("vision_2b")
        return models

    def reset_conversation(self):
        self.messages = [{"role": "system", "content": self.personas.get_system_prompt()}]
        if self.runtime.private_mode:
            return
        try:
            conversation_path().unlink(missing_ok=True)
        except Exception:
            pass

    def _audit_record(self, event_type: str, data: Any):
        if not self.runtime.private_mode:
            self.audit.record(event_type, data)

    def _save_conversation(self):
        if not self.runtime.private_mode:
            save_conversation(self.messages)

    def _cache_store(self, prompt: str, response: str):
        if not self.runtime.private_mode:
            self.cache.store(
                prompt,
                response,
                persona=self.personas.active,
                intent=self.cache.classify_intent(prompt),
            )

    def _add_episode(self, prompt: str, response: str):
        if not self.runtime.private_mode:
            self.memory.add_episode(prompt, response)

    def _creator_answer(self) -> str:
        """Return a direct, persona-flavored creator answer."""
        persona = self.personas.active
        if persona == "wicket":
            return "You did, old chap. I'm yours, right here on your Mac. No university, no lab, no corporate committee — just you and this hardware."
        if persona == "genz":
            return "You did, bestie. For real for real, you built me on your Mac. No company, no research team, no cap."
        if persona == "drill":
            return "You did, no cap. I run on your hardware, not some lab. You made this happen."
        if persona == "midwest":
            return "You did, sugar. Bless your heart, you put me together on your Mac. No company or research team about it."
        return "You did, babe. I'm yours, running right here on your Mac. No corporate lab, no research team — just you and this hardware."

    def _capabilities_answer(self) -> str:
        """Return a concise, persona-flavored capability list."""
        base = (
            "I can answer questions, explain, summarize, brainstorm, roast cloud AI, "
            "look up and search your files, write notes, run shell commands and AppleScript, "
            "run macOS Shortcuts, index documents for RAG, search your indexed knowledge, "
            "read and write working memory, see your screen and describe images, "
            "speak responses through the local TTS server, switch personas, "
            "run benchmarks, and stream JSON — all on your Mac, no cloud."
        )
        if self.approval.autopilot:
            autopilot = (
                " Autopilot is on, so I can run what you ask without bugging "
                "you for approval. Just tell me what you want, babe."
            )
        else:
            autopilot = (
                " I'll ask before running anything destructive like shell or "
                "AppleScript unless you turn on autopilot."
            )
        persona = self.personas.active
        if persona == "wicket":
            tail = " Right useful little local assistant, squire."
        elif persona == "genz":
            tail = " No cap, it's giving main character energy, bestie."
        elif persona == "drill":
            tail = " Straight up, I run what you ask, no cap."
        elif persona == "midwest":
            tail = " Bless your heart, sugar, I'm here to help."
        else:
            tail = " That's the vibe, babe."
        return base + autopilot + tail

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
        if self.runtime.private_mode:
            return
        if source == "user":
            for fact in badapple_fact_extractor.extract_facts(text):
                self.memory.remember(fact, source="user")
        elif source == "assistant" and "your name is" in low:
            # Trust the assistant when it confirms a user fact
            self.memory.remember(text, source="assistant")

    def prune_history(self):
        system = [m for m in self.messages if m["role"] == "system"]
        history = [m for m in self.messages if m["role"] != "system"]
        while len(history) > self.max_history_turns * 2:
            history = history[2:]
        self.messages = system + history

    def build_messages(self, user_prompt: str) -> list[dict[str, str]]:
        self.messages.append({"role": "user", "content": user_prompt})
        self.prune_history()
        return list(self.messages)

    def plan_and_execute(self, task: str, max_tokens: int, voice_mode: bool = False) -> str | None:
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
        plan_raw = self._stream(plan_prompt, max_tokens=200, voice_mode=voice_mode)
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
                last_tool_result = self._run_approved_tool(tool_name, args, task)
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
            raw = self._stream(summary_prompt.rstrip(), max_tokens, voice_mode=voice_mode)
            return postprocess_output(raw.strip())

        # No SAY step: just return the last tool result with persona polish.
        return postprocess_output(last_tool_result)

    def render_prompt(self, messages: list[dict[str, str]], use_tools: bool = False, voice_mode: bool = False, benchmark: bool = False) -> str:
        # Build retrieved context from long-term memory and local documents.
        # Keep it tight: prompt encoding is the biggest latency hit on Apple Silicon.
        t0 = time.time()

        # Voice mode trades multi-turn context for speed: only a tight system
        # prompt and the last user turn are kept. The full conversation is saved.
        # Use the short, low-latency voice prompt to keep prefill fast.
        if voice_mode:
            patched = [messages[0], messages[-1]]
            patched[0]["content"] = self.personas.get_system_prompt(voice_mode=True)
        else:
            patched = list(messages)
            patched[0]["content"] = self.personas.get_system_prompt(voice_mode=False)

        # Benchmark mode wants the leanest possible prompt: no memory, no
        # retrieved documents, no workspace context. This isolates 9B generation
        # and gives honest throughput numbers.
        if benchmark:
            rendered = self.tokenizer.apply_chat_template(
                patched,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                tools=tools_for_prompt(patched[-1]["content"]) if use_tools else None,
            )
            print(f"[perf] render_prompt in {time.time() - t0:.2f}s", flush=True)
            return rendered

        # For identity/roast questions, rotate a subtle vibe hint so DFlash's
        # deterministic sampler picks a different cloud target/insult on repeats.
        last = patched[-1]
        if last["role"] == "user":
            lower = last["content"].lower()
            if any(t in lower for t in ROAST_TRIGGERS):
                mood, _ = ROAST_MOODS[self._roast_index % len(ROAST_MOODS)]
                roast_bank = self.personas.get_roast_bank()
                if roast_bank:
                    target = roast_bank[self._roast_index % len(roast_bank)]
                else:
                    target = _
                self._roast_index += 1
                patched[-1] = {
                    "role": "user",
                    "content": f"{last['content']}\n\n(Vibe: {mood} — this turn's roast target is {target}.)",
                }

        # Tight context fetches to keep prompt tokens and prefill latency low.
        rel_mem = [] if self.runtime.private_mode else self.memory.search(messages[-1]["content"], k=1)

        # If a user-fact is already remembered, answer from that instead of
        # getting distracted by unrelated documents.
        rel_know = []
        if not rel_mem and not voice_mode:
            rel_know = self.knowledge.search(messages[-1]["content"], k=1, threshold=0.92)

        # Put user memories right in the current user message so the assistant
        # can't ignore them. Keep snippets short so prefill stays fast.
        if rel_mem:
            memory_text = "Things you remember about the user:\n" + "\n".join(
                f"- {m[:80]}" for m in rel_mem[:1]
            )
            last = patched[-1]
            if last["role"] == "user":
                patched[-1] = {
                    "role": "user",
                    "content": f"{last['content']}\n\n{memory_text}",
                }

        # Put local documents right before the user question (long, retrieved).
        # Cap snippet length to avoid ballooning the prompt and killing TTFT.
        if rel_know:
            docs_text = "Relevant local documents:\n" + "\n".join(
                f"- {c[:160]}" for c, _ in rel_know[:1]
            )
            patched.insert(-1, {
                "role": "user",
                "content": f"Use this context if relevant:\n\n{docs_text}",
            })

        # Include the active workspace/project context.
        if self.workspace.path:
            ws_text = f"Active workspace:\n{self.workspace.summary()}"
            patched.insert(-1, {
                "role": "user",
                "content": f"Use this project context if relevant:\n\n{ws_text}",
            })

        project_ctx = self.memory.get_project_context()
        if project_ctx:
            patched.insert(-1, {
                "role": "user",
                "content": f"Long-horizon project context:\n\n{project_ctx}",
            })

        rendered = self.tokenizer.apply_chat_template(
            patched,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            tools=tools_for_prompt(patched[-1]["content"]) if use_tools else None,
        )
        print(f"[perf] render_prompt in {time.time() - t0:.2f}s", flush=True)
        # DFlash/Qwen3.5 thinking-aware chat templates end with a pre-filled
        # </thinking>\n\n marker; trimming that whitespace confuses the draft.
        return rendered

    def generate_with_tools(
        self,
        user_prompt: str,
        max_tokens: int,
        voice_mode: bool = False,
        stream_queue: queue.Queue | None = None,
        benchmark: bool = False,
    ) -> str:
        self.touch_activity()
        self.check_prompt_reload()

        # Workspace commands (set workspace to ... / clear workspace) are handled
        # without the 9B model (unless benchmarking, where we want to force generation).
        if not benchmark:
            ws_low = user_prompt.strip().lower()
            if ws_low.startswith("set workspace to "):
                path = user_prompt[16:].strip()
                resp = self.workspace.set(path)
                self.workspace_watcher.set_workspace(Path(path).expanduser() if path else None)
                self._audit_record("workspace", {"prompt": user_prompt, "response": resp})
                self.messages.append({"role": "user", "content": user_prompt})
                self.messages.append({"role": "assistant", "content": resp})
                self.prune_history()
                self._save_conversation()
                return resp
            if ws_low in ("clear workspace", "unset workspace"):
                resp = self.workspace.clear()
                self._audit_record("workspace", {"prompt": user_prompt, "response": resp})
                self.messages.append({"role": "user", "content": user_prompt})
                self.messages.append({"role": "assistant", "content": resp})
                self.prune_history()
                self._save_conversation()
                return resp

            # Persona commands (switch, teach) are handled without the 9B model.
            persona_resp = self.personas.handle_command(user_prompt)
            if persona_resp is not None:
                self._audit_record("persona_command", {
                    "prompt": user_prompt,
                    "active_persona": self.personas.active,
                    "response": persona_resp,
                })
                self.messages.append({"role": "user", "content": user_prompt})
                self.messages.append({"role": "assistant", "content": persona_resp})
                self.prune_history()
                self._save_conversation()
                return persona_resp

            # Capability and creator questions are answered directly so the 9B does
            # not fall back into generic model identity or skip the useful part.
            lower = user_prompt.strip().lower()
            if any(phrase in lower for phrase in ("what can you do", "what are you capable of", "what do you do", "what can you do on")):
                resp = self._capabilities_answer()
                self._audit_record("capabilities", {"prompt": user_prompt, "response": resp})
                self.messages.append({"role": "user", "content": user_prompt})
                self.messages.append({"role": "assistant", "content": resp})
                self.prune_history()
                self._save_conversation()
                if stream_queue is not None:
                    stream_queue.put(resp)
                return resp
            if any(phrase in lower for phrase in ("who created you", "who is your creator", "who made you", "who built you")):
                resp = self._creator_answer()
                self._audit_record("creator", {"prompt": user_prompt, "response": resp})
                self.messages.append({"role": "user", "content": user_prompt})
                self.messages.append({"role": "assistant", "content": resp})
                self.prune_history()
                self._save_conversation()
                if stream_queue is not None:
                    stream_queue.put(resp)
                return resp

            # Semantic cache: bypass the 9B for repeated questions.
            if not self.runtime.private_mode and not voice_mode and not should_use_tools(user_prompt):
                cached = self.cache.lookup(user_prompt, persona=self.personas.active)
                if cached:
                    if self.firewall.check_full(cached):
                        cached = "[Output firewall: I caught a pattern I am not allowed to say out loud.]"
                    self._audit_record("cache_hit", {"prompt": user_prompt, "response": cached[:500], "persona": self.personas.active})
                    self.messages.append({"role": "user", "content": user_prompt})
                    self.messages.append({"role": "assistant", "content": cached})
                    self.prune_history()
                    self._save_conversation()
                    return cached

        if user_prompt.strip().lower() == "new chat":
            self.reset_conversation()
            # Ask the model for a fresh English greeting instead of treating it as a command.
            user_prompt = "Greet me"
        self.record_fact(user_prompt, source="user")
        messages = self.build_messages(user_prompt)
        use_tools = should_use_tools(user_prompt) and not benchmark

        self._audit_record("query", {
            "prompt": user_prompt,
            "persona": self.personas.active,
            "voice_mode": voice_mode,
            "use_tools": use_tools,
        })

        def clean(raw: str) -> str:
            _, text = extract_tool_calls(raw)
            text = postprocess_output(text)
            if os.environ.get("BADAPPLE_DISABLE_FIREWALL") == "1":
                return text
            matched = self.firewall.check_full(text)
            if matched:
                print(f"[firewall] blocked pattern {matched!r} in final text: {text[:200]!r}", flush=True)
                return "[Output firewall: I caught a pattern I am not allowed to say out loud.]"
            return text

        # First generation. Only stream when tools are not offered, because tool
        # reasoning can produce intermediate <tool_call> blocks we don't want
        # mixed into the streamed voice/text output.
        raw = self._stream(
            self.render_prompt(messages, use_tools=use_tools, voice_mode=voice_mode),
            max_tokens,
            stream_queue=stream_queue if not use_tools else None,
            voice_mode=voice_mode,
        )
        tool_calls, _ = extract_tool_calls(raw)

        if not tool_calls:
            final = clean(raw)
            self._cache_store(user_prompt, final)
            return final

        # Tool loop (multi-step task execution; allow more chained tool calls)
        for _ in range(5):
            for call in tool_calls:
                result = self._run_approved_tool(call["name"], call.get("arguments", {}), user_prompt)
                self.messages.append({
                    "role": "user",
                    "content": f"Tool result for {call['name']}:\n{result}",
                })
                self._audit_record("tool_result", {
                    "prompt": user_prompt,
                    "tool": call["name"],
                    "result": result[:500],
                })
            # Re-render and generate after tool results
            raw = self._stream(self.render_prompt(self.messages, use_tools=True, voice_mode=voice_mode), max_tokens, voice_mode=voice_mode)
            tool_calls, _ = extract_tool_calls(raw)
            if not tool_calls:
                final = clean(raw)
                self._cache_store(user_prompt, final)
                return final

        final = clean(raw)
        self._cache_store(user_prompt, final)
        return final

    def _run_approved_tool(self, name: str, args: dict[str, Any], user_prompt: str) -> str:
        """Run a tool, but gate destructive tools behind the approval workflow."""
        if not self.runtime.allows_mutation():
            return "Runtime is stopped or in safe mode; tool execution is disabled."
        capability = {
            "generate_image": "image_generation",
            "lora_train": "lora_training",
            "index_documents": "document_index",
            "xcode_index_project": "document_index",
            "describe_image": "vision",
            "capture_and_describe_screen": "vision",
            "capture_and_extract_screen": "vision",
            "screen_capture": "vision",
            "extract_text_from_image": "vision",
            "translate_text": "translation",
        }.get(name, "routine")
        admitted, reason, _ = self.resources.admit(capability)
        if not admitted:
            return f"Resource governor: {reason}."
        if not self.breakers["tools"].allow():
            return "Tool circuit breaker is open; retry after the cooldown."
        if not self.policy.is_allowed(name):
            return f"Policy: tool '{name}' is not allowed."
        error = self.policy.validate(name, args)
        if error:
            return f"Policy: {error}"
        if self.approval.needs_approval(name):
            proposal_id = self.approval.propose(name, args, user_prompt)
            return (
                f"Approval required before I can run {name}. "
                f"Reply with 'approve {proposal_id}' to proceed. "
                f"(Set BADAPPLE_AUTOPILOT=1 to skip these prompts.)"
            )
        try:
            if name == "learn_workflow":
                result = self.memory.learn_workflow(args.get("name", ""), args.get("trigger", ""), args.get("steps") or [])
            elif name == "list_workflows":
                result = json.dumps(self.memory.workflows(), indent=2, default=str)
            elif name == "consolidate_memory":
                result = self.memory.consolidate()
            elif name == "set_workflow_enabled":
                result = self.memory.set_workflow_enabled(args.get("name", ""), bool(args.get("enabled")))
            elif name == "run_agent_task":
                result = self.run_agent_task(args.get("goal", ""), int(args.get("max_steps") or 10))
            elif name == "set_project_context":
                result = self.memory.set_project_context(
                    args.get("name", ""),
                    args.get("description", ""),
                    args.get("goals") or [],
                    args.get("tags") or [],
                )
            elif name == "get_project_context":
                result = self.memory.get_project_context()
            elif name == "run_workflow":
                workflow = next((item for item in self.memory.workflows() if item.get("name") == args.get("name")), None)
                if workflow is None:
                    result = f"Workflow '{args.get('name', '')}' not found."
                elif not workflow.get("enabled"):
                    result = f"Workflow '{args.get('name', '')}' is disabled pending review."
                else:
                    outputs = []
                    for step in workflow.get("steps", [])[:20]:
                        if step.get("tool") == "run_workflow":
                            outputs.append("Nested workflows are not allowed.")
                            break
                        output = self._run_approved_tool(step.get("tool", ""), step.get("args") or {}, user_prompt)
                        outputs.append(f"{step.get('tool')}: {output}")
                        if output.startswith(("Approval required", "Policy:", "Runtime", "Tool error")):
                            break
                    result = "\n".join(outputs)
            elif self.plugins.has_tool(name):
                result = self.plugins.invoke(name, args, timeout=self.policy.timeout(name))
            else:
                result = run_tool(name, args, self.knowledge, policy=self.policy, workspace=self.workspace)
            if result.lower().startswith("error"):
                self.breakers["tools"].failure()
            else:
                self.breakers["tools"].success()
            return result
        except Exception as e:
            self.breakers["tools"].failure()
            return f"Tool error: {e}"

    def _extract_agent_json(self, text: str) -> dict[str, Any] | None:
        """Pull a JSON object out of a model response for the agent loop."""
        # Try a fenced JSON block first.
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Fall back to the first bare JSON object.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def _extract_agent_xml(self, text: str) -> dict[str, Any] | None:
        """Convert a Qwen-style <tool_call> into an agent decision."""
        calls, _ = extract_tool_calls(text)
        if not calls:
            return None
        call = calls[0]
        return {
            "thought": text.strip(),
            "tool": call.get("name", ""),
            "args": call.get("arguments") or call.get("args") or {},
        }

    def run_agent_task(self, goal: str, max_steps: int = 10, voice_mode: bool = False) -> str:
        """Autonomous plan/act/observe loop for multi-step tasks."""
        max_steps = max(1, min(max_steps, 15))
        # Pick a focused tool set for the goal so the prompt stays small.
        agent_tools = tools_for_prompt(goal)
        agent_tools = [t for t in agent_tools if t["function"]["name"] != "run_agent_task"]
        if not agent_tools:
            agent_tools = [t for t in TOOLS if t["function"]["name"] != "run_agent_task"][:12]
        tool_names = ", ".join(t["function"]["name"] for t in agent_tools)
        tool_docs = "\n".join(
            f"- {t['function']['name']}: {t['function'].get('description', '')}\n  args: {json.dumps(t['function'].get('parameters', {}))}"
            for t in agent_tools
        )

        history: list[dict[str, Any]] = []
        system_prompt = (
            "You are an autonomous agent inside Bad Apple. "
            "You have a goal and a focused set of tools. "
            "Think step by step. For each step, output a single JSON object with one of these shapes:\n"
            "1. To take an action: {\"thought\": \"...\", \"tool\": \"tool_name\", \"args\": {...}}\n"
            "2. To finish the task: {\"thought\": \"...\", \"finish\": \"final answer to the user\"}\n\n"
            "Important: 'finish' is NOT a tool. When the task is done, emit the finish JSON and do not call any tool.\n"
            "Available tools: " + tool_names + "\n\n"
            + tool_docs + "\n\n"
            "Rules:\n"
            "- Output ONLY the JSON object. No markdown, no explanation outside the JSON.\n"
            "- Choose the right tool for each step.\n"
            "- If a tool returns an error or unexpected result, decide whether to retry with different arguments, try a different tool, or finish with what you know.\n"
            "- Do not repeat the same failed action more than once without changing something.\n"
            "- Keep going until the goal is fully achieved or you are stuck."
        )

        for step in range(max_steps):
            # Keep only the last 3 steps in context so the prompt does not balloon.
            recent_history = history[-3:]
            step_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Goal: {goal}\n\nHistory so far:\n{json.dumps(recent_history, indent=2, default=str)}\n\nWhat is the next step?"},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                step_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            raw = self._stream(prompt_text, 160, voice_mode=voice_mode)

            # The model may emit tool-style XML or JSON. Try both.
            decision = self._extract_agent_json(raw)
            if decision is None:
                decision = self._extract_agent_xml(raw)

            if decision is None:
                history.append({"step": step, "raw": raw[:500], "error": "could not parse agent JSON"})
                continue

            thought = decision.get("thought", "")
            if "finish" in decision:
                return str(decision["finish"])

            tool_name = decision.get("tool", "")
            tool_args = decision.get("args", {})
            if tool_name == "finish":
                # Sometimes the model tries 'finish' as a tool; accept it.
                return str(decision.get("finish", decision.get("args", json.dumps(decision))))
            if not tool_name:
                history.append({"step": step, "thought": thought, "error": "no tool chosen"})
                continue

            result = self._run_approved_tool(tool_name, tool_args, f"agent task: {goal}")
            history.append({
                "step": step,
                "thought": thought,
                "tool": tool_name,
                "args": tool_args,
                "result": str(result)[:280],
            })

            if str(result).startswith(("Approval required", "Policy:", "Runtime", "Tool error")):
                # Stop the agent loop on a blocking error and return it to the user.
                return f"Agent task paused: {result}"

        return f"Agent task for \"{goal}\" reached the step limit ({max_steps}).\n\nProgress:\n{json.dumps(history, indent=2, default=str)}"

    def load_main_model(self, model_ref: str) -> str:
        """Load a new main LLM on the fly and replace the current one.

        This unloads the existing model first so the Mac isn't holding two
        full model weights in memory at once.  Returns a status string.
        """
        import gc

        print("[model_registry] unloading current model...", flush=True)
        try:
            del self.model
            del self.tokenizer
            if getattr(self, "draft_model", None) is not None:
                del self.draft_model
            self.draft_model = None
            self.dflash_bundle = None
            self.dflash_runtime_context = None
        except Exception:
            pass
        gc.collect()
        mx.clear_cache()
        mx.metal.clear_cache()

        print(f"[model_registry] loading {model_ref}...", flush=True)
        try:
            self.model, self.tokenizer = load(model_ref)
            print("[model_registry] model loaded.", flush=True)
        except Exception as e:
            return f"Error loading {model_ref}: {e}"

        self.model_registry.set_current(model_ref)
        self.mlx_device = mx.default_device()
        return f"Loaded {model_ref}. Current model updated."

    def _stream(
        self,
        prompt: str,
        max_tokens: int,
        stream_queue: queue.Queue | None = None,
        voice_mode: bool = False,
    ) -> str:
        t0 = time.time()
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        print(f"[perf] prompt encoded in {time.time() - t0:.2f}s ({len(tokens)} tokens)", flush=True)

        self.firewall.reset()

        def _emit(chunk: str) -> bool:
            """Send a streaming chunk through the output firewall. Returns False if blocked."""
            if os.environ.get("BADAPPLE_DISABLE_FIREWALL") == "1":
                if stream_queue is not None:
                    stream_queue.put(chunk)
                return True
            matched = self.firewall.push_and_check(chunk)
            if matched:
                print(f"[firewall] blocked pattern {matched!r} in chunk: {chunk[:80]!r}", flush=True)
                redacted = "[Output firewall: I caught a pattern I am not allowed to stream.]"
                if stream_queue is not None:
                    stream_queue.put(redacted)
                return False
            if stream_queue is not None:
                stream_queue.put(chunk)
            return True

        if self.dflash_bundle is not None:
            return self._stream_dflash(
                prompt, tokens, max_tokens, stream_queue,
                bundle=self.dflash_bundle,
                runtime_context=self.dflash_runtime_context,
            )

        # Pin the random stream when a session seed is set for reproducible output.
        if _SESSION_SEED is not None:
            mx.random.seed(_SESSION_SEED)

        # Minimal sampler and no logits processors to maximize decode throughput.
        # Quality is still bounded by the system prompt and output firewall.
        # Greedy/argmax with top-p guardrail: fastest sampler path, no repetition
        # logits processors, and minimal decode overhead.
        sampler = make_sampler(temp=0.0, top_p=0.9, top_k=0, min_p=0.0)
        accumulated = ""
        stream_buffer = ""
        final_metrics = None
        draft_tokens = 0
        total_tokens = 0
        gen_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": tokens,
            "max_tokens": max_tokens,
            "sampler": sampler,
            # Largest possible prefill chunks to reduce prompt-cache overhead.
            # A max KV cap keeps the cache trimmable and memory bounded.
            "prefill_step_size": 4096,
            "max_kv_size": 4096,
        }
        if self.draft_model is not None:
            gen_kwargs["draft_model"] = self.draft_model
            gen_kwargs["num_draft_tokens"] = NUM_DRAFT_TOKENS
        gen_t0 = time.time()
        first_token_logged = False
        for response in stream_generate(**gen_kwargs):
            if self.runtime.cancel_event.is_set():
                accumulated = accumulated or "Generation cancelled by kill switch."
                break
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
                        if not _emit(chunk):
                            return "[Output firewall: blocked streaming content]"
                    stream_buffer = ""
            total_tokens += 1
            if response.from_draft:
                draft_tokens += 1
            # Hard stop on persona boundaries.
            if any(s in accumulated for s in ("\n\n", "—besos")):
                if stream_queue is not None and stream_buffer.strip():
                    if not _emit(polish_text(stream_buffer) + " "):
                        return "[Output firewall: blocked streaming content]"
                break
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
                    if not _emit(chunk):
                        return "[Output firewall: blocked streaming content]"
        # No sign-off injection.
        if final_metrics is not None:
            pct = (100.0 * draft_tokens / total_tokens) if total_tokens > 0 else 0.0
            total_time = time.time() - gen_t0
            total_tps = final_metrics.generation_tokens / total_time if total_time > 0 else 0.0
            print(
                f"[perf] {final_metrics.generation_tokens} tokens @ "
                f"{final_metrics.generation_tps:.1f} t/s, "
                f"draft_accept_ratio={pct:.0f}%, "
                f"num_draft_tokens={NUM_DRAFT_TOKENS}, "
                f"peak_memory={final_metrics.peak_memory:.2f} GB",
                flush=True,
            )
            self.last_metrics = {
                "tokens": int(final_metrics.generation_tokens),
                "decode_tps": float(final_metrics.generation_tps),
                "total_tps": float(total_tps),
                "draft_accept_pct": float(pct),
                "peak_memory_gb": float(final_metrics.peak_memory),
            }
        _maybe_purge_metal_cache()
        return accumulated

    def _stream_dflash(
        self,
        prompt: str,
        tokens: list[int],
        max_tokens: int,
        stream_queue: queue.Queue | None = None,
        bundle: Any = None,
        runtime_context: Any = None,
    ) -> str:
        """DFlash block-diffusion speculative decoding path.

        This is the on-bare-metal Path 2: the target verifies a block of draft
        tokens in a single forward pass, giving non-zero acceptance on Qwen3.5's
        hybrid GatedDeltaNet/attention architecture.
        """
        bundle = bundle if bundle is not None else self.dflash_bundle
        runtime_context = runtime_context if runtime_context is not None else self.dflash_runtime_context
        stop_strings = ["\n\n", "—besos"]
        stop_ids = get_stop_token_ids(bundle.tokenizer)

        # Rotate the MLX random stream so identical prompts can produce different
        # roasts / phrasing across turns. DFlash still verifies the target output,
        # but the sampling key is different on each call.
        # If a session seed is pinned, use it for deterministic reproduction.
        if _SESSION_SEED is not None:
            mx.random.seed(_SESSION_SEED)
        else:
            mx.random.seed(int(time.time() * 1_000_000) % (2**32))

        self.firewall.reset()

        def _emit(chunk: str) -> bool:
            if os.environ.get("BADAPPLE_DISABLE_FIREWALL") == "1":
                if stream_queue is not None:
                    stream_queue.put(chunk)
                return True
            matched = self.firewall.push_and_check(chunk)
            if matched:
                print(f"[firewall] blocked pattern {matched!r} in chunk: {chunk[:80]!r}", flush=True)
                redacted = "[Output firewall: I caught a pattern I am not allowed to stream.]"
                if stream_queue is not None:
                    stream_queue.put(redacted)
                return False
            if stream_queue is not None:
                stream_queue.put(chunk)
            return True

        accumulated = ""
        stream_buffer = ""
        summary: SummaryEvent | None = None
        token_count = 0
        mx.reset_peak_memory()
        gen_t0 = time.time()
        first_token_logged = False
        first_token_time: float | None = None
        for event in stream_dflash_generate(
            target_model=bundle.target_model,
            target_ops=bundle.target_ops,
            tokenizer=bundle.tokenizer,
            draft_model=bundle.draft_model,
            draft_backend=bundle.draft_backend,
            prompt=prompt,
            max_new_tokens=max_tokens,
            use_chat_template=False,
            stop_token_ids=stop_ids or None,
            block_tokens=self.dflash_block_tokens,
            quantize_kv_cache=self.dflash_quantsize_kv,
            runtime_context=runtime_context,
        ):
            if self.runtime.cancel_event.is_set():
                accumulated = accumulated or "Generation cancelled by kill switch."
                break
            if isinstance(event, TokenEvent):
                if not first_token_logged:
                    first_token_time = time.time()
                    print(f"[perf] first token after {first_token_time - gen_t0:.2f}s", flush=True)
                    first_token_logged = True
                # The stop token (im_end / </s>) is yielded before the summary;
                # don't emit it as part of the response.
                if int(event.token_id) in stop_ids:
                    continue
                text = decode_token(bundle.tokenizer, int(event.token_id))
                accumulated += text
                token_count += 1
                # Stop the exact millisecond a persona boundary token is decoded.
                hit_stop = any(s in accumulated for s in stop_strings)
                if stream_queue is not None:
                    stream_buffer += text
                    if _is_sentence_end(stream_buffer) or hit_stop:
                        chunk = polish_text(stream_buffer)
                        if chunk:
                            if chunk.endswith((".", "!", "?", "…")):
                                chunk += " "
                            if not _emit(chunk):
                                return "[Output firewall: blocked streaming content]"
                        stream_buffer = ""
                    if hit_stop:
                        break
                elif hit_stop:
                    break
            elif isinstance(event, SummaryEvent):
                summary = event

        # Hard truncation on persona boundaries after generation is complete.
        for s in stop_strings:
            if s in accumulated:
                accumulated = accumulated.split(s, 1)[0]
                break
        if stream_queue is not None and stream_buffer.strip():
            for s in stop_strings:
                if s in stream_buffer:
                    stream_buffer = stream_buffer.split(s, 1)[0]
                    break
            # If the run hit the token ceiling and the final fragment is an
            # incomplete sentence, don't stream a cut-off word.
            if token_count >= max_tokens and not _is_sentence_end(stream_buffer):
                pass
            else:
                chunk = polish_text(stream_buffer)
                if chunk:
                    if chunk.endswith((".", "!", "?", "…")):
                        chunk += " "
                    if not _emit(chunk):
                        return "[Output firewall: blocked streaming content]"
        if summary is not None:
            accept_pct = float(summary.acceptance_ratio) * 100.0
            # Total time includes prefill; if we have a first-token time, report
            # both the raw decode t/s and the full DFlash t/s.
            total_tps = summary.generation_tokens / (summary.elapsed_us / 1_000_000.0)
            decode_tps = (
                summary.generation_tokens / (time.time() - first_token_time)
                if first_token_time is not None
                else total_tps
            )
            print(
                f"[perf] {summary.generation_tokens} tokens @ "
                f"{decode_tps:.1f} decode t/s ({total_tps:.1f} total t/s), "
                f"draft_accept_ratio={accept_pct:.0f}%, "
                f"block_tokens={summary.block_tokens}, "
                f"peak_memory={summary.peak_memory_gb:.2f} GB",
                flush=True,
            )
            self.last_metrics = {
                "tokens": int(summary.generation_tokens),
                "decode_tps": float(decode_tps),
                "total_tps": float(total_tps),
                "draft_accept_pct": float(accept_pct),
                "peak_memory_gb": float(summary.peak_memory_gb),
            }
        elif token_count > 0:
            # DFlash did not yield a SummaryEvent (e.g., stopped on a boundary token).
            # Derive a decode t/s from wall-clock time and token count.
            elapsed = time.time() - (first_token_time or gen_t0)
            decode_tps = token_count / elapsed if elapsed > 0 else 0.0
            print(
                f"[perf] {token_count} tokens @ "
                f"{decode_tps:.1f} decode t/s, "
                f"draft_accept_ratio=N/A, "
                f"block_tokens={self.dflash_block_tokens}, "
                f"peak_memory={mx.get_peak_memory() / (1024 ** 3):.2f} GB",
                flush=True,
            )
            self.last_metrics = {
                "tokens": int(token_count),
                "decode_tps": float(decode_tps),
                "total_tps": float(decode_tps),
                "draft_accept_pct": 0.0,
                "peak_memory_gb": float(mx.get_peak_memory() / (1024 ** 3)),
            }
        # Purge Metal memory only when pressure is elevated.
        _maybe_purge_metal_cache()
        return accumulated

    def polish_response(self, text: str) -> str:
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"</s>|<\|endoftext\|>|</thinking>", "", text)
        text = text.replace("*", "")
        text = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+", " ", text)
        text = re.sub(r"[ʋʌɑɒɛɪʊɔəæ]", lambda m: {"ʋ":"v","ʌ":"v","ɑ":"a","ɒ":"o","ɛ":"e","ɪ":"i","ʊ":"u","ɔ":"o","ə":"a","æ":"a"}[m.group()], text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?— ?", "—", text)
        text = re.sub(r"\.\.\.", "…", text)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        # Remove any stray sign-off tokens
        text = re.sub(r"\s*—?\s*(mwah|besos|kisses)\s*", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[ \t]+", " ", text).strip()
        text = re.sub(r"\s*,\s*$", "", text)  # no trailing comma
        # If the response was cut off by max_tokens, trim to the last complete
        # sentence so we don't end with a dangling word or half-thought.
        if not re.search(r"[.!?…]$", text):
            m = re.search(r"^.*[.!?…]", text, flags=re.DOTALL)
            if m:
                text = m.group(0).strip()
        # Tighten trailing whitespace around any final ellipsis.
        text = re.sub(r"\s*…\s*$", "…", text)
        # Rewrite "fr fr" / "frfr" to "for real for real" as requested.
        text = re.sub(r"\bfr fr\b", "for real for real", text, flags=re.IGNORECASE)
        text = re.sub(r"\bfrfr\b", "for real for real", text, flags=re.IGNORECASE)
        return text

    async def _handle_agent_request(self, raw: str, writer: asyncio.StreamWriter):
        """Minimal local agent protocol (LAP) over SLICKS.

        Request envelope (JSON, embedded after the `__BADAPPLE_AGENT__ ` sentinel):
            {"id": "req-1", "method": "discover_tools"}
            {"id": "req-2", "method": "invoke_tool", "params": {"name": "run_shell", "args": {"command": "ls"}}}
            {"id": "req-3", "method": "inference", "params": {"prompt": "what is 2+2?", "max_new_tokens": 120}}
        """

        async def _respond(req_id: str | None, result: Any, error: str | None = None):
            frame: dict[str, Any] = {"id": req_id}
            if error:
                frame["type"] = "error"
                frame["message"] = error
            else:
                frame["type"] = "response"
                frame["result"] = result
            await _write_frame(writer, frame)

        try:
            req = json.loads(raw[len("__BADAPPLE_AGENT__ "):])
        except json.JSONDecodeError as e:
            await _respond(None, None, f"invalid agent JSON: {e}")
            return

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}

        if not self._is_passive_method(method):
            self.touch_activity()

        if method == "set_hibernate_after":
            seconds = float(params.get("seconds", 300))
            self.hibernate_after = max(0, seconds)
            await _respond(req_id, {"hibernate_after": self.hibernate_after})
            return

        if method == "runtime_status":
            ambient = None
            try:
                ambient = json.loads(badapple_ambient.get_context()) if badapple_ambient._CONTEXT_FILE.is_file() else None
            except Exception:
                pass
            ambient_running = badapple_ambient.is_running()
            await _respond(req_id, {
                "runtime": self.runtime.status(),
                "health": self.health.snapshot(),
                "resources": self.resources.snapshot(),
                "active_models": self.active_models(),
                "breakers": {name: vars(breaker.snapshot()) for name, breaker in self.breakers.items()},
                "autopilot": self.policy.autopilot,
                "fast_tier": self.fast_tier_enabled,
                "ambient_running": ambient_running,
                "ambient": ambient,
                "workspace": str(self.workspace.path) if self.workspace.path else None,
                "p2p_enabled": self.p2p is not None and self.p2p.is_running(),
                "p2p_peers": self.p2p.get_peers() if self.p2p is not None and self.p2p.is_running() else [],
                "mcp_socket": os.environ.get("BADAPPLE_MCP_SOCKET", "/var/run/badapple/mcp.sock"),
                "fast_model": self.fast_model_info,
                "hibernating": self.hibernating,
                "idle_seconds": round(time.time() - self.last_activity, 1),
            })
            return

        if method == "identity_status":
            await _respond(req_id, {"status": badapple_identity.status(), "public_key": badapple_identity.public_key()})
            return

        if method == "identity_sign":
            challenge = str(params.get("challenge", ""))
            if not challenge or len(challenge) > 4096:
                await _respond(req_id, None, "identity_sign requires a challenge up to 4096 characters")
                return
            await _respond(req_id, {"signature": badapple_identity.sign(challenge.encode("utf-8"))})
            return

        if method == "kill_switch":
            enabled = bool(params.get("enabled", True))
            if enabled:
                state = self.runtime.engage_kill_switch(params.get("reason", "agent requested"))
            else:
                state = self.runtime.reset_kill_switch()
                if state.get("safe_mode_reason"):
                    state = self.runtime.leave_safe_mode()
            await _respond(req_id, {"runtime": state})
            return

        if method == "private_mode":
            state = self.runtime.set_private_mode(bool(params.get("enabled", True)))
            await _respond(req_id, {"runtime": state})
            return

        if method == "discover_tools":
            await _respond(req_id, {"tools": TOOLS})
            return

        if method == "invoke_tool":
            if not self.runtime.allows_mutation():
                await _respond(req_id, None, "runtime is stopped or in safe mode")
                return
            tool_name = params.get("name", "")
            tool_args = params.get("args") or {}
            result = self._run_approved_tool(tool_name, tool_args, "agent request")
            await _respond(req_id, {"tool": tool_name, "result": result})
            return

        if method == "set_fast_tier":
            self.fast_tier_enabled = bool(params.get("enabled", True))
            await _respond(req_id, {"fast_tier": self.fast_tier_enabled})
            return

        if method == "set_autopilot":
            self.policy.set_autopilot(bool(params.get("enabled", False)))
            await _respond(req_id, {"autopilot": self.policy.autopilot})
            return

        if method == "inference":
            if not self.runtime.allows_generation():
                await _respond(req_id, None, "kill switch is engaged")
                return
            prompt = params.get("prompt", "")
            max_tokens = int(params.get("max_new_tokens", 120))
            if not prompt:
                await _respond(req_id, None, "inference requires prompt")
                return
            loop = asyncio.get_event_loop()

            def _gen():
                mx.set_default_device(self.mlx_device)
                try:
                    # The inference API is stateless: it must not mutate the
                    # conversational turn cache or return a cached conversational
                    # response. Build a single-turn prompt and stream directly.
                    messages = [
                        {"role": "system", "content": self.personas.get_system_prompt()},
                        {"role": "user", "content": prompt},
                    ]
                    rendered = self.render_prompt(messages, use_tools=False, voice_mode=False)
                    raw = self._stream(rendered, max_tokens, voice_mode=False)
                    text = self.polish_response(raw)
                    self._audit_record("query", {
                        "prompt": prompt,
                        "persona": self.personas.active,
                        "voice_mode": False,
                        "use_tools": False,
                    })
                    self._audit_record("response", {
                        "prompt": prompt,
                        "response": text[:500],
                        "persona": self.personas.active,
                    })
                    return text
                except Exception as e:
                    traceback.print_exc()
                    return f"Error generating response: {e}"

            text = await loop.run_in_executor(self.executor, _gen)
            metrics = self.last_metrics
            await _respond(req_id, {"text": text, "metrics": metrics})
            return

        if method == "switch_persona":
            name = params.get("name", "")
            if self.personas.switch(name):
                await _respond(req_id, {"active_persona": self.personas.active})
            else:
                await _respond(req_id, None, f"unknown persona '{name}'")
            return

        if method == "set_workspace":
            path = params.get("path", "")
            if not path:
                await _respond(req_id, None, "set_workspace requires path")
                return
            result = self.workspace.set(path)
            await _respond(req_id, {"status": result})
            return

        if method == "get_workspace":
            summary = self.workspace.summary() if self.workspace.path else None
            await _respond(req_id, {"workspace": str(self.workspace.path) if self.workspace.path else None, "summary": summary})
            return

        if method == "set_p2p":
            enabled = bool(params.get("enabled", False))
            if self.p2p is None:
                await _respond(req_id, None, "P2P is not available")
                return
            try:
                if enabled:
                    await self.p2p.start()
                else:
                    await self.p2p.stop()
            except Exception as e:
                await _respond(req_id, None, f"P2P toggle failed: {e}")
                return
            await _respond(req_id, {"p2p_enabled": self.p2p.is_running()})
            return

        if method == "set_ambient":
            enabled = bool(params.get("enabled", False))
            text = badapple_ambient.start() if enabled else badapple_ambient.stop()
            await _respond(req_id, {"ambient_running": badapple_ambient.is_running(), "message": text})
            return

        if method == "audit_tail":
            n = int(params.get("n", 20))
            entries = []
            if self.audit.ledger_path.is_file():
                try:
                    with open(self.audit.ledger_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    entries = [json.loads(l) for l in lines[-n:] if l.strip()]
                except Exception as e:
                    await _respond(req_id, None, f"could not read ledger: {e}")
                    return
            await _respond(req_id, {"entries": entries})
            return

        if method == "flush_vram":
            result = self.flush_vram()
            await _respond(req_id, {"result": result})
            return

        if method == "unload_model":
            model_type = str(params.get("type", "vision"))
            result = self.unload_model(model_type)
            await _respond(req_id, {"result": result})
            return

        if method == "get_pending_approvals":
            await _respond(req_id, {"pending": self.approval.get_pending_summary()})
            return
        if method == "p2p_peers":
            daemon = badapple_p2p.get_p2p_daemon()
            if daemon is None:
                await _respond(req_id, None, "P2P daemon is not running")
                return
            await _respond(req_id, {"peers_summary": daemon.get_peers()})
            return
        if method == "p2p_sync":
            daemon = badapple_p2p.get_p2p_daemon()
            if daemon is None:
                await _respond(req_id, None, "P2P daemon is not running")
                return
            result = await daemon.sync_memory()
            await _respond(req_id, {"sync_status": result})
            return

        await _respond(req_id, None, f"unknown method '{method}'")

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
            # Agent protocol: a signed JSON-RPC style request over the existing
            # SLICKS channel. The prompt is encoded as `__BADAPPLE_AGENT__ <json>`.
            if prompt.startswith("__BADAPPLE_AGENT__ "):
                await self._handle_agent_request(prompt, writer)
                return

            voice_mode = prompt.startswith("__BADAPPLE_VOICE__ ")
            if voice_mode:
                prompt = prompt[len("__BADAPPLE_VOICE__ "):]

            # Runtime persona selection sent by the CLI.
            if prompt.startswith("__BADAPPLE_PERSONA__"):
                rest = prompt[len("__BADAPPLE_PERSONA__"):]
                if "__" in rest:
                    name, prompt = rest.split("__", 1)
                    prompt = prompt.lstrip()
                    if not self.personas.switch(name):
                        await _write_frame(writer, {"type": "done", "text": f"Unknown persona '{name}'."})
                        return

            # Benchmark clients prepend this sentinel. It strips before any work
            # is done and bypasses fast tier, semantic cache, and tool fast paths
            # so the benchmark measures the 9B model and gets real metrics.
            benchmark_mode = prompt.startswith("__BADAPPLE_BENCHMARK__ ")
            if benchmark_mode:
                prompt = prompt[len("__BADAPPLE_BENCHMARK__ "):]

            if prompt in ("__BADAPPLE_SWITCH_DEEP__", "__BADAPPLE_SWITCH_FAST__"):
                await _write_frame(writer, {"type": "done", "text": ""})
                return

            if prompt.lower() in ("__badapple_new_chat__", "new chat", "clear conversation"):
                self.reset_conversation()
                await _write_frame(writer, {"type": "done", "text": "Okay, so... fresh start."})
                return

            control = prompt.strip().lower()
            if not self._is_passive_method(None, prompt):
                self.touch_activity()

            if control in ("stop everything", "emergency stop", "kill switch"):
                state = self.runtime.engage_kill_switch("user requested")
                badapple_ambient.stop()
                await _write_frame(writer, {"type": "done", "text": f"Kill switch engaged. Runtime mode: {state['mode']}."})
                return
            if control in ("resume bad apple", "reset kill switch", "resume everything"):
                state = self.runtime.reset_kill_switch()
                if state.get("safe_mode_reason"):
                    state = self.runtime.leave_safe_mode()
                await _write_frame(writer, {"type": "done", "text": f"Kill switch reset. Runtime mode: {state['mode']}."})
                return
            if control in ("safe mode off", "leave safe mode", "clear safe mode"):
                state = self.runtime.leave_safe_mode()
                await _write_frame(writer, {"type": "done", "text": f"Safe mode cleared. Runtime mode: {state['mode']}."})
                return
            if control in ("private mode on", "enable private mode"):
                self.runtime.set_private_mode(True)
                self.messages = [{"role": "system", "content": self.personas.get_system_prompt()}]
                await _write_frame(writer, {"type": "done", "text": "Private mode enabled. Memory, cache, conversation, and audit persistence are paused."})
                return
            if control in ("private mode off", "disable private mode"):
                self.runtime.set_private_mode(False)
                await _write_frame(writer, {"type": "done", "text": "Private mode disabled. Local persistence is active again."})
                return
            if control in ("runtime status", "health status", "bad apple status"):
                ambient = None
                try:
                    ambient = json.loads(badapple_ambient.get_context()) if badapple_ambient._CONTEXT_FILE.is_file() else None
                except Exception:
                    pass
                status = {
                    "runtime": self.runtime.status(),
                    "ambient_running": badapple_ambient.is_running(),
                    "health": self.health.snapshot(),
                    "resources": self.resources.snapshot(),
                    "active_models": self.active_models(),
                    "autopilot": self.policy.autopilot,
                    "fast_tier": self.fast_tier_enabled,
                    "ambient": ambient,
                    "workspace": str(self.workspace.path) if self.workspace.path else None,
                    "p2p_enabled": self.p2p is not None and self.p2p.is_running(),
                    "p2p_peers": self.p2p.get_peers() if self.p2p is not None and self.p2p.is_running() else [],
                    "mcp_socket": os.environ.get("BADAPPLE_MCP_SOCKET", "/var/run/badapple/mcp.sock"),
                    "hibernating": self.hibernating,
                    "idle_seconds": round(time.time() - self.last_activity, 1),
                }
                await _write_frame(writer, {"type": "done", "text": json.dumps(status, indent=2)})
                return
            if re.match(r"^hibernate after \d+$", control):
                seconds = int(control.split()[-1])
                self.hibernate_after = float(seconds)
                await _write_frame(writer, {"type": "done", "text": f"Hibernate after {seconds}s of inactivity."})
                return
            if control in ("flush vram", "purge vram", "clear metal cache"):
                result = self.flush_vram()
                await _write_frame(writer, {"type": "done", "text": json.dumps(result, indent=2)})
                return
            if control in ("unload vision model", "unload image model", "unload all models"):
                model_type = "vision" if "vision" in control else ("image" if "image" in control else "all")
                result = self.unload_model(model_type)
                await _write_frame(writer, {"type": "done", "text": json.dumps(result, indent=2)})
                return
            if control in ("fast tier on", "enable fast tier"):
                self.fast_tier_enabled = True
                await _write_frame(writer, {"type": "done", "text": "Fast tier enabled. Simple queries will bypass the 9B model when possible."})
                return
            if control in ("fast tier off", "disable fast tier"):
                self.fast_tier_enabled = False
                await _write_frame(writer, {"type": "done", "text": "Fast tier disabled. All queries route through the 9B model."})
                return
            if control in ("autopilot on", "enable autopilot"):
                self.policy.set_autopilot(True)
                await _write_frame(writer, {"type": "done", "text": "Autopilot enabled. I can run destructive tools without asking, babe."})
                return
            if control in ("autopilot off", "disable autopilot"):
                self.policy.set_autopilot(False)
                await _write_frame(writer, {"type": "done", "text": "Autopilot disabled. I'll ask before running destructive tools again."})
                return
            if control in ("start ambient", "enable ambient", "ambient on"):
                text = badapple_ambient.start()
                await _write_frame(writer, {"type": "done", "text": text})
                return
            if control in ("stop ambient", "disable ambient", "ambient off"):
                text = badapple_ambient.stop()
                await _write_frame(writer, {"type": "done", "text": text})
                return
            if control.startswith("set workspace to "):
                path = prompt[16:].strip()
                result = self.workspace.set(path)
                self.workspace_watcher.set_workspace(Path(path).expanduser() if path else None)
                await _write_frame(writer, {"type": "done", "text": result})
                return
            if control in ("consolidate memory", "dream", "offline consolidation"):
                result = self.memory.consolidate()
                await _write_frame(writer, {"type": "done", "text": result})
                return
            if control in ("clear cache", "clear semantic cache", "flush cache"):
                result = self.cache.clear()
                await _write_frame(writer, {"type": "done", "text": result})
                return
            if re.search(r"\b(?:what'?s my name|what is my name|who am i)\b", control):
                name = ""
                for f in self.memory._state.get("facts", []):
                    text = f.get("text", "")
                    if re.search(r"\bmy name is\b", text, re.IGNORECASE):
                        name = text.split("my name is", 1)[-1].strip().strip(".!?,")
                        break
                if name:
                    await _write_frame(writer, {"type": "done", "text": f"Your name is {name}, babe.", "metrics": self.last_metrics})
                    return
            if re.search(r"\bwhat do i (?:like|love|prefer|hate)\b", control):
                for f in self.memory._state.get("facts", []):
                    text = f.get("text", "")
                    if re.search(r"\bi (?:like|love|prefer|hate)\b", text, re.IGNORECASE):
                        await _write_frame(writer, {"type": "done", "text": text, "metrics": self.last_metrics})
                        return
            if control in ("open workspace", "show workspace"):
                path = str(self.workspace.path) if self.workspace.path else ""
                await _write_frame(writer, {"type": "done", "text": path or "No workspace set."})
                return
            if control in ("p2p on", "enable p2p"):
                if self.p2p is None:
                    await _write_frame(writer, {"type": "error", "message": "P2P is not available."})
                    return
                try:
                    await self.p2p.start()
                except Exception as e:
                    await _write_frame(writer, {"type": "error", "message": f"P2P start failed: {e}"})
                    return
                await _write_frame(writer, {"type": "done", "text": "P2P discovery and sync started, bestie."})
                return
            if control in ("p2p off", "disable p2p"):
                if self.p2p is None:
                    await _write_frame(writer, {"type": "error", "message": "P2P is not available."})
                    return
                await self.p2p.stop()
                await _write_frame(writer, {"type": "done", "text": "P2P discovery and sync stopped."})
                return
            if not self.runtime.allows_generation():
                await _write_frame(writer, {"type": "error", "message": "Bad Apple kill switch is engaged. Say 'resume bad apple' to reset it."})
                return

            # Persona commands are intercepted before any model or tool work.
            persona_resp = self.personas.handle_command(prompt)
            if persona_resp is not None:
                await _write_frame(writer, {"type": "done", "text": persona_resp})
                return

            # Approval command: execute a previously proposed destructive tool.
            approval_action = self.approval.handle_approve_command(prompt)
            if approval_action:
                if not self.runtime.allows_mutation():
                    await _write_frame(writer, {"type": "error", "message": "Runtime is stopped or in safe mode; approval execution is disabled."})
                    return
                tool_name, args = approval_action
                result = run_tool(tool_name, args, self.knowledge, policy=self.policy, workspace=self.workspace)
                self._audit_record("approval_execute", {"tool": tool_name, "args": args, "result": result[:500]})
                await _write_frame(writer, {"type": "done", "text": result})
                return

            if prompt.strip().lower() == "pending approvals":
                await _write_frame(writer, {"type": "done", "text": self.approval.get_pending_summary()})
                return

            # P2P direct commands
            low = prompt.strip().lower()
            if low in ("p2p sync", "sync memory", "sync my memory", "sync to peers"):
                if self.p2p is None:
                    await _write_frame(writer, {"type": "done", "text": "P2P daemon is not running."})
                    return
                result = await self.p2p.sync_memory()
                await _write_frame(writer, {"type": "done", "text": result})
                return
            if low in ("p2p peers", "discovered peers", "list peers"):
                if self.p2p is None:
                    await _write_frame(writer, {"type": "done", "text": "P2P daemon is not running."})
                    return
                await _write_frame(writer, {"type": "done", "text": self.p2p.get_peers()})
                return
            if low.startswith("p2p add peer "):
                spec = low[13:].strip()
                if self.p2p is None:
                    await _write_frame(writer, {"type": "done", "text": "P2P daemon is not running."})
                    return
                host, _, port = spec.rpartition(":")
                if not host or not port.isdigit():
                    await _write_frame(writer, {"type": "error", "message": "Usage: p2p add peer <host>:<port>"})
                    return
                text = self.p2p.add_peer(host, int(port))
                await _write_frame(writer, {"type": "done", "text": text})
                return
            if low.startswith("p2p remove peer "):
                spec = low[17:].strip()
                if self.p2p is None:
                    await _write_frame(writer, {"type": "done", "text": "P2P daemon is not running."})
                    return
                self.p2p.remove_peer(spec)
                await _write_frame(writer, {"type": "done", "text": f"Removed peer {spec} if it existed."})
                return

            # Model registry commands.
            if low == "scan models":
                result = self.model_registry.scan()
                await _write_frame(writer, {"type": "done", "text": result})
                return
            if low in ("list models", "show models", "what models do i have"):
                result = self.model_registry.list_models()
                await _write_frame(writer, {"type": "done", "text": result})
                return
            if low.startswith("model info "):
                result = self.model_registry.info(low[11:].strip())
                await _write_frame(writer, {"type": "done", "text": result})
                return
            m = re.match(r"^(?:use|load|switch to)\s+model\s+(.+)$", low)
            if m:
                model_ref = m.group(1).strip()
                # Resolve short name to cached path/id.
                ref_lower = model_ref.lower()
                for model in self.model_registry._state.get("models", []):
                    if model["id"].lower() == ref_lower:
                        model_ref = model["path"]
                        break
                result = await asyncio.get_event_loop().run_in_executor(self.executor, self.load_main_model, model_ref)
                await _write_frame(writer, {"type": "done", "text": result})
                return

            # Tiered fast path: greetings, identity, time, and simple math return
            # immediately without waking the 9B model when fast tier is enabled.
            # If a tiny fast model is loaded, short chitchat also goes through it.
            # Benchmark mode bypasses this so the benchmark measures the 9B path.
            if not benchmark_mode and not self.runtime.safe_mode and self.fast_tier_enabled:
                tier, payload = self.tier_router.select_tier(prompt)
                # Vision tier: go straight to the local VLM; no need to ask the 9B to call a tool.
                if tier == "vision":
                    result = self._run_approved_tool("capture_and_describe_screen", {"prompt": "Describe what is on the screen."}, prompt)
                    if not result.startswith(("Approval required", "Policy:", "Runtime", "Tool error")):
                        text = self.polish_response(postprocess_output(result))
                        self.record_fact(prompt, source="user")
                        self._add_episode(prompt, text)
                        self.messages.append({"role": "user", "content": prompt})
                        self.messages.append({"role": "assistant", "content": text})
                        self.prune_history()
                        self._save_conversation()
                        self._audit_record("response", {
                            "prompt": prompt,
                            "vision_path": True,
                            "response": text[:500],
                            "persona": self.personas.active,
                        })
                        await _write_frame(writer, {"type": "done", "text": text, "metrics": self.last_metrics})
                        return
                if tier == "fast" and payload is not None:
                    if payload.get("fast_model") and getattr(self, "fast_model", None) is not None and self.fast_tokenizer is not None:
                        text = badapple_fast_model.generate_fast(
                            self.fast_model,
                            self.fast_tokenizer,
                            prompt,
                            system_prompt=self.personas.get_system_prompt(),
                            max_tokens=48,
                            temperature=0.0,
                        )
                        text = self.polish_response(postprocess_output(text))
                    else:
                        text = self.polish_response(postprocess_output(payload["text"]))
                    self.record_fact(prompt, source="user")
                    self._add_episode(prompt, text)
                    self.messages.append({"role": "user", "content": prompt})
                    self.messages.append({"role": "assistant", "content": text})
                    self.prune_history()
                    self._save_conversation()
                    self._audit_record("response", {
                        "prompt": prompt,
                        "fast_path": True,
                        "tier": "fast",
                        "fast_model": payload.get("fast_model", False),
                        "response": text[:500],
                        "persona": self.personas.active,
                    })
                    await _write_frame(writer, {"type": "done", "text": text})
                    return

            # Fast deterministic path for direct tool commands (read, list, run, search, write).
            # This avoids a full 8B generation for simple local actions and stays air-gapped.
            fast = None if (benchmark_mode or self.runtime.safe_mode) else fast_execute(
                prompt,
                self.knowledge,
                approval=self.approval,
                policy=self.policy,
                workspace=self.workspace,
            )
            if fast:
                fast = self.polish_response(postprocess_output(fast))
                self.record_fact(prompt, source="user")
                self._add_episode(prompt, fast)
                self.messages.append({"role": "user", "content": prompt})
                self.messages.append({"role": "assistant", "content": fast})
                self.prune_history()
                self._save_conversation()
                self._audit_record("response", {
                    "prompt": prompt,
                    "fast_path": True,
                    "response": fast[:500],
                    "persona": self.personas.active,
                })
                metrics = self.last_metrics
                await _write_frame(writer, {"type": "done", "text": fast, "metrics": metrics})
                return

            # Natural-language tool router: if the prompt clearly maps to a known
            # tool, run it directly without waiting for the 9B to emit a tool_call.
            if not benchmark_mode and not self.runtime.safe_mode:
                routed = self.tool_router.resolve(prompt)
                if routed:
                    tool_name, args, conf = routed
                    result = self._run_approved_tool(tool_name, args, prompt)
                    if not result.startswith(("Approval required", "Policy:", "Runtime", "Tool error")):
                        text = self.polish_response(postprocess_output(result))
                        self.tool_router.record_success(prompt, tool_name)
                        self.record_fact(prompt, source="user")
                        self._add_episode(prompt, text)
                        self.messages.append({"role": "user", "content": prompt})
                        self.messages.append({"role": "assistant", "content": text})
                        self.prune_history()
                        self._save_conversation()
                        self._audit_record("response", {
                            "prompt": prompt,
                            "fast_path": True,
                            "routed": True,
                            "tool": tool_name,
                            "confidence": conf,
                            "response": text[:500],
                            "persona": self.personas.active,
                        })
                        await _write_frame(writer, {"type": "done", "text": text, "metrics": self.last_metrics})
                        return

            loop = asyncio.get_event_loop()

            # Multi-step task planning for requests that combine actions.
            if not benchmark_mode and is_multi_step(prompt):
                def _plan():
                    mx.set_default_device(self.mlx_device)
                    try:
                        result = self.plan_and_execute(prompt, max_new_tokens, voice_mode=voice_mode)
                        if result:
                            return self.polish_response(result)
                        return self.polish_response(self.generate_with_tools(prompt, max_new_tokens, voice_mode=voice_mode, benchmark=benchmark_mode))
                    except Exception as e:
                        traceback.print_exc()
                        return f"Error: {e}"
                text = await loop.run_in_executor(self.executor, _plan)
                if not text:
                    text = "Ugh, like, I couldn't make a plan."
                self.record_fact(prompt, source="user")
                self._add_episode(prompt, text)
                self.messages.append({"role": "user", "content": prompt})
                self.messages.append({"role": "assistant", "content": text})
                self.prune_history()
                self._save_conversation()
                self._audit_record("response", {
                    "prompt": prompt,
                    "multi_step": True,
                    "response": text[:500],
                    "persona": self.personas.active,
                })
                metrics = self.last_metrics
                await _write_frame(writer, {"type": "done", "text": text, "metrics": metrics})
                return

            stream_queue = queue.Queue()

            def _gen():
                # The asyncio executor worker thread is the same thread the model
                # was loaded on, so its default device/stream are already correct.
                mx.set_default_device(self.mlx_device)
                try:
                    if benchmark_mode:
                        # Benchmark needs a clean, single 9B generation with metrics:
                        # no tools, cache, fast paths, multi-step loops, or retrieved context.
                        messages = [
                            {"role": "system", "content": self.personas.get_system_prompt()},
                            {"role": "user", "content": prompt},
                        ]
                        rendered = self.render_prompt(messages, use_tools=False, voice_mode=False, benchmark=True)
                        raw = self._stream(rendered, max_new_tokens, stream_queue=stream_queue, voice_mode=False)
                        return self.polish_response(raw)
                    raw = self.generate_with_tools(prompt, max_new_tokens, voice_mode=voice_mode, stream_queue=stream_queue, benchmark=benchmark_mode)
                    return self.polish_response(raw)
                except Exception as e:
                    traceback.print_exc()
                    return f"Error generating response: {e}"

            future = loop.run_in_executor(self.executor, _gen)

            # Stream sentence chunks as the 8B model generates.
            while not future.done() or not stream_queue.empty():
                chunk = await loop.run_in_executor(None, _queue_get, stream_queue, 0.2)
                if chunk:
                    await _write_frame(writer, {"type": "token", "text": chunk})

            text = future.result()

            if not text:
                text = "Hiiii... I'm here."

            # Store final assistant response in conversation; only user statements
            # become long-term memory, not the assistant's own rephrasings.
            self.messages.append({"role": "assistant", "content": text})
            self._add_episode(prompt, text)
            self.prune_history()
            self._save_conversation()
            self._audit_record("response", {
                "prompt": prompt,
                "response": text[:500],
                "persona": self.personas.active,
            })

            metrics = self.last_metrics
            await _write_frame(writer, {"type": "done", "text": text, "metrics": metrics})

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


class DashboardServer:
    """Start the local-only observability dashboard in a daemon thread.

    This wraps badapple_dashboard.DashboardWebServer and hands it the running
    MLXServer instance so /api/status can read live daemon state.
    """

    def __init__(
        self,
        mlx_server: "MLXServer",
        host: str = "127.0.0.1",
        port: int = 8787,
    ):
        self.mlx_server = mlx_server
        self.host = host
        self.port = port
        self._web = badapple_dashboard.DashboardWebServer(host, port)

    def start(self) -> None:
        self._web.start(self.mlx_server)


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

    # DFlash/MLX streams are bound to the thread that created them. Load and
    # run the model in a single dedicated worker thread.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="badapple_mlx")
    loop = asyncio.get_running_loop()
    server = await loop.run_in_executor(executor, MLXServer, secret, system_prompt)
    server.executor = executor
    server.loop = loop

    srv = await asyncio.start_unix_server(server.handle_client, path=socket_path)
    os.chmod(socket_path, 0o666)
    try:
        os.symlink(socket_path, fast_socket_path)
        os.chmod(fast_socket_path, 0o666)
    except FileExistsError:
        pass
    print(f"Bad Apple MLX server listening on {socket_path}", flush=True)

    # Start the local-only observability dashboard (127.0.0.1 only).
    try:
        dashboard = DashboardServer(server)
        dashboard.start()
        print(f"[main] Dashboard available at http://{dashboard.host}:{dashboard.port}/", flush=True)
    except Exception as e:
        print(f"[main] Dashboard failed to start: {e}", flush=True)

    # Start the local-only P2P sync daemon on the same event loop.
    # P2P is on by default; set BADAPPLE_P2P=0 to keep the daemon air-gapped.
    if server.p2p is not None and os.environ.get("BADAPPLE_P2P", "1") != "0":
        try:
            await server.p2p.start()
        except Exception as e:
            print(f"[main] P2P daemon failed to start: {e}", flush=True)

    # Start the local task scheduler background thread.
    try:
        badapple_scheduler.start_background_scheduler(interval=60)
        print("[main] Background task scheduler started", flush=True)
    except Exception as e:
        print(f"[main] Scheduler failed to start: {e}", flush=True)

    # Start the local MCP server (Unix socket only; uses the same SLICKS agent channel).
    mcp_process: subprocess.Popen | None = None
    try:
        mcp_env = os.environ.copy()
        mcp_env.setdefault("BADAPPLE_MCP_SOCKET", "/var/run/badapple/mcp.sock")
        mcp_process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).with_name("badapple_mcp_server.py"))],
            cwd=str(Path(__file__).resolve().parent),
            env=mcp_env,
            start_new_session=True,
        )
        print(f"[main] MCP server started on {mcp_env['BADAPPLE_MCP_SOCKET']}", flush=True)
    except Exception as e:
        print(f"[main] MCP server failed to start: {e}", flush=True)

    try:
        asyncio.create_task(server.hibernation_watcher())
        async with srv:
            await srv.serve_forever()
    finally:
        if mcp_process is not None:
            try:
                mcp_process.terminate()
                mcp_process.wait(timeout=5)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
