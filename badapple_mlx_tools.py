"""Bad Apple MLX server — tool schemas, tool execution, and text processing.

Extracted from badapple_mlx_server.py to keep the main daemon file focused on
model loading, generation, streaming, and the SLICKS server loop.
"""
import json
import os
import queue
import re
from pathlib import Path
from typing import Any

from langdetect import LangDetectException, detect

from badapple_knowledge import BadAppleKnowledge
from badapple_tools import run_tool

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
            "description": "Schedule a local shell command or Bad Apple query to run later. Commands are restricted to the same read-only allowlist as run_shell (ls, cat, head, tail, find, grep, wc, file, pwd, mdfind, ps, df, du, echo, whoami, id, git, swift, cargo, rustc, python3, python), no redirection/pipes/multiple commands. `when` is seconds from now or an ISO timestamp. `repeat` is optional seconds for recurring tasks.",
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
            "name": "ocular_start",
            "description": "Start the Ocular UI Stream. Captures the screen every capture_interval seconds and, if describe_interval > 0, runs the local VLM to describe it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "capture_interval": {"type": "number", "description": "Screen capture interval in seconds. Default 5."},
                    "describe_interval": {"type": "number", "description": "VLM describe interval in seconds. 0 disables description. Default 0."},
                    "prompt": {"type": "string", "description": "Optional prompt for the VLM description."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ocular_stop",
            "description": "Stop the Ocular UI Stream and unload the vision model.",
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
            "name": "ocular_context",
            "description": "Get the latest Ocular UI Stream context: screenshot, active app/window, and the most recent VLM description.",
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
            "name": "browser_action",
            "description": "Drive Safari autonomously via AppleScript. Actions: 'navigate' opens a URL; 'url' returns the current page URL; 'title' returns the page title; 'text' returns visible page text; 'click' clicks an element by CSS selector; 'type' fills an input by CSS selector; 'scroll' scrolls the page; 'exec' runs arbitrary JavaScript in the page and returns the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "url", "title", "text", "click", "type", "scroll", "exec"],
                        "description": "The browser action to perform.",
                    },
                    "url": {
                        "type": "string",
                        "description": "For navigate, the URL to open.",
                    },
                    "selector": {
                        "type": "string",
                        "description": "For click/type, a CSS selector for the target element.",
                    },
                    "text": {
                        "type": "string",
                        "description": "For type, the text to enter into the field.",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "For scroll, pixels to scroll (positive=down, negative=up).",
                    },
                    "javascript": {
                        "type": "string",
                        "description": "For exec, the JavaScript code to run in the page.",
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
    (["browser", "safari", "web page", "website", "navigate to", "open url", "click on page", "fill form", "search the web", "go to website"], ["browser_action"]),
    (["mcp", "model context protocol", "mcp server", "mcp tool", "mcp marketplace"], ["list_mcp_servers", "add_mcp_server", "list_mcp_tools", "invoke_mcp_tool", "mcp_marketplace"]),
    (["do for me", "do this", "do the following", "run a task", "execute a task", "plan and", "multi-step", "step by step"], ["run_agent_task"]),
    (["set project", "this project is", "project context", "project goals"], ["set_project_context", "get_project_context"]),
]

DEFAULT_TOOL_NAMES = {
    "get_current_time", "list_directory", "read_file", "write_file", "run_shell",
    "run_applescript", "run_shortcut", "search_content", "search_local_files",
    "git_status", "index_documents", "search_notes", "read_working_memory",
    "capture_and_describe_screen", "workspace_status", "ui_action", "browser_action",
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

def should_use_tools(prompt: str) -> bool:
    low = prompt.lower()
    return any(k in low for k in TOOL_KEYWORDS)


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

def fast_execute(
    prompt: str,
    knowledge: BadAppleKnowledge | None = None,
    approval: Any | None = None,
    policy: Any | None = None,
    workspace: Any | None = None,
    mcp_marketplace: Any | None = None,
) -> str | None:
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
        return run_tool(name, args, knowledge, approval=approval, policy=policy, workspace=workspace, user_prompt=prompt, mcp_marketplace=mcp_marketplace)

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



def generate_with_tools(
    server,
    user_prompt: str,
    max_tokens: int,
    voice_mode: bool = False,
    stream_queue: queue.Queue | None = None,
    benchmark: bool = False,
) -> str:
    server.touch_activity()
    server.check_prompt_reload()

    # Workspace commands (set workspace to ... / clear workspace) are handled
    # without the 9B model (unless benchmarking, where we want to force generation).
    if not benchmark:
        ws_low = user_prompt.strip().lower()
        if ws_low.startswith("set workspace to "):
            path = user_prompt[16:].strip()
            resp = server.workspace.set(path)
            server.workspace_watcher.set_workspace(Path(path).expanduser() if path else None)
            server._audit_record("workspace", {"prompt": user_prompt, "response": resp})
            server.messages.append({"role": "user", "content": user_prompt})
            server.messages.append({"role": "assistant", "content": resp})
            server.prune_history()
            server._save_conversation()
            return resp
        if ws_low in ("clear workspace", "unset workspace"):
            resp = server.workspace.clear()
            server._audit_record("workspace", {"prompt": user_prompt, "response": resp})
            server.messages.append({"role": "user", "content": user_prompt})
            server.messages.append({"role": "assistant", "content": resp})
            server.prune_history()
            server._save_conversation()
            return resp

        # Persona commands (switch, teach) are handled without the 9B model.
        persona_resp = server.personas.handle_command(user_prompt)
        if persona_resp is not None:
            server._audit_record("persona_command", {
                "prompt": user_prompt,
                "active_persona": server.personas.active,
                "response": persona_resp,
            })
            server.messages.append({"role": "user", "content": user_prompt})
            server.messages.append({"role": "assistant", "content": persona_resp})
            server.prune_history()
            server._save_conversation()
            return persona_resp

        # Capability and creator questions are answered directly so the 9B does
        # not fall back into generic model identity or skip the useful part.
        meta_resp = server._try_meta_response(user_prompt, voice_mode=voice_mode)
        if meta_resp is not None:
            kind = "capabilities" if "can do" in meta_resp or "what I can do" in meta_resp else "identity"
            server._audit_record(kind, {"prompt": user_prompt, "response": meta_resp})
            server.messages.append({"role": "user", "content": user_prompt})
            server.messages.append({"role": "assistant", "content": meta_resp})
            server.prune_history()
            server._save_conversation()
            if stream_queue is not None:
                stream_queue.put(meta_resp)
            return meta_resp

        # Semantic cache: bypass the 9B for repeated questions.
        if not server.runtime.private_mode and not voice_mode and not should_use_tools(user_prompt):
            cached = server.cache.lookup(user_prompt, persona=server.personas.active)
            if cached:
                if server.firewall.check_full(cached):
                    cached = "[Output firewall: I caught a pattern I am not allowed to say out loud.]"
                server._audit_record("cache_hit", {"prompt": user_prompt, "response": cached[:500], "persona": server.personas.active})
                server.messages.append({"role": "user", "content": user_prompt})
                server.messages.append({"role": "assistant", "content": cached})
                server.prune_history()
                server._save_conversation()
                return cached

    if user_prompt.strip().lower() == "new chat":
        server.reset_conversation()
        # Ask the model for a fresh English greeting instead of treating it as a command.
        user_prompt = "Greet me"
    server.record_fact(user_prompt, source="user")
    messages = server.build_messages(user_prompt)
    use_tools = should_use_tools(user_prompt) and not benchmark

    server._audit_record("query", {
        "prompt": user_prompt,
        "persona": server.personas.active,
        "voice_mode": voice_mode,
        "use_tools": use_tools,
    })

    def clean(raw: str) -> str:
        _, text = extract_tool_calls(raw)
        text = postprocess_output(text)
        if os.environ.get("BADAPPLE_DISABLE_FIREWALL") == "1":
            return text
        matched = server.firewall.check_full(text)
        if matched:
            print(f"[firewall] blocked pattern {matched!r} in final text: {text[:200]!r}", flush=True)
            return "[Output firewall: I caught a pattern I am not allowed to say out loud.]"
        return text

    # Ensure the main model is loaded before we try to use its tokenizer
    # in render_prompt (lazy loading defers this until the first request).
    server._ensure_main_model()

    # First generation. Only stream when tools are not offered, because tool
    # reasoning can produce intermediate <tool_call> blocks we don't want
    # mixed into the streamed voice/text output.
    raw = server._stream(
        server.render_prompt(messages, use_tools=use_tools, voice_mode=voice_mode),
        max_tokens,
        stream_queue=stream_queue if not use_tools else None,
        voice_mode=voice_mode,
    )
    tool_calls, _ = extract_tool_calls(raw)

    if not tool_calls:
        final = clean(raw)
        server._cache_store(user_prompt, final)
        return final

    # Tool loop (multi-step task execution; allow more chained tool calls)
    for _ in range(5):
        for call in tool_calls:
            result = server._run_approved_tool(call["name"], call.get("arguments", {}), user_prompt)
            server.messages.append({
                "role": "user",
                "content": f"Tool result for {call['name']}:\n{result}",
            })
            server._audit_record("tool_result", {
                "prompt": user_prompt,
                "tool": call["name"],
                "result": result[:500],
            })
        # Re-render and generate after tool results
        raw = server._stream(server.render_prompt(server.messages, use_tools=True, voice_mode=voice_mode), max_tokens, voice_mode=voice_mode)
        tool_calls, _ = extract_tool_calls(raw)
        if not tool_calls:
            final = clean(raw)
            server._cache_store(user_prompt, final)
            return final

    final = clean(raw)
    server._cache_store(user_prompt, final)
    return final




def run_approved_tool(server, name: str, args: dict[str, Any], user_prompt: str) -> str:
    """Run a tool, but gate destructive tools behind the approval workflow."""
    if not server.runtime.allows_mutation():
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
    admitted, reason, _ = server.resources.admit(capability)
    if not admitted:
        return f"Resource governor: {reason}."
    if not server.breakers.allow("tools"):
        return "Tool circuit breaker is open; retry after the cooldown."
    if not server.policy.is_allowed(name):
        return f"Policy: tool '{name}' is not allowed."
    error = server.policy.validate(name, args)
    if error:
        return f"Policy: {error}"
    if server.approval.needs_approval(name):
        proposal_id = server.approval.propose(name, args, user_prompt)
        return (
            f"Approval required before I can run {name}. "
            f"Reply with 'approve {proposal_id}' to proceed. "
            f"(Set BADAPPLE_AUTOPILOT=1 to skip these prompts.)"
        )
    try:
        if name == "learn_workflow":
            result = server.memory.learn_workflow(args.get("name", ""), args.get("trigger", ""), args.get("steps") or [])
        elif name == "list_workflows":
            result = json.dumps(server.memory.workflows(), indent=2, default=str)
        elif name == "consolidate_memory":
            result = server.memory.consolidate()
        elif name == "set_workflow_enabled":
            result = server.memory.set_workflow_enabled(args.get("name", ""), bool(args.get("enabled")))
        elif name == "run_agent_task":
            result = server.run_agent_task(args.get("goal", ""), int(args.get("max_steps") or 10))
        elif name == "set_project_context":
            result = server.memory.set_project_context(
                args.get("name", ""),
                args.get("description", ""),
                args.get("goals") or [],
                args.get("tags") or [],
            )
        elif name == "get_project_context":
            result = server.memory.get_project_context()
        elif name == "run_workflow":
            workflow = next((item for item in server.memory.workflows() if item.get("name") == args.get("name")), None)
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
                    output = server._run_approved_tool(step.get("tool", ""), step.get("args") or {}, user_prompt)
                    outputs.append(f"{step.get('tool')}: {output}")
                    if output.startswith(("Approval required", "Policy:", "Runtime", "Tool error")):
                        break
                result = "\n".join(outputs)
        elif server.plugins.has_tool(name):
            result = server.plugins.invoke(name, args, timeout=server.policy.timeout(name))
        else:
            result = run_tool(name, args, server.knowledge, approval=server.approval, policy=server.policy, workspace=server.workspace, user_prompt=user_prompt, mcp_marketplace=server.mcp_marketplace)
        if result.lower().startswith("error"):
            server.breakers.failure("tools")
        else:
            server.breakers.success("tools")
        return result
    except (TypeError, ValueError, LookupError) as e:
        server.breakers.failure("tools")
        return f"Tool error: {e}"

