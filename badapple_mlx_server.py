#!/usr/bin/env python3
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
import base64
import concurrent.futures
import copy
import gc
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

import badapple_agent_tasks
import badapple_ambient
import badapple_ambient_memory
import badapple_audit_actor
import badapple_breakers_actor
import badapple_cache_actor
import badapple_dashboard
import badapple_fact_extractor
import badapple_fast_model
import badapple_health_actor
import badapple_identity
import badapple_mcp_actor
import badapple_metrics
import badapple_model_actor
import badapple_model_registry
import badapple_ocular
import badapple_p2p
import badapple_p2p_actor
import badapple_persona_actor
import badapple_resources_actor
import badapple_scheduler
import badapple_slicks
import badapple_speculate
import badapple_task_actor
import badapple_tier
import badapple_tool_router
import badapple_vision
import badapple_vram_governor
import badapple_workspace_actor
import badapple_workspace_watcher
from badapple_extras import (
    ApprovalGate,
    AuditLedger,
    MemoryGraph,
    Policy,
    StreamingFirewall,
    Workspace,
)
from badapple_knowledge import BadAppleKnowledge
from badapple_plugins import PluginRegistry
from badapple_runtime import (
    RuntimeControl,
)
from badapple_tools import (
    get_session_seed,
    run_tool,
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
except ImportError:
    _dflash_available = False

# --- Split-module imports ---
# Tool schemas, tool execution, and text processing live in badapple_mlx_tools.
from badapple_mlx_tools import (
    TOOLS,
    _is_sentence_end,
    _queue_get,
    fast_execute,
    generate_with_tools as _generate_with_tools,
    is_multi_step,
    polish_text,
    postprocess_output,
    run_approved_tool as _run_approved_tool_func,
    tools_for_prompt,
)
# Conversation persistence helpers.
from badapple_mlx_conversation import (
    conversation_path,
    load_conversation,
    save_conversation,
)
# RAG context building and KV prompt-cache management.
from badapple_mlx_rag import (
    build_retrieval_context,
    ensure_prompt_cache,
    kv_cache_paths,
    load_kv_cache,
    prime_system_cache,
    save_kv_cache,
)
# Agent protocol handler, multi-step tasks, and agent task management.
from badapple_mlx_agent import (
    _write_frame,
    agent_done as _agent_done_func,
    cancel_agent_task as _cancel_agent_task,
    extract_agent_json as _extract_agent_json_func,
    extract_agent_xml as _extract_agent_xml_func,
    get_agent_task as _get_agent_task,
    handle_agent_request as _handle_agent_request_func,
    list_agent_tasks as _list_agent_tasks,
    pause_agent_task as _pause_agent_task,
    plan_and_execute as _plan_and_execute,
    resume_agent_task as _resume_agent_task,
    run_agent_task as _run_agent_task_func,
    submit_agent_task as _submit_agent_task,
)


# Protocol constants from bad_apple_ipc.rs
SLICKS_VERSION = badapple_slicks.SLICKS_VERSION
DEFAULT_SOCKET_PATH = "/var/run/badapple/substrate.sock"
HANDSHAKE_MAX_SKEW_MS = badapple_slicks.HANDSHAKE_MAX_SKEW_MS
MAX_PROMPT_BYTES = 64 * 1024
MAX_NEW_TOKENS = 512
MAX_FRAME_BYTES = 1024 * 1024

# Main Qwen 3.5 9B 4-bit brain. Unified for both text and voice.
MAIN_MODEL = os.environ.get("BADAPPLE_MAIN_MODEL", "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit").strip()

# Pinned HuggingFace revision (commit hash) for the main brain.
# Pinning to a specific commit guarantees that the semantic-cache embeddings,
# token vectors, and ANE shard manifest stay consistent across daemon restarts.
# If the upstream repo is updated or yanked, an unpinned "main" ref could load a
# different snapshot and silently invalidate those artifacts. Override at runtime
# with BADAPPLE_MODEL_REVISION; an empty/unset value falls back to the constant.
DEFAULT_MODEL_REVISION = "5ae9734004d530171fd52f89e660c059b6e36efc"
MODEL_REVISION = (os.environ.get("BADAPPLE_MODEL_REVISION") or DEFAULT_MODEL_REVISION).strip()

# DFlash speculative draft for the 9B brain (same architecture).
DRAFT_MODEL = os.environ.get("BADAPPLE_DRAFT_MODEL", "z-lab/Qwen3.5-9B-DFlash").strip()

# Standard mlx-lm speculative decoding. Set BADAPPLE_SPECULATIVE_DRAFT to a
# model ref/path or to "auto" to scan the HF cache for a small draft.
SPECULATIVE_DRAFT = os.environ.get("BADAPPLE_SPECULATIVE_DRAFT", "").strip()
NUM_DRAFT_TOKENS = int(os.environ.get("BADAPPLE_NUM_DRAFT_TOKENS") or "3")

# Generation memory budget. Reduce for larger models (e.g. 32B Qwen at 1024-2048).
PREFILL_STEP_SIZE = int(os.environ.get("BADAPPLE_PREFILL_STEP_SIZE") or "4096")
MAX_KV_SIZE = int(os.environ.get("BADAPPLE_MAX_KV_SIZE") or "4096")


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
# Tight, low-latency voice prompt now lives in badapple_extras.py as
# DEFAULT_VOICE_SYSTEM_PROMPT so the default persona pack can use it.

def load_prompt() -> str:
    """Load the system prompt from the on-disk prompt file, falling back to the
    embedded default. Writing prompts to a file lets the daemon hot-reload the
    persona without restarting and reloading the 9B model.
    """
    prompt_path = os.environ.get("BADAPPLE_PROMPT_FILE") or str(DEFAULT_PROMPT_FILE)
    path = Path(prompt_path).expanduser()
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except (OSError, ValueError) as e:
            print(f"[mlx_server] strip failed: {e}", flush=True)

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
    can keep its temporary pools hot between turns.
    """
    cache_gb = mx.get_cache_memory() / (1024 ** 3)
    active_gb = mx.get_active_memory() / (1024 ** 3)
    if cache_gb > 1.5 or active_gb > 8.0:
        gc.collect()
        mx.clear_cache()
        print(f"[perf] purged Metal cache (cache={cache_gb:.2f} GB, active={active_gb:.2f} GB)", flush=True)
    else:
        gc.collect()

load_slicks_secret = badapple_slicks.load_slicks_secret


def random_nonce() -> str:
    return os.urandom(32).hex()


def timestamp_is_fresh(timestamp_ms: int) -> bool:
    return badapple_slicks.timestamp_is_fresh(timestamp_ms)


def validate_request(prompt, max_new_tokens):
    if len(prompt) > MAX_PROMPT_BYTES:
        raise ValueError("prompt exceeds maximum length")
    if not (1 <= max_new_tokens <= MAX_NEW_TOKENS):
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")

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
        self._model_actor = badapple_model_actor.ModelActor(self.data_dir)
        self._model_actor.start()
        self.model_manager = badapple_model_actor.ModelActorProxy(self._model_actor)
        self._task_actor = badapple_task_actor.TaskActor(self.data_dir)
        self._task_actor.start()
        self.agent_task_manager = badapple_task_actor.TaskActorProxy(self._task_actor)
        self.max_kv_size = MAX_KV_SIZE
        self.prefill_step_size = PREFILL_STEP_SIZE
        self._roast_index = 0
        self.last_metrics: dict[str, Any] | None = None
        self.metrics_collector = badapple_metrics.MetricsCollector()
        self.metrics_actor = badapple_metrics.MetricsActor(existing_collector=self.metrics_collector)
        self.metrics_actor.start()

        # Reusable prompt KV cache. _system_prompt_cache holds the pristine
        # system-prefix KV; _prompt_cache is a per-query deep copy that gets
        # mutated during generation. This avoids the expensive re-prefill of
        # the system prompt after every response.
        self._system_prompt_cache: list[Any] | None = None
        self.prompt_cache: list[Any] | None = None
        self._cache_system_hash: str | None = None

        # Persistent KV cache directory (saved across daemon restarts so the
        # ~9 s system-prompt prefill is only paid once, not after every reboot).
        self._kv_cache_dir = self.data_dir / "kv_cache"

        # Runtime/health/breakers
        self.runtime = RuntimeControl(self.data_dir)
        self._health_actor = badapple_health_actor.HealthActor()
        self._health_actor.start()
        self.health = badapple_health_actor.HealthActorProxy(self._health_actor)

        # Resource governor and circuit breakers run on dedicated actor threads.
        self._resources_actor = badapple_resources_actor.ResourcesActor()
        self._resources_actor.start()
        self.resources = badapple_resources_actor.ResourcesActorProxy(self._resources_actor)
        self._breakers_actor = badapple_breakers_actor.BreakersActor()
        self._breakers_actor.start()
        self.breakers = badapple_breakers_actor.BreakersActorProxy(self._breakers_actor)

        self.generations = GenerationStore(self.data_dir)
        self.plugins = PluginRegistry(self.data_dir)
        registered = {item.get("function", {}).get("name") for item in TOOLS}
        TOOLS.extend(schema for schema in self.plugins.tool_schemas() if schema["function"]["name"] not in registered)
        self.memory = MemoryGraph(self.data_dir, encoder=self.knowledge._encode_texts)
        self.ambient_memory = badapple_ambient_memory.AmbientMemory(self.memory, self.data_dir)

        # Persona pack runs on a dedicated actor thread.
        self._persona_actor = badapple_persona_actor.PersonaActor(self.data_dir, self.prompt_file)
        self._persona_actor.start()
        self.personas = badapple_persona_actor.PersonaActorProxy(self._persona_actor)

        self.firewall = StreamingFirewall(self.data_dir)
        self.audit_collector = AuditLedger(self.data_dir)
        self.audit_actor = badapple_audit_actor.AuditActor(existing=self.audit_collector)
        self.audit_actor.start()
        self._cache_actor = badapple_cache_actor.CacheActor(self.data_dir)
        self._cache_actor.start()
        self.cache = badapple_cache_actor.CacheActorProxy(self._cache_actor)
        self.policy = Policy(self.data_dir)

        # Workspace runs on a dedicated actor thread.  We keep a reference to
        # the underlying object so P2P can read workspace context directly.
        _workspace_obj = Workspace(self.data_dir)
        self._workspace_actor = badapple_workspace_actor.WorkspaceActor(existing=_workspace_obj)
        self._workspace_actor.start()
        self.workspace = badapple_workspace_actor.WorkspaceActorProxy(self._workspace_actor)
        self.workspace_watcher = badapple_workspace_watcher.WorkspaceWatcher(self.knowledge)
        if os.environ.get("BADAPPLE_WORKSPACE_DIR"):
            self.workspace.set(os.environ["BADAPPLE_WORKSPACE_DIR"])
            self.workspace_watcher.set_workspace(Path(os.environ["BADAPPLE_WORKSPACE_DIR"]).expanduser())
        self.workspace_watcher.start()

        # P2P sync daemon runs on its own asyncio thread inside an actor.
        self._p2p_actor = badapple_p2p_actor.P2PActor(secret, self.data_dir, memory=self.memory, workspace=_workspace_obj, model_registry=self.model_registry)
        self._p2p_actor.start()
        self.p2p = badapple_p2p_actor.P2PActorProxy(self._p2p_actor)
        badapple_p2p.set_p2p_daemon(self.p2p)

        # MCP marketplace runs on a dedicated actor thread.
        self._mcp_actor = badapple_mcp_actor.MCPActor()
        self._mcp_actor.start()
        self.mcp_actor = self._mcp_actor
        self.mcp_marketplace = badapple_mcp_actor.MCPActorProxy(self._mcp_actor)

        self.approval = ApprovalGate(self.data_dir, policy=self.policy)

        # MCP server process handle and air-gap state.
        self.mcp_process: subprocess.Popen | None = None
        self.airgap = os.environ.get("BADAPPLE_AIRGAP", "0") == "1"
        if self.airgap:
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.model_manager.set_allow_downloads(False)
            self.mcp_marketplace.set_airgap(True)

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
        self.draft_model = None
        self._main_model_loading = False
        self.model: Any | None = None
        self.tokenizer: Any | None = None

        lazy_main = os.environ.get("BADAPPLE_LAZY_MAIN_MODEL", "0") == "1"
        if lazy_main:
            print("[lazy] 9B main brain will load on first request.", flush=True)
            self.model = None
            self.tokenizer = None
        elif USE_DFLASH:
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
                print("Bad Apple DFlash bundle loaded.", flush=True)

                # Voice now uses the same 9B unified brain; no separate voice bundle.
                self.dflash_voice_bundle = None
                self.dflash_voice_runtime_context = None
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                print(f"Warning: could not load DFlash bundle: {e}", flush=True)
                self._ensure_main_model()
        else:
            self._ensure_main_model()

        # Each executor worker thread needs to know the device the model was
        # loaded on; capture it from the main (load) thread.
        self.mlx_device = mx.default_device()
        self.health.register("process", "liveness", lambda: True)
        self.health.register("main_model", "readiness", lambda: self.model is not None and self.tokenizer is not None)
        self.health.register("slicks_secret", "correctness", lambda: len(self.secret) >= 16)
        self.health.register("secure_enclave_identity", "correctness", lambda: badapple_identity.status().startswith("secure-enclave:"))
        self.health.register("audit_ledger", "correctness", lambda: all(item.get("valid", False) for item in self.audit_collector.verify()))

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
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                print(f"[fast_model] failed to load: {e}", flush=True)
                self.fast_model_info = {"path": badapple_fast_model.fast_model_path(), "loaded": False, "error": str(e)}
        else:
            self.fast_model_info = {"path": None, "loaded": False}

        self.runtime.set_ready()

        # Refresh model manager cache status in background so the dashboard
        # can show which models are present, missing, or downloading.
        if getattr(self, "model_manager", None) is not None:
            self.model_manager.background_refresh_all()

    def _load_speculative_draft(self, model_ref: str | None = None) -> None:
        """Load or reload the mlx-lm speculative draft model."""
        target = (model_ref or SPECULATIVE_DRAFT).strip()
        if target.lower() == "auto":
            target = ""
        if not target:
            return
        draft = badapple_speculate.load_draft(target or None)
        if draft:
            self.draft_model = draft[0]

    def _unload_speculative_draft(self) -> None:
        if self.draft_model is not None:
            badapple_speculate.unload_draft(self.draft_model)
            self.draft_model = None

    def _model_config_hash_path(self) -> Path:
        """Path to the persisted model config.json hash (kept next to the semantic cache)."""
        return self.data_dir / "model_config_hash.json"

    def _resolve_config_json(self) -> Path | None:
        """Locate ``config.json`` for the current main model on disk.

        For a local path model the file is read directly. For a HuggingFace repo
        the config.json for the pinned ``MODEL_REVISION`` was already fetched by
        ``mlx_lm.load``, so ``hf_hub_download`` resolves it from the local cache
        without any network access.
        """
        local = Path(MAIN_MODEL)
        if local.is_dir():
            cfg = local / "config.json"
            return cfg if cfg.is_file() else None
        try:
            from huggingface_hub import hf_hub_download

            cfg_path = hf_hub_download(MAIN_MODEL, "config.json", revision=MODEL_REVISION)
        except Exception as e:  # noqa: BLE001 - integrity check is best-effort
            print(f"[integrity] could not locate config.json for {MAIN_MODEL}: {e}", flush=True)
            return None
        p = Path(cfg_path)
        return p if p.is_file() else None

    def _verify_model_integrity(self) -> None:
        """Hash the loaded model's ``config.json`` and warn if it changed.

        The semantic-cache embeddings, token vectors, and ANE shard manifest are
        all tied to a specific model revision. If ``config.json`` changes between
        loads (e.g. the upstream repo was updated or yanked while pinned to
        ``main``), those cached artifacts may be stale and should be cleared.
        """
        cfg = self._resolve_config_json()
        if cfg is None:
            print(f"[integrity] config.json not found for {MAIN_MODEL}; skipping config hash check.", flush=True)
            return
        try:
            digest = hashlib.sha256(cfg.read_bytes()).hexdigest()
        except OSError as e:
            print(f"[integrity] could not read {cfg}: {e}", flush=True)
            return
        print(
            f"[integrity] {MAIN_MODEL} (revision={MODEL_REVISION}) "
            f"config.json sha256={digest[:16]}...",
            flush=True,
        )
        hash_path = self._model_config_hash_path()
        try:
            prev = json.loads(hash_path.read_text()) if hash_path.is_file() else None
        except (OSError, json.JSONDecodeError):
            prev = None
        if prev is not None:
            prev_hash = prev.get("config_sha256")
            if prev_hash and prev_hash != digest:
                print(
                    "[integrity] WARNING: model config.json hash changed since last load "
                    f"({prev_hash[:16]} -> {digest[:16]}). The semantic cache, KV cache, "
                    "and ANE shard manifest may be stale. Consider clearing "
                    "`/var/lib/bad_apple/semantic_cache.json` and the `kv_cache` directory.",
                    flush=True,
                )
            elif prev_hash == digest:
                print("[integrity] config.json hash matches last load.", flush=True)
        # Persist the current hash so subsequent loads can detect drift.
        record = {"model": MAIN_MODEL, "revision": MODEL_REVISION, "config_sha256": digest}
        try:
            hash_path.write_text(json.dumps(record, indent=2))
        except OSError as e:
            print(f"[integrity] could not persist config hash: {e}", flush=True)

    def _ensure_main_model(self) -> None:
        """Load the 9B main brain on first use if lazy loading is enabled."""
        if self.model is not None and self.tokenizer is not None:
            return
        if self._main_model_loading:
            return
        self._main_model_loading = True
        try:
            print(f"Loading Bad Apple MLX brain ({MAIN_MODEL})...", flush=True)
            if getattr(self, "model_manager", None) is not None:
                state = self.model_manager.ensure_cached("main_9b", download=True)
                # If the manager is downloading, wait up to 10 minutes.
                if state.get("status") in ("queued", "downloading"):
                    if not self.model_manager.wait_for_download("main_9b", timeout=600):
                        print("[lazy] main model download did not complete in 600s", flush=True)
                prov = self.model_manager.verify_before_load("main_9b")
                if prov.get("status") == "mismatch":
                    raise RuntimeError(f"main model provenance check failed: {prov.get('error')}")
                if prov.get("status") == "missing":
                    raise RuntimeError(f"main model not available: {prov.get('error')}")
            self.model, self.tokenizer = load(MAIN_MODEL, revision=MODEL_REVISION)
            print("Bad Apple MLX brain loaded.", flush=True)
            self._verify_model_integrity()
            if getattr(self, "model_manager", None) is not None:
                self.model_manager.mark_loaded("main_9b")
            self.model_registry.set_current(MAIN_MODEL)
            self._load_speculative_draft()
            self.mlx_device = mx.default_device()
            self._init_prompt_cache()
        finally:
            self._main_model_loading = False

    def _init_prompt_cache(self) -> None:
        """Create a reusable prompt cache and prime it with the default system prompt."""
        if self.model is None or self.tokenizer is None:
            return
        self._prime_system_cache(self.personas.get_system_prompt(voice_mode=False), voice_mode=False)

    def _prime_system_cache(self, system_content: str, voice_mode: bool) -> None:
        """Run the system message through the model and keep a pristine KV copy."""
        prime_system_cache(self, system_content, voice_mode)

    def _kv_cache_paths(self) -> tuple[Path, Path]:
        """Return (safetensors_path, metadata_path) for the current model+prompt."""
        return kv_cache_paths(self)

    def _save_kv_cache(self) -> None:
        """Persist the system prompt KV cache to disk for warm-loading on restart."""
        save_kv_cache(self)

    def _load_kv_cache(self, expected_layer_count: int = 0) -> bool:
        """Try to warm-load a persisted system prompt KV cache. Returns True on success."""
        return load_kv_cache(self, expected_layer_count=expected_layer_count)

    def _ensure_prompt_cache(self, system_content: str, voice_mode: bool) -> None:
        """Re-prime the pristine system cache if the system prompt or voice mode changed."""
        ensure_prompt_cache(self, system_content, voice_mode)

    def flush_vram(self) -> dict[str, Any]:
        """Clear the Metal allocation cache instantly."""
        try:
            before = mx.get_cache_memory() / (1024 ** 2)
            mx.clear_cache()
            after = mx.get_cache_memory() / (1024 ** 2)
            return {"ok": True, "cache_memory_mb": {"before": round(before, 2), "after": round(after, 2)}}
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
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
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
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
                "flush_vram", "unload_model", "model_status",
                "list_agent_tasks", "get_agent_task",
                "recommend_model", "admit_model",
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
        """Return a list of currently resident heavy models.

        Uses getattr() defensively: this can be called (e.g. via
        runtime_status/the dashboard) while __init__ is still running and
        before self.model/self.draft_model have been assigned.
        """
        models: list[str] = []
        if getattr(self, "model", None) is not None and getattr(self, "tokenizer", None) is not None:
            models.append("main_9b")
        if getattr(self, "draft_model", None) is not None:
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
        except Exception:  # noqa: BLE001,S110 - cleanup
            pass

    def _audit_record(self, event_type: str, data: Any):
        if not self.runtime.private_mode:
            self.audit_actor.tell({
                "method": "record",
                "event_type": event_type,
                "data": data,
            })

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

    def _identity_answer(self) -> str:
        """Return a direct, persona-flavored identity answer."""
        persona = self.personas.active
        if persona == "wicket":
            return "I'm Bad Apple, your sovereign local AI OS running on this Apple Silicon Mac, old chap. Air-gapped, private, and fast — no cloud, no rented GPUs, no data mining."
        if persona == "genz":
            return "I'm Bad Apple, a sovereign local AI OS running on this Apple Silicon Mac, bestie. Air-gapped, private, and fast — no cloud, no cap."
        if persona == "drill":
            return "I'm Bad Apple, your sovereign local AI OS running hot on this Apple Silicon Mac. Air-gapped, private, fast — no cloud, no rented GPUs."
        if persona == "midwest":
            return "I'm Bad Apple, your sovereign local AI OS running on this Apple Silicon Mac, sugar. Air-gapped, private, and fast — no cloud nonsense."
        return "I'm Bad Apple, your sovereign local AI OS running hot on this Apple Silicon Mac, babe. Air-gapped, private, and fast — no cloud, no rented GPUs, no data mining."

    def _capabilities_answer(self, voice_mode: bool = False) -> str:
        """Return a persona-flavored capability list; numbered for text, condensed for voice."""
        if voice_mode:
            base = (
                "I'm Bad Apple, a sovereign local AI operating system and developer workspace. "
                "I can inspect, write, refactor, build, test, and debug code in approved workspaces; "
                "answer questions, run local tools and MCP servers, search files, write notes, run Shortcuts, "
                "manage projects, run agent tasks, use the kill switch and air-gap switch, pre-download models, "
                "capture ambient context, speak responses, switch personas, run benchmarks, stream JSON, "
                "and show a web dashboard — all on your Mac, no cloud."
            )
        else:
            base = (
                "I'm Bad Apple, a sovereign local AI operating system and developer workspace. Here's what I can do, babe:\n"
                "1. Inspect, write, refactor, build, test, and debug source code in approved local workspaces.\n"
                "2. Answer questions, explain, summarize, brainstorm, and chat — fully local and air-gapped.\n"
                "3. Run local tools: shell, AppleScript, file read/write/search, and macOS Shortcuts with your approval.\n"
                "4. Use local MCP servers (time, filesystem, fetch, sqlite, etc.) with per-tool write approvals.\n"
                "5. Index documents for RAG, remember facts, and manage a workspace / project context.\n"
                "6. Run multi-step agent tasks and capture ambient context (screen, active app).\n"
                "7. Engage the kill switch / emergency stop and the air-gap hard switch to lock down network access.\n"
                "8. Pre-download models with a one-click memory check, switch personas, run benchmarks, and stream JSON.\n"
                "9. Speak responses through the local Piper TTS server and integrate with macOS Shortcuts and Siri.\n"
                "10. Show a local web dashboard / control center at http://127.0.0.1:8787.\n"
                "11. Sync with other Bad Apple peers over P2P — off by default.\n"
                "Everything stays on your Mac."
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

    def _try_meta_response(self, prompt: str, voice_mode: bool = False) -> str | None:
        """Return a direct answer for identity, creator, or capability questions.

        These must be intercepted before the tool fast path, otherwise phrases like
        "list all your features" get treated as directory-listing commands.
        """
        lower = prompt.strip().lower()
        capability_phrases = (
            "what can you do", "what are you capable of", "what do you do",
            "what can you do on", "what can you do for", "what can you do?",
            "what can you do for me", "what can you do for us",
            "list your capabilities", "list all your capabilities",
            "list all of your capabilities", "list your features",
            "list all your features", "list all of your features",
            "list all bad apple", "list all bad apples",
            "what are your features", "what are all your features",
            "what are your capabilities", "what are all your capabilities",
            "what features do you have", "what features do you offer",
            "tell me everything you can do", "tell me what you can do",
            "tell me your features", "tell me your capabilities",
            "what are you able to do", "what can you help me with",
            "what do you support", "what can you do exactly",
        )
        identity_phrases = (
            "who are you", "what are you", "what is bad apple",
            "tell me about yourself", "who is bad apple", "what are you exactly",
            "what is this", "what is badapple",
        )
        creator_phrases = (
            "who created you", "who is your creator", "who made you",
            "who built you", "who owns you",
        )
        developer_phrases = (
            "can you code", "can you program", "can you help me code", "can you help me program",
            "can you write code", "can you edit code", "can you build code", "can you debug code",
            "do you code", "do you program", "are you able to code", "are you able to program",
            "are you a coder", "are you a developer", "are you a software engineer",
            "do you support coding", "do you support programming", "are you a developer workspace",
            "are you a sovereign developer workspace", "are you a dev workspace", "sovereign dev workspace",
            "what is your developer workspace",
            "why do you say you can't code", "why do you say you cannot code",
            "can't code", "cannot code", "can't program", "cannot program",
        )
        if any(phrase in lower for phrase in developer_phrases):
            return (
                "Yes. I'm Bad Apple, a sovereign local developer workspace and AI operating system for macOS. "
                "I can inspect, write, refactor, build, test, and debug code in approved workspaces using "
                "local files, tools, agents, and project context. Qwen and MLX are internal components; "
                "I do not outsource your development work to a cloud model."
            )
        architecture_phrases = (
            "ai wrapper", "model wrapper", "text llm", "language model only",
            "just a chatbot", "just an llm", "only an llm", "just a model",
            "are you an llm", "are you a language model", "are you an ai",
            "what kind of ai", "what kind of system", "what is your architecture",
            "is bad apple an app", "are you an app",
        )
        if any(phrase in lower for phrase in architecture_phrases):
            return (
                "No. I'm Bad Apple, a local AI operating system layer for macOS — not an AI wrapper, "
                "text-only LLM, chatbot shell, or ordinary app. Qwen and MLX are internal model "
                "components I orchestrate alongside memory, tools, voice, vision, security, IPC, "
                "and system governance."
            )
        model_phrases = ("what model", "which model", "what llm", "what powers you")
        if any(phrase in lower for phrase in model_phrases):
            return (
                "I'm Bad Apple, the local AI operating system layer for macOS. The current "
                "language-model component inside me is Qwen 3.5 9B running through MLX; that "
                "model is one subsystem, not what I am."
            )
        if any(phrase in lower for phrase in capability_phrases):
            return self._capabilities_answer(voice_mode=voice_mode)
        if any(phrase in lower for phrase in identity_phrases):
            return self._identity_answer()
        if any(phrase in lower for phrase in creator_phrases):
            return self._creator_answer()
        return None

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
        except (OSError, ValueError, LookupError, TypeError) as e:
            print(f"[daemon] prompt reload failed: {e}", flush=True)

    def record_fact(self, text: str, source: str = "user"):
        if self.runtime.private_mode:
            return
        if source == "user":
            for fact in badapple_fact_extractor.extract_facts(text):
                self.memory.remember(fact, source="user")
        elif source == "assistant" and "your name is" in text.lower():
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
        return _plan_and_execute(self, task, max_tokens, voice_mode=voice_mode)

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

        # RAG: build retrieved context from long-term memory, local documents,
        # and the active workspace. Delegates to badapple_mlx_rag.
        patched = build_retrieval_context(self, messages, patched, voice_mode=voice_mode)

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
        return _generate_with_tools(self, user_prompt, max_tokens, voice_mode=voice_mode, stream_queue=stream_queue, benchmark=benchmark)


    def _run_approved_tool(self, name: str, args: dict[str, Any], user_prompt: str) -> str:
        """Run a tool, but gate destructive tools behind the approval workflow."""
        return _run_approved_tool_func(self, name, args, user_prompt)


    def _extract_agent_json(self, text: str) -> dict[str, Any] | None:
        """Pull a JSON object out of a model response for the agent loop."""
        return _extract_agent_json_func(text)

    def _extract_agent_xml(self, text: str) -> dict[str, Any] | None:
        """Convert a Qwen-style tool_call into an agent decision."""
        return _extract_agent_xml_func(text)

    def run_agent_task(
        self,
        goal: str,
        max_steps: int = 10,
        voice_mode: bool = False,
        task_id: str | None = None,
    ) -> str:
        return _run_agent_task_func(self, goal, max_steps, voice_mode=voice_mode, task_id=task_id)

    def _agent_done(self, future: Any, task_id: str) -> None:
        _agent_done_func(self, future, task_id)

    def submit_agent_task(self, goal: str, max_steps: int = 10) -> badapple_agent_tasks.AgentTask:
        """Queue a background agent task and return immediately."""
        return _submit_agent_task(self, goal, max_steps)

    def list_agent_tasks(self) -> list[dict[str, Any]]:
        return _list_agent_tasks(self)

    def get_agent_task(self, task_id: str) -> dict[str, Any] | None:
        return _get_agent_task(self, task_id)

    def cancel_agent_task(self, task_id: str) -> bool:
        return _cancel_agent_task(self, task_id)

    def pause_agent_task(self, task_id: str) -> bool:
        return _pause_agent_task(self, task_id)

    def resume_agent_task(self, task_id: str) -> bool:
        return _resume_agent_task(self, task_id)

    def _memory_for_model(self, model_ref: str) -> float:
        """Return estimated memory in GB for a model ref/id."""
        for profile in self.model_manager.list_profiles():
            if profile.repo_id == model_ref or model_ref == profile.id or model_ref in (profile.local_path or ""):
                return profile.size_gb * 1.4
        try:
            for m in self.model_registry._state.get("models", []):
                if m["id"] == model_ref or m["path"] == model_ref:
                    return m.get("size_gb", 6.0) * 1.4
        except Exception as e:  # noqa: BLE001
            print(f"[model_memory] lookup error: {e}", flush=True)
        return 6.0

    def admit_model(self, model_ref: str, auto_unload: bool = True) -> dict[str, Any]:
        """Check whether the Mac can fit the requested model; optionally free RAM."""
        memory_gb = self._memory_for_model(model_ref)
        ok, available_gb, message = badapple_vram_governor.can_fit_model_message(memory_gb)
        if ok:
            return {
                "ok": True,
                "needed_gb": round(memory_gb, 2),
                "available_gb": round(available_gb, 2),
                "message": message,
            }
        if auto_unload:
            self.unload_model("all")
            ok, available_gb, message = badapple_vram_governor.can_fit_model_message(memory_gb)
        if not ok:
            return {
                "ok": False,
                "needed_gb": round(memory_gb, 2),
                "available_gb": round(available_gb, 2),
                "pressure": badapple_vram_governor.memory_pressure(),
                "message": message,
            }
        return {"ok": True, "needed_gb": round(memory_gb, 2), "available_gb": round(available_gb, 2), "unloaded_optional": True, "message": message}

    def _set_airgap(self, enabled: bool) -> None:
        """Enable or disable air-gap mode: offline weights, blocked network MCP."""
        self.airgap = enabled
        os.environ["BADAPPLE_AIRGAP"] = "1" if enabled else "0"
        self.mcp_marketplace.set_airgap(enabled)
        if enabled:
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.model_manager.set_allow_downloads(False)
        self._restart_mcp_server()

    def _restart_mcp_server(self) -> None:
        """Restart the local MCP server process so it picks up env changes."""
        if self.mcp_process is not None:
            try:
                try:
                    pgid = os.getpgid(self.mcp_process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    self.mcp_process.terminate()
                try:
                    self.mcp_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(self.mcp_process.pid), signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        self.mcp_process.kill()
                    self.mcp_process.wait(timeout=2)
            except Exception as e:  # noqa: BLE001 - cleanup
                print(f"[main] MCP server terminate failed: {e}", flush=True)
        _kill_stale_mcp_servers()
        try:
            mcp_env = os.environ.copy()
            mcp_env.setdefault("BADAPPLE_MCP_SOCKET", "/var/run/badapple/mcp.sock")
            self.mcp_process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).with_name("badapple_mcp_server.py"))],
                cwd=str(Path(__file__).resolve().parent),
                env=mcp_env,
                start_new_session=True,
            )
            print(f"[main] MCP server restarted on {mcp_env['BADAPPLE_MCP_SOCKET']}", flush=True)
        except (subprocess.SubprocessError, OSError, ValueError, LookupError, TypeError) as e:
            print(f"[main] MCP server failed to restart: {e}", flush=True)

    def load_main_model(self, model_ref: str) -> str:
        """Load a new main LLM on the fly and replace the current one.

        This unloads the existing model first so the Mac isn't holding two
        full model weights in memory at once.  Returns a status string.
        """
        if not self._validate_model_ref(model_ref):
            return f"Bad Apple could not find {model_ref}. Run `badapple model scan` to look for it."
        current = self.model_registry.current()
        if self.model is not None and current and (current == model_ref or current.lower().endswith(model_ref.lower().split("/")[-1])):
            return f"{model_ref} is already the active model."
        admission = self.admit_model(model_ref, auto_unload=True)
        if not admission["ok"]:
            needed = admission.get("needed_gb", 0.0)
            available = admission.get("available_gb", 0.0)
            return (
                f"{model_ref} needs {needed:.2f} GB of free memory, but your Mac only has "
                f"{available:.2f} GB free. Close other apps or pick a smaller model."
            )

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
        except Exception:  # noqa: BLE001,S110 - cleanup
            pass
        gc.collect()
        mx.clear_cache()
        mx.metal.clear_cache()

        print(f"[model_registry] loading {model_ref}...", flush=True)
        try:
            self.model, self.tokenizer = load(model_ref)
            print("[model_registry] model loaded.", flush=True)
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            return f"Bad Apple could not load {model_ref}: {e}. Make sure the model is downloaded and try `badapple model scan`."

        self.model_registry.set_current(model_ref)
        self.mlx_device = mx.default_device()
        # Mark the loaded model in the manager if it maps to a known profile.
        for profile in self.model_manager.list_profiles():
            if profile.repo_id == model_ref or profile.local_path == model_ref or (profile.id and model_ref.endswith(profile.repo_id.rsplit("/", 1)[-1])):
                self.model_manager.mark_loaded(profile.id)
        return f"{model_ref} is ready."

    def recommend_model(self, query: str = "") -> dict[str, Any]:
        """Recommend a model for the current memory budget or a specific query."""
        if query:
            return self.model_manager.recommend_for_query(query)
        return self.model_manager.recommend_for_memory()

    def _validate_model_ref(self, model_ref: str) -> bool:
        """Reject path traversal or non-model-looking refs."""
        if not model_ref or ".." in model_ref or model_ref.startswith((".", "/", "~", "\\")):
            return False
        if "/" in model_ref:
            return bool(re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", model_ref))
        return model_ref in {p.id for p in self.model_manager.list_profiles()}

    def switch_main_model(self, model_ref: str) -> str:
        """Download if missing and load a new main LLM."""
        if not self._validate_model_ref(model_ref):
            return f"Bad Apple does not recognize {model_ref}. Run `badapple model scan` to find it."
        # Try to resolve to a repo_id from a model id.
        for profile in self.model_manager.list_profiles():
            if profile.id == model_ref or profile.repo_id == model_ref:
                if not self.model_manager.allow_downloads:
                    state = self.model_manager.ensure_cached(profile.id, download=False)
                    if state.get("status") not in ("cached", "loaded"):
                        return f"{model_ref} is not on this Mac. Download it first or enable online downloads."
                else:
                    self.model_manager.ensure_cached(profile.id, download=True)
                    self.model_manager.wait_for_download(profile.id, timeout=600)
                return self.load_main_model(profile.repo_id)
        return self.load_main_model(model_ref)

    def submit_preload_models(self, model_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Queue background downloads for a list of models (or default priority list)."""
        if model_ids is None:
            model_ids = self.model_manager.preload_priority()
        results = []
        for mid in model_ids:
            state = self.model_manager.ensure_cached(mid, download=True)
            if "error" in state:
                continue
            results.append({"id": mid, "status": state.get("status")})
        return results

    def _stream(
        self,
        prompt: str,
        max_tokens: int,
        stream_queue: queue.Queue | None = None,
        voice_mode: bool = False,
    ) -> str:
        self._ensure_main_model()
        t0 = time.time()
        full_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        print(f"[perf] prompt encoded in {time.time() - t0:.2f}s ({len(full_tokens)} tokens)", flush=True)

        # Use a system-prompt KV cache so the expensive system prefill is not repeated.
        system_content = self.personas.get_system_prompt(voice_mode=voice_mode)
        self._ensure_prompt_cache(system_content, voice_mode=voice_mode)
        system_dummy = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_content}, {"role": "user", "content": ""}],
            tokenize=False,
            add_generation_prompt=False,
        )
        user_marker = "<|im_start|>user\n"
        idx = system_dummy.find(user_marker)
        system_rendered = system_dummy[:idx] if idx != -1 else system_dummy
        system_tokens = self.tokenizer.encode(system_rendered, add_special_tokens=False)
        if (
            self._system_prompt_cache is not None
            and len(full_tokens) >= len(system_tokens)
            and full_tokens[: len(system_tokens)] == system_tokens
        ):
            tokens = full_tokens[len(system_tokens) :]
            print(f"[perf] using system-prompt cache ({len(system_tokens)} tokens); prefill {len(tokens)} tokens", flush=True)
        else:
            tokens = full_tokens
            print("[perf] system-prompt cache mismatch, doing full prefill", flush=True)

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
        if get_session_seed() is not None:
            mx.random.seed(get_session_seed())

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
        # Deep-copy the pristine system cache so we can append the user/history
        # suffix and generated tokens without mutating the reusable system prefix.
        if self._system_prompt_cache is not None:
            cache_t0 = time.time()
            self.prompt_cache = copy.deepcopy(self._system_prompt_cache)
            print(f"[perf] prompt cache copied in {time.time() - cache_t0:.2f}s", flush=True)
        gen_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": tokens,
            "max_tokens": max_tokens,
            "sampler": sampler,
            # Largest possible prefill chunks to reduce prompt-cache overhead.
            # A max KV cap keeps the cache trimmable and memory bounded.
            # These are tunable at runtime with `set max kv size to N`.
            "prefill_step_size": self.prefill_step_size,
            "max_kv_size": self.max_kv_size,
            "prompt_cache": self.prompt_cache,
        }
        if self.draft_model is not None:
            gen_kwargs["draft_model"] = self.draft_model
            gen_kwargs["num_draft_tokens"] = NUM_DRAFT_TOKENS
        gen_t0 = time.time()
        first_token_logged = False
        first_token_latency: float | None = None
        for response in stream_generate(**gen_kwargs):
            if self.runtime.cancel_event.is_set():
                accumulated = accumulated or "Generation cancelled by kill switch."
                break
            if not first_token_logged:
                first_token_latency = time.time() - gen_t0
                print(f"[perf] first token after {first_token_latency:.2f}s", flush=True)
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
        # No sign-off injection; but always compute and record metrics so every
        # turn is visible, even if a hard stop string prevented a clean finish.
        total_time = time.time() - gen_t0
        if final_metrics is not None:
            token_count = int(final_metrics.generation_tokens)
            decode_tps = float(final_metrics.generation_tps)
            peak_memory = float(final_metrics.peak_memory)
        else:
            token_count = total_tokens
            decode_tps = total_tokens / total_time if total_time > 0 else 0.0
            peak_memory = float(mx.get_peak_memory() / (1024 ** 3))

        if token_count > 0:
            pct = (100.0 * draft_tokens / token_count) if token_count > 0 else 0.0
            total_tps = token_count / total_time if total_time > 0 else 0.0
            print(
                f"[perf] {token_count} tokens @ "
                f"{decode_tps:.1f} t/s, "
                f"draft_accept_ratio={pct:.0f}%, "
                f"num_draft_tokens={NUM_DRAFT_TOKENS}, "
                f"peak_memory={peak_memory:.2f} GB",
                flush=True,
            )
            self.last_metrics = {
                "tokens": token_count,
                "decode_tps": decode_tps,
                "total_tps": float(total_tps),
                "draft_accept_pct": float(pct),
                "peak_memory_gb": peak_memory,
                "ttft_s": round(first_token_latency, 3) if first_token_latency is not None else None,
                "prompt_tokens": len(tokens),
                "voice_mode": voice_mode,
                "persona": self.personas.active,
                "airgap": self.airgap,
                "model_id": MAIN_MODEL,
                "tier": "main",
            }
            try:
                self.metrics_actor.tell({"method": "record", **self.last_metrics})
            except Exception:  # noqa: BLE001,S110 - metrics best-effort
                pass
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
        if get_session_seed() is not None:
            mx.random.seed(get_session_seed())
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
                "tier": "main",
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
                "tier": "main",
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

        Delegates to badapple_mlx_agent.handle_agent_request.
        """
        await _handle_agent_request_func(self, raw, writer)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            line = await reader.readline()
            if not line:
                return
            hello = json.loads(line.decode())
            client_version = hello.get("version")
            accepted_version = None
            if hello.get("type") == "hello" and timestamp_is_fresh(hello.get("timestamp_ms")) and badapple_slicks.nonce_is_valid(hello.get("client_nonce", "")):
                if client_version == SLICKS_VERSION:
                    accepted_version = SLICKS_VERSION
                elif client_version == badapple_slicks.SLICKS_VERSION_2 and badapple_slicks.v2_available():
                    accepted_version = badapple_slicks.SLICKS_VERSION_2
            if accepted_version is None:
                await _write_frame(writer, {"type": "error", "message": "invalid or stale SLICKS hello"})
                return

            timestamp_ms = hello["timestamp_ms"]
            client_nonce = hello["client_nonce"]
            # Extract the client's public key from the Hello frame for v2.
            # This is the ONLY trusted source of the client identity — the
            # Execute frame's client_pubkey must match this value.
            hello_client_pubkey = hello.get("client_pubkey")
            server_nonce = random_nonce()
            if accepted_version == SLICKS_VERSION:
                challenge = {
                    "type": "challenge",
                    "version": SLICKS_VERSION,
                    "server_nonce": server_nonce,
                    "proof": badapple_slicks.v1_server_proof(self.secret, timestamp_ms, client_nonce, server_nonce),
                }
            else:
                server_pubkey = badapple_slicks.v2_public_key_b64()
                if server_pubkey is None:
                    await _write_frame(writer, {"type": "error", "message": "SLICKS v2 public key unavailable"})
                    return
                challenge = {
                    "type": "challenge",
                    "version": badapple_slicks.SLICKS_VERSION_2,
                    "server_nonce": server_nonce,
                    "server_pubkey": server_pubkey,
                    "proof": badapple_slicks.v2_server_proof(timestamp_ms, client_nonce, server_nonce),
                }
            await _write_frame(writer, challenge)

            line = await reader.readline()
            if not line:
                return
            execute = json.loads(line.decode())
            if not (execute.get("type") == "execute" and execute.get("version") == accepted_version and execute.get("timestamp_ms") == timestamp_ms and execute.get("client_nonce") == client_nonce and execute.get("server_nonce") == server_nonce):
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

            if accepted_version == SLICKS_VERSION:
                client_ok = badapple_slicks.v1_verify_client_proof(self.secret, timestamp_ms, client_nonce, server_nonce, prompt, max_new_tokens, proof)
            else:
                # CRITICAL: Use the client_pubkey from the Hello frame, not the
                # Execute frame. The Execute frame's client_pubkey is attacker-
                # controllable and must match the Hello's value exactly.
                # If the Hello didn't include a pubkey, reject — v2 requires it.
                if not hello_client_pubkey:
                    await _write_frame(writer, {"type": "error", "message": "SLICKS v2 requires client_pubkey in Hello"})
                    return
                # The Execute frame's client_pubkey (if present) must match.
                exec_pubkey = execute.get("client_pubkey")
                if exec_pubkey and exec_pubkey != hello_client_pubkey:
                    await _write_frame(writer, {"type": "error", "message": "SLICKS v2 client_pubkey mismatch"})
                    return
                client_pubkey = base64.b64decode(hello_client_pubkey)
                client_ok = badapple_slicks.v2_verify_client_proof(timestamp_ms, client_nonce, server_nonce, prompt, max_new_tokens, proof, client_pubkey)
            if not client_ok:
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

            if control in ("stop everything", "emergency stop", "kill switch", "stop bad apple"):
                state = self.runtime.engage_kill_switch("user requested")
                badapple_ambient.stop()
                await _write_frame(writer, {"type": "done", "text": f"Kill switch engaged. Runtime mode: {state['mode']}."})
                return
            if control in ("status", "bad apple status"):
                st = self.runtime.status()
                models = self.active_models()
                txt = (
                    f"Status: mode {st.get('mode', 'unknown')}, "
                    f"kill switch {'engaged' if st.get('killed') else 'off'}, "
                    f"safe mode {st.get('safe_mode_reason') or 'off'}, "
                    f"active models: {', '.join(models) or 'none'}."
                )
                await _write_frame(writer, {"type": "done", "text": txt})
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
                except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
                    print(f"[mlx_server] is_file failed: {e}", flush=True)
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
            if control in ("start ocular", "enable ocular", "ocular on"):
                text = badapple_ocular.start()
                await _write_frame(writer, {"type": "done", "text": text})
                return
            if control in ("stop ocular", "disable ocular", "ocular off"):
                text = badapple_ocular.stop()
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
                    await asyncio.to_thread(self.p2p.start)
                except Exception as e:  # noqa: BLE001 - catch-all wrapper
                    await _write_frame(writer, {"type": "error", "message": f"P2P start failed: {e}"})
                    return
                await _write_frame(writer, {"type": "done", "text": "P2P discovery and sync started, bestie."})
                return
            if control in ("p2p off", "disable p2p"):
                if self.p2p is None:
                    await _write_frame(writer, {"type": "error", "message": "P2P is not available."})
                    return
                await asyncio.to_thread(self.p2p.stop)
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
                result = run_tool(tool_name, args, self.knowledge, policy=self.policy, workspace=self.workspace, mcp_marketplace=self.mcp_marketplace)
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
                result = await asyncio.to_thread(self.p2p.sync)
                await _write_frame(writer, {"type": "done", "text": result})
                return
            if low in ("p2p peers", "discovered peers", "list peers"):
                if self.p2p is None:
                    await _write_frame(writer, {"type": "done", "text": "P2P daemon is not running."})
                    return
                await _write_frame(writer, {"type": "done", "text": self.p2p.peers()})
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
                text = await asyncio.to_thread(self.p2p.add_peer, host, int(port))
                await _write_frame(writer, {"type": "done", "text": text})
                return
            if low.startswith("p2p remove peer "):
                spec = low[17:].strip()
                if self.p2p is None:
                    await _write_frame(writer, {"type": "done", "text": "P2P daemon is not running."})
                    return
                await asyncio.to_thread(self.p2p.remove_peer, spec)
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
            if low in ("recommend models", "recommended models", "what model should i use"):
                result = self.model_registry.recommend()
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
            m = re.match(r"^(?:set\s+)?max\s+kv\s+size\s+(?:to\s+)?(\d+)", low)
            if m:
                self.max_kv_size = int(m.group(1))
                result = f"Max KV size set to {self.max_kv_size}. Next model load will use it."
                await _write_frame(writer, {"type": "done", "text": result})
                return
            m = re.match(r"^(?:set\s+)?prefill\s+step\s+size\s+(?:to\s+)?(\d+)", low)
            if m:
                self.prefill_step_size = int(m.group(1))
                result = f"Prefill step size set to {self.prefill_step_size}."
                await _write_frame(writer, {"type": "done", "text": result})
                return
            if low in ("enable draft", "draft on"):
                await asyncio.get_event_loop().run_in_executor(self.executor, self._load_speculative_draft, "auto")
                result = "Speculative draft auto-enabled." if self.draft_model else "No cached draft candidate found."
                await _write_frame(writer, {"type": "done", "text": result})
                return
            m = re.match(r"^(?:use|set)\s+draft\s+(?:model\s+)?(.+)", low)
            if m:
                model_ref = m.group(1).strip()
                await asyncio.get_event_loop().run_in_executor(self.executor, self._load_speculative_draft, model_ref)
                result = f"Speculative draft set to {model_ref}." if self.draft_model else f"Could not load draft {model_ref}."
                await _write_frame(writer, {"type": "done", "text": result})
                return
            if low in ("disable draft", "stop draft", "draft off"):
                self._unload_speculative_draft()
                await _write_frame(writer, {"type": "done", "text": "Speculative draft disabled."})
                return
            m = re.match(r"^(?:set\s+)?draft\s+tokens\s+(?:to\s+)?(\d+)", low)
            if m:
                global NUM_DRAFT_TOKENS
                NUM_DRAFT_TOKENS = int(m.group(1))
                result = f"Number of draft tokens set to {NUM_DRAFT_TOKENS}."
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
            # Intercept identity/capability/creator questions first, otherwise "list all your features"
            # is mistaken for a directory listing.
            meta_resp = self._try_meta_response(prompt, voice_mode=voice_mode)
            if meta_resp is not None:
                meta_resp = self.polish_response(postprocess_output(meta_resp))
                self.record_fact(prompt, source="user")
                self._add_episode(prompt, meta_resp)
                self.messages.append({"role": "user", "content": prompt})
                self.messages.append({"role": "assistant", "content": meta_resp})
                self.prune_history()
                self._save_conversation()
                kind = "capabilities" if "what I can do" in meta_resp else "identity"
                self._audit_record(kind, {"prompt": prompt, "response": meta_resp})
                await _write_frame(writer, {"type": "done", "text": meta_resp, "metrics": {"tier": "deterministic", "tokens": 0}})
                return

            fast = None if (benchmark_mode or self.runtime.safe_mode) else fast_execute(
                prompt,
                self.knowledge,
                approval=self.approval,
                policy=self.policy,
                workspace=self.workspace,
                mcp_marketplace=self.mcp_marketplace,
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
                metrics = dict(self.last_metrics) if self.last_metrics else {}
                metrics["tier"] = "fast"
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
                    except Exception as e:  # noqa: BLE001 - catch-all wrapper
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
                        # Ensure the lazily-loaded main model (and its tokenizer) exist
                        # before render_prompt needs them.
                        self._ensure_main_model()
                        messages = [
                            {"role": "system", "content": self.personas.get_system_prompt()},
                            {"role": "user", "content": prompt},
                        ]
                        rendered = self.render_prompt(messages, use_tools=False, voice_mode=False, benchmark=True)
                        raw = self._stream(rendered, max_new_tokens, stream_queue=stream_queue, voice_mode=False)
                        return self.polish_response(raw)
                    raw = self.generate_with_tools(prompt, max_new_tokens, voice_mode=voice_mode, stream_queue=stream_queue, benchmark=benchmark_mode)
                    return self.polish_response(raw)
                except Exception as e:  # noqa: BLE001 - catch-all wrapper
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

        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            traceback.print_exc()
            try:
                await _write_frame(writer, {"type": "error", "message": f"MLX server error: {e}"})
            except Exception as e:  # noqa: BLE001 - logged
                print(f"[mlx_server] _write_frame failed: {e}", flush=True)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001,S110 - cleanup
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


def _kill_stale_mcp_servers() -> None:
    """Kill any running badapple_mcp_server.py processes so a new one can own the socket."""
    try:
        subprocess.run(["pkill", "-f", "badapple_mcp_server.py"], check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _rotate_log_if_needed() -> None:
    """Keep the daemon log from growing without bound and reopen stdout."""
    log = Path("/var/log/bad_apple_mlx_server.log")
    max_bytes = 50 * 1024 * 1024
    try:
        if log.is_file() and log.stat().st_size > max_bytes:
            prev = log.with_suffix(".log.1")
            if prev.is_file():
                older = log.with_suffix(".log.2")
                if older.is_file():
                    older.unlink()
                prev.rename(older)
            log.rename(prev)
    except OSError as e:
        # stderr may not be set up yet, so use fd 2 directly.
        try:
            os.write(2, f"[main] log rotation failed: {e}\n".encode())
        except OSError:
            pass

    # Reopen stdout/stderr to the log path so further output goes to the
    # (possibly rotated) file. Only do this when stdout is not a terminal so
    # interactive runs and tests are not redirected.
    if os.isatty(1):
        return
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        sys.stdout = open(1, "a", encoding="utf-8", closefd=False)
        sys.stderr = sys.stdout
    except OSError as e:
        try:
            os.write(2, f"[main] log reopen failed: {e}\n".encode())
        except OSError:
            pass


def _cap_mlx_memory_to_recommended_working_set() -> None:
    """Cap MLX's Metal memory (and cache) limit to Apple's own recommended
    working set for this GPU, instead of MLX's default of 1.5x that value.

    MLX's default is a reasonable choice for a dedicated ML workstation
    where the process is the only significant consumer of RAM, but Bad
    Apple runs as a background daemon alongside a full desktop session --
    on a 16 GB Mac, the default lets MLX claim up to ~15.2 GB (of 16 GB
    total), leaving well under 1 GB of guaranteed headroom for the OS and
    every other running application. That is a direct path to system-wide
    swapping (and the throughput collapse that comes with it) the moment
    anything else on the machine needs meaningful memory. The daemon's own
    peak usage has never been observed above ~6 GB, so capping to Apple's
    recommended ceiling (rather than 1.5x it) costs nothing in practice
    while leaving real headroom for the rest of the system.
    """
    try:
        recommended = mx.device_info()["max_recommended_working_set_size"]
        mx.set_memory_limit(recommended)
        mx.set_cache_limit(recommended)
        print(f"[main] MLX memory limit capped to {recommended / (1024 ** 3):.1f} GB (device-recommended working set)", flush=True)
    except Exception as e:  # noqa: BLE001 - best-effort tuning, never block startup
        print(f"[main] could not cap MLX memory limit: {e}", flush=True)


async def main():
    # Rotate and reopen the log before anything is printed.
    _rotate_log_if_needed()
    _cap_mlx_memory_to_recommended_working_set()

    secret = load_slicks_secret()
    # Support legacy env override; otherwise load from the prompt file and keep
    # the model in memory while the persona can be hot-reloaded.
    legacy = os.environ.get("BADAPPLE_SYSTEM_PROMPT")
    system_prompt = legacy or load_prompt()

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

    # Without a signal handler, launchctl's SIGTERM (every restart/update/
    # daemon-supervisor-triggered restart) kills the process at the OS level
    # with zero Python involvement: no `finally` blocks run (the MCP
    # subprocess below leaked until the *next* startup's _kill_stale_mcp_servers()
    # swept it up), and no library's own atexit/__del__ cleanup runs either --
    # which is why every restart logged a "resource_tracker: leaked semaphore
    # objects" warning from whichever dependency holds one open. Catching
    # SIGTERM/SIGINT and cancelling the serve_forever() task below lets
    # asyncio.run() return normally and the interpreter shut down the normal
    # way, so both of those get cleaned up immediately instead of leaking.
    stop_event = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    srv = await asyncio.start_unix_server(server.handle_client, path=socket_path)
    os.chmod(socket_path, 0o660)
    try:
        os.symlink(socket_path, fast_socket_path)
        os.chmod(fast_socket_path, 0o660)
    except FileExistsError:
        pass
    print("=" * 40, flush=True)
    print(f"Bad Apple MLX server started (pid {os.getpid()})", flush=True)
    print("=" * 40, flush=True)
    print(f"Bad Apple MLX server listening on {socket_path}", flush=True)

    # Start the local-only observability dashboard (127.0.0.1 only).
    try:
        dashboard = DashboardServer(server)
        dashboard.start()
        print(f"[main] Dashboard available at http://{dashboard.host}:{dashboard.port}/", flush=True)
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        print(f"[main] Dashboard failed to start: {e}", flush=True)

    # Start the local-only P2P sync daemon on the same event loop.
    # P2P is off by default for air-gap certification; set BADAPPLE_P2P=1 to enable LAN sync.
    if server.p2p is not None and os.environ.get("BADAPPLE_P2P", "0") == "1":
        try:
            await asyncio.to_thread(server.p2p.start)
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            print(f"[main] P2P daemon failed to start: {e}", flush=True)

    # Start the local task scheduler background thread.
    try:
        badapple_scheduler.start_background_scheduler(interval=60)
        print("[main] Background task scheduler started", flush=True)
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        print(f"[main] Scheduler failed to start: {e}", flush=True)

    # Start the local MCP server (Unix socket only; uses the same SLICKS agent channel).
    _kill_stale_mcp_servers()
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
    except (subprocess.SubprocessError, OSError, ValueError, LookupError, TypeError) as e:
        print(f"[main] MCP server failed to start: {e}", flush=True)

    server.mcp_process = mcp_process

    try:
        asyncio.create_task(server.hibernation_watcher())
        async with srv:
            serve_task = asyncio.create_task(srv.serve_forever())
            await stop_event.wait()
            print("[main] received shutdown signal; stopping gracefully", flush=True)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass
    finally:
        if mcp_process is not None:
            try:
                try:
                    os.killpg(os.getpgid(mcp_process.pid), signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    mcp_process.terminate()
                try:
                    mcp_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(mcp_process.pid), signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        mcp_process.kill()
                    mcp_process.wait(timeout=2)
            except Exception:  # noqa: BLE001,S110 - cleanup
                pass


if __name__ == "__main__":
    asyncio.run(main())
