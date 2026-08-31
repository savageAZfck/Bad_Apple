"""Bad Apple MLX server — RAG context building and KV prompt-cache management.

Extracted from badapple_mlx_server.py. Builds retrieved context from long-term
memory, local documents (knowledge base), and the active workspace to inject
into the model prompt. Also manages the persistent system-prompt KV cache so
the expensive system prefill is only paid once across daemon restarts.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


def build_retrieval_context(
    server: Any,
    messages: list[dict[str, str]],
    patched: list[dict[str, str]],
    voice_mode: bool = False,
) -> list[dict[str, str]]:
    """Augment *patched* messages with retrieved memory, documents, and workspace context.

    This is the RAG retrieval portion of ``MLXServer.render_prompt``.  It mutates
    and returns the *patched* message list by inserting relevant context blocks
    before the final user message.
    """
    # Tight context fetches to keep prompt tokens and prefill latency low.
    rel_mem = [] if server.runtime.private_mode else server.memory.search(messages[-1]["content"], k=1)

    # If a user-fact is already remembered, answer from that instead of
    # getting distracted by unrelated documents.
    rel_know: list = []
    if not rel_mem and not voice_mode:
        rel_know = server.knowledge.search(messages[-1]["content"], k=1, threshold=0.92)

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
    if server.workspace.path:
        ws_text = f"Active workspace:\n{server.workspace.summary()}"
        patched.insert(-1, {
            "role": "user",
            "content": f"Use this project context if relevant:\n\n{ws_text}",
        })

    project_ctx = server.memory.get_project_context()
    if project_ctx:
        patched.insert(-1, {
            "role": "user",
            "content": f"Long-horizon project context:\n\n{project_ctx}",
        })

    return patched


# ---------------------------------------------------------------------------
# System-prompt KV cache management (warm-load / persist across restarts)
# ---------------------------------------------------------------------------

def prime_system_cache(server: Any, system_content: str, voice_mode: bool) -> None:
    """Run the system message through the model and keep a pristine KV copy.

    This populates ``server._system_prompt_cache`` once.  ``MLXServer._stream``
    deep-copies it for each query, so the expensive system prefill is paid once
    on model load (or when the system prompt changes), not after every response.

    If a persisted KV cache matching the current model + system prompt exists
    on disk, it is warm-loaded instead of re-prefilling.
    """
    if server.model is None or server.tokenizer is None:
        return
    t0 = time.time()
    # The model's chat template requires at least a user message, so render
    # a dummy one and keep only the system-message prefix.
    rendered_dummy = server.tokenizer.apply_chat_template(
        [{"role": "system", "content": system_content}, {"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=False,
    )
    user_marker = "<|im_start|>user\n"
    idx = rendered_dummy.find(user_marker)
    if idx == -1:
        rendered = rendered_dummy
    else:
        rendered = rendered_dummy[:idx]
    tokens = server.tokenizer.encode(rendered, add_special_tokens=False)
    server._cache_system_hash = f"{voice_mode}:{hashlib.sha256(rendered.encode()).hexdigest()[:16]}"

    # Try warm-loading a persisted cache before paying the prefill cost.
    # Create a throwaway cache to learn the expected layer count for validation.
    _probe = make_prompt_cache(server.model, max_kv_size=server.max_kv_size)
    expected_layer_count = len(_probe)
    del _probe
    if load_kv_cache(server, expected_layer_count=expected_layer_count):
        print(f"[perf] system prompt cache warm-loaded ({len(tokens)} tokens) in {time.time() - t0:.2f}s", flush=True)
        return

    try:
        server._system_prompt_cache = make_prompt_cache(server.model, max_kv_size=server.max_kv_size)
        _ = server.model(mx.array(tokens)[None], cache=server._system_prompt_cache)
        mx.eval([c.state for c in server._system_prompt_cache])
        mx.clear_cache()
        print(f"[perf] system prompt cache primed ({len(tokens)} tokens) in {time.time() - t0:.2f}s", flush=True)
    except Exception as e:  # noqa: BLE001 - cache priming is best-effort
        print(f"[main] system prompt cache priming failed: {e}", flush=True)
        server._cache_system_hash = None
        server._system_prompt_cache = None
    else:
        # Persist the freshly primed cache so the next daemon restart
        # can warm-load it instead of re-prefilling the system prompt.
        save_kv_cache(server)


def kv_cache_paths(server: Any) -> tuple[Path, Path]:
    """Return (safetensors_path, metadata_path) for the current model+prompt."""
    from hashlib import sha256
    model_ref = os.environ.get("BADAPPLE_MAIN_MODEL", "unknown")
    key = f"{model_ref}:{server.max_kv_size}:{server._cache_system_hash}"
    digest = sha256(key.encode()).hexdigest()[:16]
    server._kv_cache_dir.mkdir(parents=True, exist_ok=True)
    return server._kv_cache_dir / f"sys_{digest}.safetensors", server._kv_cache_dir / f"sys_{digest}.json"


def save_kv_cache(server: Any) -> None:
    """Persist the system prompt KV cache to disk for warm-loading on restart."""
    if server._system_prompt_cache is None or server._cache_system_hash is None:
        return
    try:
        arrays: dict[str, mx.array] = {}
        metadata: list[dict[str, Any]] = []
        for i, c in enumerate(server._system_prompt_cache):
            ctype = type(c).__name__
            state = c.state
            meta = c.meta_state
            if ctype == "ArraysCache":
                # state is a list of arrays (or None); save each non-None entry.
                cache_list = state if isinstance(state, list) else list(state)
                none_indices = []
                for j, arr in enumerate(cache_list):
                    if arr is not None:
                        arrays[f"layer_{i}_arr_{j}"] = arr
                    else:
                        none_indices.append(j)
                metadata.append({
                    "type": ctype,
                    "meta_state": list(meta) if not isinstance(meta, str) else meta,
                    "cache_size": len(cache_list),
                    "none_indices": none_indices,
                })
            else:
                # KVCache / RotatingKVCache: state is (keys, values).
                k, v = state
                arrays[f"layer_{i}_keys"] = k
                arrays[f"layer_{i}_values"] = v
                metadata.append({
                    "type": ctype,
                    "meta_state": list(meta),
                })
        weights_path, meta_path = kv_cache_paths(server)
        # mx.save_safetensors appends .safetensors if not already present.
        tmp_w = weights_path.with_name(weights_path.stem + ".tmp.safetensors")
        tmp_m = meta_path.with_name(meta_path.stem + ".tmp.json")
        mx.save_safetensors(str(tmp_w), arrays)
        tmp_m.write_text(json.dumps({
            "model": os.environ.get("BADAPPLE_MAIN_MODEL", ""),
            "max_kv_size": server.max_kv_size,
            "system_hash": server._cache_system_hash,
            "num_layers": len(server._system_prompt_cache),
            "layers": metadata,
        }), encoding="utf-8")
        tmp_w.replace(weights_path)
        tmp_m.replace(meta_path)
        print(f"[kv] system prompt cache saved to {weights_path.name}", flush=True)
    except Exception as e:  # noqa: BLE001 - persistence is best-effort
        print(f"[kv] failed to save system cache: {e}", flush=True)


def load_kv_cache(server: Any, expected_layer_count: int = 0) -> bool:
    """Try to warm-load a persisted system prompt KV cache. Returns True on success."""
    if server._cache_system_hash is None or server.model is None:
        return False
    try:
        weights_path, meta_path = kv_cache_paths(server)
        if not weights_path.is_file() or not meta_path.is_file():
            return False

        from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
        _cache_types = {"ArraysCache": ArraysCache, "RotatingKVCache": RotatingKVCache, "KVCache": KVCache}
        meta_doc = json.loads(meta_path.read_text(encoding="utf-8"))
        # Validate the persisted cache matches the current model architecture.
        if expected_layer_count and meta_doc.get("num_layers") != expected_layer_count:
            return False
        if meta_doc.get("max_kv_size") != server.max_kv_size:
            return False
        loaded = mx.load(str(weights_path))
        reconstructed: list[Any] = []
        for i, layer_meta in enumerate(meta_doc["layers"]):
            ctype = layer_meta["type"]
            cls = _cache_types.get(ctype, KVCache)
            if ctype == "ArraysCache":
                cache_size = layer_meta.get("cache_size", 0)
                none_indices = set(layer_meta.get("none_indices", []))
                cache_list: list[Any] = []
                for j in range(cache_size):
                    if j in none_indices:
                        cache_list.append(None)
                    else:
                        cache_list.append(loaded[f"layer_{i}_arr_{j}"])
                meta_state = layer_meta["meta_state"]
                obj = cls.from_state(cache_list, meta_state)
            else:
                state = (loaded[f"layer_{i}_keys"], loaded[f"layer_{i}_values"])
                meta_state = tuple(layer_meta["meta_state"])
                obj = cls.from_state(state, meta_state)
            reconstructed.append(obj)
        server._system_prompt_cache = reconstructed
        print(f"[kv] system prompt cache warm-loaded from {weights_path.name}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001 - loading is best-effort
        print(f"[kv] failed to warm-load system cache: {e}", flush=True)
        return False


def ensure_prompt_cache(server: Any, system_content: str, voice_mode: bool) -> None:
    """Re-prime the pristine system cache if the system prompt or voice mode changed."""
    # Compute the same rendered prefix used in prime_system_cache to compare hashes.
    rendered_dummy = server.tokenizer.apply_chat_template(
        [{"role": "system", "content": system_content}, {"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=False,
    )
    user_marker = "<|im_start|>user\n"
    idx = rendered_dummy.find(user_marker)
    rendered = rendered_dummy if idx == -1 else rendered_dummy[:idx]
    expected_hash = f"{voice_mode}:{hashlib.sha256(rendered.encode()).hexdigest()[:16]}"
    if server._system_prompt_cache is None or server._cache_system_hash != expected_hash:
        prime_system_cache(server, system_content, voice_mode=voice_mode)
