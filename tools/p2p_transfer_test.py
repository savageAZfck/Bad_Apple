#!/usr/bin/env python3
"""Standalone loopback test for P2P model file transfer.

This script starts two in-memory Bad Apple P2P nodes, copies a tiny real
model into node A, and sends it to node B. It then verifies the files and
provenance manifest arrived intact.

Run from the repo root:
    .venv/bin/python tools/p2p_transfer_test.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

# The script lives in tools/; the package modules live in the repo root.
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

import badapple_model_registry  # noqa: E402
import badapple_p2p_actor  # noqa: E402


SECRET = b"p2p transfer test secret"
REAL_MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def _find_real_model() -> Path | None:
    """Return the latest snapshot dir for a tiny cached model, if present."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hub / "models--mlx-community--Qwen2.5-0.5B-Instruct-4bit"
    if not repo_dir.is_dir():
        return None
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    snap_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
    if not snap_dirs:
        return None
    return sorted(snap_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _make_tiny_model(tmp: Path) -> tuple[Path, str]:
    """Create a tiny synthetic model and return (path, model_id)."""
    model_id = "test-org/TinyLoopback-4bit"
    model_dir = tmp / "tiny_model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({
            "architectures": ["TinyForCausalLM"],
            "max_position_embeddings": 128,
            "vocab_size": 100,
        }),
        encoding="utf-8",
    )
    (model_dir / "weights-4bit.safetensors").write_bytes(b"fake 4-bit weights" * 1024)
    (model_dir / "tokenizer.json").write_text(
        json.dumps({"version": "1.0"}),
        encoding="utf-8",
    )
    return model_dir, model_id


def main() -> int:
    real_model = _find_real_model()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if real_model is not None:
            model_id = REAL_MODEL_ID
            model_dir = real_model
        else:
            print(f"[transfer_test] real model {REAL_MODEL_ID} not in HF cache, using synthetic model")
            model_dir, model_id = _make_tiny_model(tmp)

        d1_data = tmp / "node_a"
        d2_data = tmp / "node_b"
        d1_data.mkdir()
        d2_data.mkdir()

        registry_a = badapple_model_registry.ModelRegistry(d1_data)
        registry_b = badapple_model_registry.ModelRegistry(d2_data)

        add_result = registry_a.add_model(str(model_dir), model_id)
        if add_result.get("status") != "added":
            print(f"[transfer_test] failed to add source model: {add_result}", file=sys.stderr)
            return 1

        actor_a = badapple_p2p_actor.P2PActor(
            SECRET,
            d1_data,
            model_registry=registry_a,
            broadcast_port=19980,
            sync_port=20020,
        )
        actor_b = badapple_p2p_actor.P2PActor(
            SECRET,
            d2_data,
            model_registry=registry_b,
            broadcast_port=19981,
            sync_port=20021,
        )

        proxy_a = badapple_p2p_actor.P2PActorProxy(actor_a)
        proxy_b = badapple_p2p_actor.P2PActorProxy(actor_b)

        actor_a.start()
        actor_b.start()

        proxy_a.start()
        proxy_b.start()

        # Give the TCP listeners a moment to bind.
        time.sleep(0.5)

        # Manually peer A at B's sync port; A needs to know where to send.
        proxy_a.add_peer("127.0.0.1", 20021)

        print(f"[transfer_test] sending model {model_id} from node A to node B...")
        result = proxy_a.send_model("127.0.0.1:20021", model_id)
        print(f"[transfer_test] send result: {result}")

        if "Sent model" not in str(result):
            print("[transfer_test] transfer did not report success", file=sys.stderr)
            return 1

        # Verify B's registry has the model and the files match the manifest.
        models = registry_b._state.get("models", [])
        if not any(m.get("id") == model_id for m in models):
            print("[transfer_test] model was not added to node B registry", file=sys.stderr)
            return 1

        verify = registry_b.verify(model_id)
        if verify.get("status") != "verified":
            print(f"[transfer_test] model verification failed: {verify}", file=sys.stderr)
            return 1

        print(f"[transfer_test] verified {verify.get('files')} files, status: {verify.get('status')}")

        proxy_a.stop()
        proxy_b.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
