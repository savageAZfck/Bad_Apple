#!/usr/bin/env python3
"""Two-node P2P loopback test on one Mac.

Simulates two Bad Apple edge nodes on different ports, syncs signed model
manifests between them, and pulls a provenance manifest from one to the other.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

import badapple_model_registry
import badapple_p2p_actor


class _MockMemory:
    def __init__(self):
        self._state = {"facts": [{"text": "hello from the mesh"}], "project_context": {}}

    def remember(self, text, source="self"):
        self._state.setdefault("facts", []).append({"text": text, "source": source})

    def _save(self):
        pass


def _secret() -> bytes:
    path = Path("/var/lib/bad_apple/slicks.key")
    if not path.is_file():
        path = Path.home() / ".local/share/badapple/slicks.key"
    return path.read_text().strip().encode()


def main():
    tmp = Path(tempfile.mkdtemp())
    secret = _secret()

    # Node A: uses the real model registry.
    a_registry = badapple_model_registry.ModelRegistry(Path("/var/lib/bad_apple"))
    a_data = Path("/var/lib/bad_apple")
    a_actor = badapple_p2p_actor.P2PActor(
        secret,
        a_data,
        memory=_MockMemory(),
        model_registry=a_registry,
        broadcast_port=19999,
        sync_port=20010,
    )
    a = badapple_p2p_actor.P2PActorProxy(a_actor)

    # Node B: empty registry in a temp dir.
    b_data = tmp / "b"
    b_data.mkdir(parents=True, exist_ok=True)
    b_registry = badapple_model_registry.ModelRegistry(b_data)
    b_actor = badapple_p2p_actor.P2PActor(
        secret,
        b_data,
        memory=_MockMemory(),
        model_registry=b_registry,
        broadcast_port=19998,
        sync_port=20011,
    )
    b = badapple_p2p_actor.P2PActorProxy(b_actor)

    try:
        a_actor.start()
        b_actor.start()
        print("[test] starting P2P daemons...", flush=True)
        a.start()
        b.start()
        time.sleep(1.0)

        print("[test] adding peers manually...", flush=True)
        a.add_peer("127.0.0.1", 20011)
        b.add_peer("127.0.0.1", 20010)
        time.sleep(0.5)

        print("[test] node A syncing to node B...", flush=True)
        sync_result = a.sync_memory()
        print(f"[test] sync result: {sync_result}", flush=True)

        print("[test] node B remote models...", flush=True)
        time.sleep(0.5)
        remote = b.remote_models()
        print(json.dumps(remote, indent=2, default=str))

        # Pick a model from A's registry that has recorded provenance.
        models = a_registry._state.get("models", [])
        target = next((m for m in models if m.get("provenance") == "recorded"), None)
        if target is None:
            print("[test] node A has no recorded provenance models; nothing to pull.", flush=True)
            return
        target_id = target["id"]
        print(f"[test] node B pulling manifest for {target_id} from node A...", flush=True)
        pull_result = b.pull_model_manifest("127.0.0.1:20010", target_id)
        print(json.dumps(pull_result, indent=2, default=str))

        if pull_result.get("manifest"):
            print("[test] SUCCESS — model manifest pulled over P2P.", flush=True)
        else:
            print("[test] FAIL — no manifest in pull response.", flush=True)
    finally:
        print("[test] stopping nodes...", flush=True)
        a_actor.stop()
        b_actor.stop()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
