#!/usr/bin/env python3
"""Test encrypted P2P sync between the live daemon and a local test peer."""

import asyncio
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import badapple_p2p
from badapple_extras import MemoryGraph
from badapple_knowledge import BadAppleKnowledge
from badapple_mlx_server import load_slicks_secret


def main():
    secret = load_slicks_secret()
    if not secret:
        print("No SLICKS secret found.", file=sys.stderr)
        return 1

    test_dir = Path(tempfile.mkdtemp(prefix="badapple_p2p_test_"))
    knowledge = BadAppleKnowledge()
    memory = MemoryGraph(test_dir, encoder=knowledge._encode_texts)
    peer = badapple_p2p.P2PDaemon(
        secret,
        test_dir,
        memory=memory,
        broadcast_port=9998,
        sync_port=10001,
    )

    async def peer_loop():
        await peer.start()
        while peer.is_running():
            await asyncio.sleep(1)

    def run_peer():
        asyncio.run(peer_loop())

    t = threading.Thread(target=run_peer, daemon=True)
    t.start()
    time.sleep(1)

    # Tell the main daemon to enable P2P, add our test peer, and sync.
    exe = "/Users/savag3/bad_apple/target/release/badapple"
    subprocess.run([exe, "p2p on"], capture_output=True, text=True, timeout=30)
    r1 = subprocess.run([exe, "p2p add peer 127.0.0.1:10001"], capture_output=True, text=True, timeout=30)
    if r1.returncode != 0:
        print(f"add peer failed: {r1.stderr or r1.stdout}")
        return 1

    # Add a fact to the main memory first so there is something to sync.
    subprocess.run([exe, "add a reminder: p2p sync test"], capture_output=True, text=True, timeout=30)

    r2 = subprocess.run([exe, "p2p sync"], capture_output=True, text=True, timeout=60)
    if r2.returncode != 0:
        print(f"p2p sync failed: {r2.stderr or r2.stdout}")
        return 1

    # Wait a bit for TCP and storage.
    time.sleep(1)
    facts = memory._state.get("facts", [])
    if facts:
        print("P2P sync test passed.")
        print(f"Peer received {len(facts)} fact(s).")
        for f in facts[:3]:
            print(f"  - {f.get('source', 'unknown')}: {f.get('text', '')[:80]}")
        return 0

    print("P2P sync test failed: no facts received.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
