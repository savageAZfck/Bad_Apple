import shutil
import tempfile
import unittest
from pathlib import Path

import badapple_p2p


class _MockMemory:
    """Minimal memory stand-in for P2P sync tests."""

    def __init__(self, facts):
        self._state = {"facts": facts, "project_context": {}}
        self.remembered = []

    def remember(self, text, source="self"):
        self._state.setdefault("facts", []).append({"text": text, "source": source})
        self.remembered.append((text, source))

    def _save(self):
        pass


class _MockWorkspace:
    """Minimal workspace stand-in."""

    def __init__(self, path):
        self._path = Path(path)

    @property
    def path(self):
        return str(self._path)

    def summary(self):
        return "test workspace"


class _MockModelRegistry:
    """Minimal model registry stand-in for P2P manifest tests."""

    def __init__(self, models):
        self._models = models

    def get_manifests_for_p2p(self):
        return self._models


class TestP2PSync(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.secret = b"this is a shared test secret not used in prod"
        # Two daemons on different ports, same workspace context.
        self.w1 = _MockWorkspace(self.tmp / "w1")
        self.w2 = _MockWorkspace(self.tmp / "w2")
        self.m1 = _MockMemory([{"text": "fact-one"}])
        self.m2 = _MockMemory([{"text": "fact-two"}])
        self.d1 = badapple_p2p.P2PDaemon(
            self.secret,
            self.tmp / "d1",
            memory=self.m1,
            workspace=self.w1,
            broadcast_port=19999,
            sync_port=20000,
        )
        self.d2 = badapple_p2p.P2PDaemon(
            self.secret,
            self.tmp / "d2",
            memory=self.m2,
            workspace=self.w2,
            broadcast_port=19999,
            sync_port=20001,
        )

    async def asyncTearDown(self):
        await self.d1.stop()
        await self.d2.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def test_manual_peer_sync(self):
        await self.d1.start()
        await self.d2.start()

        # Mutual manual peering so loopback test works without UDP broadcast.
        self.d1.add_peer("127.0.0.1", 20001)
        self.d2.add_peer("127.0.0.1", 20000)

        result = await self.d1.sync_memory()
        self.assertIn("1 peer(s)", result)

        # d2 should have remembered d1's fact.
        remembered_texts = [f for f, _ in self.m2.remembered]
        self.assertIn("fact-one", remembered_texts)

        # d1 should have discovered d2.
        self.assertIn("127.0.0.1:20001", self.d1.get_peers())

    async def test_blocklist_rejects_sync_origin(self):
        await self.d1.start()
        await self.d2.start()

        # Block d1's origin on d2; d2 should reject the incoming sync frame.
        self.d2._blocklist.add(self.d1.origin_id)
        self.d1.add_peer("127.0.0.1", 20001)

        result = await self.d1.sync_memory()
        self.assertIn("0 peer(s)", result)

    async def test_oversized_beacon_rejected(self):
        await self.d1.start()
        # Handlers should silently drop oversized or invalid frames; no crash.
        await self.d1._handle_udp(b"x" * (badapple_p2p.P2P_MAX_BEACON_SIZE + 1), ("127.0.0.1", 12345))


class TestP2PModelManifest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.secret = b"model manifest sync test secret"
        self.m1 = _MockMemory([{"text": "model-fact"}])
        self.m2 = _MockMemory([{"text": "other-fact"}])
        self.registry = _MockModelRegistry([
            {
                "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "size_gb": 0.5,
                "quantization": "4-bit",
                "architecture": "Qwen2ForCausalLM",
                "provenance": "recorded",
                "signature": "se-v2",
                "manifest": {"repo_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "files": {}},
            },
        ])
        self.d1 = badapple_p2p.P2PDaemon(
            self.secret,
            self.tmp / "d1",
            memory=self.m1,
            model_registry=self.registry,
            broadcast_port=19999,
            sync_port=20002,
        )
        self.d2 = badapple_p2p.P2PDaemon(
            self.secret,
            self.tmp / "d2",
            memory=self.m2,
            broadcast_port=19999,
            sync_port=20003,
        )

    async def asyncTearDown(self):
        await self.d1.stop()
        await self.d2.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def test_p2p_model_manifest_sync(self):
        await self.d1.start()
        await self.d2.start()

        # Manually peer d1 at d2's sync port.
        self.d1.add_peer("127.0.0.1", 20003)

        result = await self.d1.sync_memory()
        self.assertIn("1 peer(s)", result)

        peer_id = f"{self.d1.origin_id}@127.0.0.1"
        remote = self.d2.remote_models()
        self.assertIn(peer_id, remote)
        self.assertEqual(len(remote[peer_id]), 1)
        self.assertEqual(remote[peer_id][0]["id"], "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        self.assertEqual(remote[peer_id][0]["provenance"], "recorded")

    async def test_p2p_pull_model_manifest(self):
        await self.d1.start()
        await self.d2.start()

        # Manually peer d2 at d1's sync port and pull a model.
        self.d2.add_peer("127.0.0.1", 20002)
        result = await self.d2.pull_model_manifest("127.0.0.1:20002", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

        self.assertIn("model", result)
        self.assertIn("manifest", result)
        self.assertEqual(result["model"]["id"], "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        self.assertEqual(result["manifest"].get("repo_id"), "mlx-community/Qwen2.5-0.5B-Instruct-4bit")


if __name__ == "__main__":
    unittest.main()
