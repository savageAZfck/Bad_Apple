import base64
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from badapple_extras import AuditLedger, MemoryGraph
from badapple_plugins import PluginRegistry
from badapple_runtime import CircuitBreaker, HealthRegistry, RuntimeControl
from badapple_undo import UndoJournal
from badapple_vault import ArtifactManifest, EncryptedBackup, GenerationStore


class RuntimeControlTests(unittest.TestCase):
    def test_private_kill_and_safe_modes_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RuntimeControl(Path(tmp))
            runtime.set_ready()
            runtime.set_private_mode(True)
            runtime.engage_kill_switch("test")
            self.assertFalse(runtime.allows_generation())
            self.assertFalse(runtime.allows_mutation())

            restored = RuntimeControl(Path(tmp))
            self.assertTrue(restored.private_mode)
            self.assertTrue(restored.killed)
            restored.reset_kill_switch()
            restored.enter_safe_mode("integrity")
            self.assertTrue(restored.allows_generation())
            self.assertFalse(restored.allows_mutation())

    def test_circuit_breaker_recovers_with_half_open_probe(self):
        breaker = CircuitBreaker("test", failure_threshold=2, recovery_seconds=0.01)
        breaker.failure()
        self.assertTrue(breaker.allow())
        breaker.failure()
        self.assertFalse(breaker.allow())
        time.sleep(0.11)
        self.assertTrue(breaker.allow())
        self.assertFalse(breaker.allow())
        breaker.success()
        self.assertTrue(breaker.allow())

    def test_health_levels_are_reported_independently(self):
        health = HealthRegistry()
        health.register("event_loop", "liveness", lambda: True)
        health.register("model", "readiness", lambda: {"loaded": True})
        health.register("ledger", "correctness", lambda: False)
        snapshot = health.snapshot()
        self.assertTrue(snapshot["summary"]["liveness"])
        self.assertTrue(snapshot["summary"]["readiness"])
        self.assertFalse(snapshot["summary"]["correctness"])

    def test_audit_ledger_serializes_concurrent_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = AuditLedger(Path(tmp))
            threads = [
                threading.Thread(target=lambda n=i: [ledger.record("test", {"writer": n, "index": j}) for j in range(20)])
                for i in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            results = ledger.verify()
            self.assertEqual(len(results), 160)
            self.assertTrue(all(result["valid"] for result in results))

    def test_generation_store_detects_corruption_and_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "state.json"
            source.write_text('{"value": 1}', encoding="utf-8")
            store = GenerationStore(root / "data")
            generation = store.commit("test", {"state.json": source})
            self.assertTrue(store.verify(generation)["valid"])
            restored = root / "restored"
            store.restore(generation, restored)
            self.assertEqual((restored / "state.json").read_text(encoding="utf-8"), '{"value": 1}')
            (store.root / generation / "files" / "state.json").write_text("corrupt", encoding="utf-8")
            self.assertFalse(store.verify(generation)["valid"])

    def test_artifact_manifest_and_encrypted_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            artifact = source / "model.bin"
            artifact.write_bytes(b"verified model")
            manifest = ArtifactManifest.create([artifact], {"model": "test"})
            signing_key = ec.generate_private_key(ec.SECP256R1())
            public_key = signing_key.public_key().public_bytes(
                serialization.Encoding.X962,
                serialization.PublicFormat.UncompressedPoint,
            )
            signature = signing_key.sign(ArtifactManifest.signing_payload(manifest), ec.ECDSA(hashes.SHA256()))
            manifest = ArtifactManifest.seal(
                manifest,
                base64.b64encode(public_key).decode(),
                base64.b64encode(signature).decode(),
            )
            verification = ArtifactManifest.verify(manifest)
            self.assertTrue(verification["valid"])
            self.assertTrue(verification["signature_valid"])

            backup = root / "backup.badapple"
            EncryptedBackup.create(source, backup, "correct horse battery staple")
            restored = root / "portable"
            result = EncryptedBackup.extract(backup, restored, "correct horse battery staple")
            self.assertEqual(result["files"], 1)
            self.assertEqual((restored / "model.bin").read_bytes(), b"verified model")

    def test_undo_restores_verified_file_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "note.txt"
            path.write_text("before", encoding="utf-8")
            journal = UndoJournal(root / "data")
            journal.capture_file(path, "write_file")
            path.write_text("after", encoding="utf-8")
            self.assertIn("Undid", journal.undo_last())
            self.assertEqual(path.read_text(encoding="utf-8"), "before")
            self.assertEqual(journal.undo_last(), "Nothing to undo.")

    def test_workflows_require_review_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryGraph(Path(tmp))
            result = memory.learn_workflow(
                "morning",
                "start my morning",
                [{"tool": "get_current_time", "args": {}}],
            )
            self.assertIn("disabled review mode", result)
            self.assertFalse(memory.workflows()[0]["enabled"])
            memory.set_workflow_enabled("morning", True)
            self.assertTrue(memory.workflows()[0]["enabled"])

    def test_signed_plugin_is_verified_and_invoked(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            registry = PluginRegistry(data_dir)
            private_key = Ed25519PrivateKey.generate()
            public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            (registry.trust_dir / "test.pub").write_text(base64.b64encode(public_key).decode(), encoding="ascii")
            plugin_dir = registry.root / "sample"
            plugin_dir.mkdir()
            entrypoint = plugin_dir / "run"
            entrypoint.write_text("#!/bin/sh\ncat\n", encoding="utf-8")
            entrypoint.chmod(0o755)
            unsigned = {
                "name": "sample",
                "version": "1.0.0",
                "publisher": "test",
                "entrypoint": "run",
                "tools": [{"name": "sample_echo", "parameters": {"type": "object"}}],
                "files": {"run": hashlib.sha256(entrypoint.read_bytes()).hexdigest()},
            }
            signature = private_key.sign(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode())
            manifest = {**unsigned, "signature": base64.b64encode(signature).decode()}
            (plugin_dir / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(registry.reload(), {})
            self.assertTrue(registry.has_tool("sample_echo"))
            output = json.loads(registry.invoke("sample_echo", {"value": 7}))
            self.assertEqual(output["arguments"]["value"], 7)


if __name__ == "__main__":
    unittest.main()
