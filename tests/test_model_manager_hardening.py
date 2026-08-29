#!/usr/bin/env python3
"""Hardening tests for badapple_model_manager.

Property/fuzz-style checks for model ID validation, repo ID validation,
local-path containment, and timeout behavior.
"""

import os
import random
import string
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import badapple_model_manager as bmm
from badapple_model_manager import ModelManager


class ModelManagerHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        os.environ["BADAPPLE_DATA_DIR"] = self.tmpdir
        os.environ["BADAPPLE_VERIFY_MODEL_HASHES"] = "0"
        # Downloads and long timeouts off by default.
        os.environ.pop("BADAPPLE_ALLOW_DOWNLOADS", None)
        os.environ.pop("BADAPPLE_DOWNLOAD_TIMEOUT_SECONDS", None)
        os.environ.pop("BADAPPLE_MAX_DOWNLOAD_GB", None)
        self.mgr = ModelManager()

    def tearDown(self) -> None:
        self.mgr.shutdown()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)
        for key in (
            "BADAPPLE_DATA_DIR",
            "BADAPPLE_VERIFY_MODEL_HASHES",
            "BADAPPLE_ALLOW_DOWNLOADS",
            "BADAPPLE_DOWNLOAD_TIMEOUT_SECONDS",
            "BADAPPLE_MAX_DOWNLOAD_GB",
        ):
            os.environ.pop(key, None)

    def _make_manager(self, **env: str) -> ModelManager:
        for k, v in env.items():
            os.environ[k] = v
        return ModelManager()

    # ------------------------------------------------------------------
    # Model ID validation
    # ------------------------------------------------------------------
    def test_known_model_ids_are_safe(self) -> None:
        for mid in ("fast_0.5b", "main_9b", "main_32b", "main_70b", "vision_2b", "flux_4b"):
            with self.subTest(mid=mid):
                self.assertTrue(bmm._is_safe_model_id(mid))
                self.assertEqual(self.mgr.status(mid)["id"], mid)
                self.assertGreater(self.mgr.memory_required(mid), 0.0)

    def test_invalid_model_ids_rejected(self) -> None:
        invalid = [
            "",
            "a" * 65,
            ".",
            "..",
            "...",
            "foo/bar",
            "a\\b",
            "../etc/passwd",
            "/tmp/m",
            "main_9b\x00",
            " main_9b",
            "main_9b ",
        ]
        for mid in invalid:
            with self.subTest(mid=mid):
                self.assertFalse(bmm._is_safe_model_id(mid))
                self.assertIn("error", self.mgr.status(mid))
                self.assertEqual(self.mgr.memory_required(mid), 0.0)
                self.assertIn("error", self.mgr.ensure_cached(mid, download=False))
                self.assertFalse(self.mgr.wait_for_download(mid))

    def test_start_download_rejects_path_traversal(self) -> None:
        os.environ["BADAPPLE_ALLOW_DOWNLOADS"] = "1"
        mgr = ModelManager()
        try:
            for mid in ("../etc/passwd", "/tmp/m", "foo/bar", "a" * 65):
                with self.subTest(mid=mid):
                    result = mgr.start_download(mid)
                    self.assertIn("error", result)
                    self.assertIn("invalid", result["error"].lower())
        finally:
            mgr.shutdown()

    def test_model_id_property_fuzz(self) -> None:
        """Fuzz-style sanity check: generated valid IDs pass, invalid ones fail."""
        rng = random.Random(0)
        valid_chars = string.ascii_letters + string.digits + "_-"
        middle_chars = valid_chars + "."
        for _ in range(50):
            length = rng.randint(1, 64)
            body = "".join(rng.choices(valid_chars, k=1))
            if length > 2:
                body += "".join(rng.choices(middle_chars, k=length - 2))
                body += "".join(rng.choices(valid_chars, k=1))
            else:
                body = "".join(rng.choices(valid_chars, k=length))
            # Ensure we do not accidentally create a ".." sequence.
            body = body.replace("..", ".x")
            with self.subTest(body=body):
                self.assertTrue(bmm._is_safe_model_id(body), f"{body!r} should be safe")

        for _ in range(50):
            bad = "".join(rng.choices(string.printable, k=rng.randint(1, 32)))
            if "/" in bad or "\\" in bad or ".." in bad or len(bad) > 64:
                self.assertFalse(bmm._is_safe_model_id(bad))

    # ------------------------------------------------------------------
    # Repo ID validation
    # ------------------------------------------------------------------
    def test_known_repo_ids_are_safe(self) -> None:
        for rid in (
            "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit",
            "mflux/flux2-klein-4b",
        ):
            with self.subTest(rid=rid):
                self.assertTrue(bmm._is_safe_repo_id(rid))

    def test_invalid_repo_ids_rejected(self) -> None:
        invalid = [
            "",
            "foo",
            "foo/bar/baz",
            "../etc/passwd",
            "/etc/passwd",
            "evil/../other",
            "org/.hidden",
            "a" * 200 + "/b",
            "a/b\x00",
        ]
        for rid in invalid:
            with self.subTest(rid=rid):
                self.assertFalse(bmm._is_safe_repo_id(rid))
                self.assertRaises(ValueError, bmm._validate_repo_id, rid)

    def test_resolve_cache_path_rejects_invalid_repo(self) -> None:
        self.assertIsNone(self.mgr._resolve_cache_path("foo/bar/baz", allow_download=False))

    def test_repo_id_property_fuzz(self) -> None:
        rng = random.Random(1)
        valid_chars = string.ascii_letters + string.digits + "_-"
        middle_chars = valid_chars + "."
        for _ in range(30):
            ns = "".join(rng.choices(valid_chars, k=1))
            if rng.random() > 0.3:
                ns += "".join(rng.choices(middle_chars, k=rng.randint(0, 20)))
                ns += "".join(rng.choices(valid_chars, k=1))
            name = "".join(rng.choices(valid_chars, k=1))
            if rng.random() > 0.3:
                name += "".join(rng.choices(middle_chars, k=rng.randint(0, 20)))
                name += "".join(rng.choices(valid_chars, k=1))
            rid = f"{ns.replace('..', '.x')}/{name.replace('..', '.x')}"
            with self.subTest(rid=rid):
                self.assertTrue(bmm._is_safe_repo_id(rid))

    # ------------------------------------------------------------------
    # Local path containment
    # ------------------------------------------------------------------
    def test_local_path_outside_hf_cache_rejected(self) -> None:
        outside_dir = tempfile.mkdtemp()
        try:
            self.mgr.mark_loaded("main_9b", outside_dir)
            status = self.mgr.status("main_9b")
            self.assertEqual(status["status"], "error")
            self.assertIn("HF cache", status["error"])
        finally:
            import shutil

            shutil.rmtree(outside_dir, ignore_errors=True)

    def test_local_path_under_hf_cache_accepted(self) -> None:
        fake_cache = Path(tempfile.mkdtemp())
        safe_dir = fake_cache / "models--test--model" / "snapshots" / "abc123"
        safe_dir.mkdir(parents=True)
        (safe_dir / "config.json").write_text("{}")
        try:
            with patch.object(bmm, "_hf_cache_root", return_value=fake_cache.resolve()):
                mgr = ModelManager()
                try:
                    mgr.mark_loaded("main_9b", str(safe_dir))
                    self.assertEqual(mgr.status("main_9b")["status"], "loaded")
                finally:
                    mgr.shutdown()
        finally:
            import shutil

            shutil.rmtree(fake_cache, ignore_errors=True)

    def test_record_provenance_rejects_outside_cache(self) -> None:
        outside_dir = tempfile.mkdtemp()
        try:
            result = self.mgr.record_provenance("main_9b", outside_dir)
            self.assertEqual(result.get("status"), "invalid_path")
        finally:
            import shutil

            shutil.rmtree(outside_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Timeout behavior
    # ------------------------------------------------------------------
    def test_run_with_timeout_raises_on_slow_call(self) -> None:
        with self.assertRaises(TimeoutError):
            self.mgr._run_with_timeout(lambda: time.sleep(0.5), timeout=0.05)

    def test_run_with_timeout_returns_fast_result(self) -> None:
        result = self.mgr._run_with_timeout(lambda: 42, timeout=5.0)
        self.assertEqual(result, 42)

    def test_download_timeout(self) -> None:
        os.environ["BADAPPLE_ALLOW_DOWNLOADS"] = "1"
        os.environ["BADAPPLE_DOWNLOAD_TIMEOUT_SECONDS"] = "0.05"
        mgr = ModelManager()
        try:
            with patch("huggingface_hub.snapshot_download", side_effect=lambda *_a, **_k: time.sleep(0.5) or "/mock/cache"):
                result = mgr.start_download("fast_0.5b")
                self.assertIn(result["status"], ("queued", "downloading"))
                ok = mgr.wait_for_download("fast_0.5b", timeout=5.0)
                self.assertFalse(ok)
                status = mgr.status("fast_0.5b")
                self.assertEqual(status["status"], "error")
                self.assertIn("timed out", status["error"].lower())
        finally:
            mgr.shutdown()

    # ------------------------------------------------------------------
    # Max download size
    # ------------------------------------------------------------------
    def test_max_download_size_enforced(self) -> None:
        fake_cache = Path(tempfile.mkdtemp())
        fake_dir = fake_cache / "models--big--model"
        fake_dir.mkdir(parents=True)
        (fake_dir / "model.safetensors").write_bytes(b"x" * 100)
        try:
            with patch.object(bmm, "_hf_cache_root", return_value=fake_cache.resolve()):
                with patch("huggingface_hub.snapshot_download", return_value=str(fake_dir)):
                    os.environ["BADAPPLE_ALLOW_DOWNLOADS"] = "1"
                    os.environ["BADAPPLE_DOWNLOAD_TIMEOUT_SECONDS"] = "30"
                    os.environ["BADAPPLE_MAX_DOWNLOAD_GB"] = "0.0"
                    mgr = ModelManager()
                    try:
                        with self.assertRaises(ValueError) as ctx:
                            mgr._resolve_cache_path("mlx-community/model", allow_download=True)
                        self.assertIn("exceeds", str(ctx.exception))
                    finally:
                        mgr.shutdown()
        finally:
            import shutil

            shutil.rmtree(fake_cache, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
