#!/usr/bin/env python3
"""Tests for badapple_model_manager."""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from badapple_model_manager import ModelManager


class ModelManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        os.environ["BADAPPLE_DATA_DIR"] = self.tmpdir
        self.mgr: ModelManager | None = None

    def tearDown(self) -> None:
        import shutil

        if self.mgr is not None:
            self.mgr.shutdown()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_initial_status_for_known_models(self) -> None:
        self.mgr = ModelManager()
        status = self.mgr.status()
        self.assertIn("main_9b", status)
        self.assertIn("fast_0.5b", status)
        self.assertIn("vision_2b", status)
        self.assertIn("flux_4b", status)
        self.assertEqual(status["main_9b"]["status"], "missing")

    def test_allow_downloads_defaults_to_false(self) -> None:
        self.mgr = ModelManager()
        self.assertFalse(self.mgr.allow_downloads)

    def test_set_allow_downloads(self) -> None:
        self.mgr = ModelManager()
        self.mgr.set_allow_downloads(True)
        self.assertTrue(self.mgr.allow_downloads)

    @patch("huggingface_hub.snapshot_download")
    def test_start_download_transitions_to_cached(self, mock_snapshot) -> None:
        mock_snapshot.return_value = "/mock/cache/main"
        os.environ["BADAPPLE_ALLOW_DOWNLOADS"] = "1"
        try:
            self.mgr = ModelManager()
            self.assertTrue(self.mgr.allow_downloads)
            result = self.mgr.start_download("main_9b")
            self.assertIn(result["status"], ("queued", "downloading"))
            for _ in range(50):
                if self.mgr.status("main_9b")["status"] == "cached":
                    break
                time.sleep(0.05)
            self.assertEqual(self.mgr.status("main_9b")["status"], "cached")
            self.assertEqual(self.mgr.status("main_9b")["local_path"], "/mock/cache/main")
        finally:
            del os.environ["BADAPPLE_ALLOW_DOWNLOADS"]

    @patch("huggingface_hub.snapshot_download")
    def test_download_without_permission_fails(self, mock_snapshot) -> None:
        mock_snapshot.return_value = "/mock/cache/main"
        self.mgr = ModelManager()
        with self.assertRaises(RuntimeError):
            self.mgr.start_download("main_9b")

    @patch("huggingface_hub.snapshot_download")
    def test_wait_for_download(self, mock_snapshot) -> None:
        mock_snapshot.return_value = "/mock/cache/fast"
        os.environ["BADAPPLE_ALLOW_DOWNLOADS"] = "1"
        try:
            self.mgr = ModelManager()
            self.mgr.start_download("fast_0.5b")
            ok = self.mgr.wait_for_download("fast_0.5b", timeout=5)
            self.assertTrue(ok)
            self.assertEqual(self.mgr.status("fast_0.5b")["status"], "cached")
        finally:
            del os.environ["BADAPPLE_ALLOW_DOWNLOADS"]

    def test_mark_loaded_and_unloaded(self) -> None:
        self.mgr = ModelManager()
        self.mgr.mark_loaded("main_9b", "/mock/path")
        status = self.mgr.status("main_9b")
        self.assertEqual(status["status"], "loaded")
        self.assertEqual(status["local_path"], "/mock/path")
        self.mgr.mark_unloaded("main_9b")
        self.assertEqual(self.mgr.status("main_9b")["status"], "cached")


if __name__ == "__main__":
    unittest.main()
