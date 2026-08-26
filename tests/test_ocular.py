#!/usr/bin/env python3
"""Unit tests for the Ocular UI Stream."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import badapple_ocular
import badapple_vision


def _fake_capture(path: Path) -> Path:
    """Side effect for mocked screen capture: touch the target file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake png")
    return path


class OcularTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_dir = badapple_ocular.OCULAR_DIR
        self._orig_screen = badapple_ocular.OCULAR_SCREEN
        self._orig_context = badapple_ocular.OCULAR_CONTEXT
        self.tmp = tempfile.TemporaryDirectory()
        badapple_ocular.OCULAR_DIR = Path(self.tmp.name) / "ocular"
        badapple_ocular.OCULAR_SCREEN = badapple_ocular.OCULAR_DIR / "screen.png"
        badapple_ocular.OCULAR_CONTEXT = badapple_ocular.OCULAR_DIR / "context.json"

    def tearDown(self) -> None:
        if badapple_ocular.is_running():
            badapple_ocular.stop()
        badapple_ocular.OCULAR_DIR = self._orig_dir
        badapple_ocular.OCULAR_SCREEN = self._orig_screen
        badapple_ocular.OCULAR_CONTEXT = self._orig_context
        self.tmp.cleanup()

    def test_status_when_stopped(self) -> None:
        """status() reports the stream as stopped with no context."""
        result = badapple_ocular.status()
        self.assertFalse(result["running"])
        self.assertEqual(result["context"], {})

    @mock.patch.object(badapple_vision, "capture_screen", side_effect=_fake_capture)
    @mock.patch.object(badapple_vision, "get_vision_host")
    @mock.patch.object(badapple_vision, "unload_vision_model")
    def test_capture_now_writes_context(self, _ul, _vh, _cap) -> None:
        """capture_now() writes a context file even if no VLM is used."""
        badapple_ocular.capture_now()
        self.assertTrue(badapple_ocular.OCULAR_CONTEXT.is_file())
        ctx = json.loads(badapple_ocular.OCULAR_CONTEXT.read_text(encoding="utf-8"))
        self.assertIn("timestamp", ctx)
        self.assertEqual(ctx.get("description"), "")

    @mock.patch.object(badapple_vision, "capture_screen", side_effect=_fake_capture)
    def test_start_stop_stream(self, _cap) -> None:
        """start() and stop() manage the background thread."""
        result = badapple_ocular.start(0.1, 0.0)
        self.assertIn("started", result)
        self.assertTrue(badapple_ocular.is_running())
        time.sleep(1.0)
        self.assertTrue(badapple_ocular.OCULAR_CONTEXT.is_file())
        result = badapple_ocular.stop()
        self.assertIn("stopped", result)
        self.assertFalse(badapple_ocular.is_running())

    def test_start_is_idempotent(self) -> None:
        """start() returns a message when the stream is already running."""
        with mock.patch.object(badapple_vision, "capture_screen", side_effect=_fake_capture):
            badapple_ocular.start(0.1, 0.0)
            result = badapple_ocular.start()
            self.assertIn("already running", result)
            badapple_ocular.stop()

    def test_stop_when_not_running(self) -> None:
        """stop() is safe when the stream is already stopped."""
        badapple_ocular._running = False
        result = badapple_ocular.stop()
        self.assertIn("not running", result)


if __name__ == "__main__":
    unittest.main()
