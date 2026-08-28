#!/usr/bin/env python3
"""Fuzz-style tests for the local Piper TTS server socket handler."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from badapple_tts_server import _handle_request, _validate_voice_name


class TTSHandlerFuzzTests(unittest.TestCase):
    def test_reject_non_string_text(self) -> None:
        for text in (None, 123, ["hello"], {"x": 1}, ""):
            req = {"text": text, "voice": "en_US-amy-medium"}
            resp = _handle_request(json.dumps(req).encode())
            self.assertFalse(resp["ok"], f"text={text!r}")

    def test_reject_oversized_text(self) -> None:
        req = {"text": "x" * 10_000, "voice": "en_US-amy-medium"}
        resp = _handle_request(json.dumps(req).encode())
        self.assertFalse(resp["ok"])
        self.assertIn("too long", resp["error"])

    def test_reject_invalid_json(self) -> None:
        for raw in (b"not json", b"{\"text\":", b"{}", b"\xff\xfe"):
            resp = _handle_request(raw)
            self.assertFalse(resp["ok"])

    def test_reject_oversized_request(self) -> None:
        big = b"x" * (8_388_608 + 1)
        resp = _handle_request(big)
        self.assertFalse(resp["ok"])
        self.assertIn("too large", resp["error"])

    def test_reject_path_traversal_voice(self) -> None:
        for voice in ("../etc/passwd", "/etc/passwd", "a/b", "a\\b", "", "voice<>"):
            req = {"text": "hello", "voice": voice}
            resp = _handle_request(json.dumps(req).encode())
            self.assertFalse(resp["ok"], f"voice={voice!r}")

    def test_accept_valid_request(self) -> None:
        import wave as wave_mod

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave_mod.open(tmp.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(22050)
                w.writeframes(b"\x00" * 22050 * 2)
            wav_path = Path(tmp.name)

        try:
            with patch("badapple_tts_server._synthesize") as mock:
                mock.return_value = wav_path
                req = {"text": "hello", "voice": "en_US-amy-medium"}
                resp = _handle_request(json.dumps(req).encode() + b"\n")
                self.assertTrue(resp["ok"], resp.get("error"))
                self.assertEqual(resp.get("sample_rate"), 22050)
                self.assertGreater(resp.get("duration_ms", 0), 0)
        finally:
            wav_path.unlink(missing_ok=True)

    def test_validate_voice_name(self) -> None:
        self.assertEqual(_validate_voice_name("en_US-amy-medium"), "en_US-amy-medium")
        for bad in ("", "../x", "x/y", "x\\y", "a b", "a<>b", "a" * 65, 123, None):
            with self.assertRaises(ValueError, msg=f"voice={bad!r}"):
                _validate_voice_name(bad)

    @patch("badapple_tts_server._synthesize", side_effect=RuntimeError("no TTS voice in test"))
    def test_unicode_and_control_characters(self, _mock: Any) -> None:
        for text in ("héllo", "Hello\nworld", "\x00", "<script>", "🔊" * 500):
            req = {"text": text, "voice": "en_US-amy-medium"}
            # It should either accept or reject cleanly, never crash.
            resp = _handle_request(json.dumps(req, ensure_ascii=False).encode())
            self.assertIsInstance(resp, dict)


class TTSServerSocketTests(unittest.TestCase):
    @patch("badapple_tts_server._handle_request", return_value={"ok": True, "mocked": True})
    def test_client_read_is_bounded(self, _mock: Any) -> None:
        from badapple_tts_server import _serve_client

        class FakeConn:
            def __init__(self, chunks: list[bytes]) -> None:
                self.chunks = chunks
                self.sent: list[bytes] = []
                self.closed = False

            def settimeout(self, t: float | None) -> None:
                pass

            def recv(self, n: int) -> bytes:
                if not self.chunks:
                    return b""
                return self.chunks.pop(0)

            def sendall(self, data: bytes) -> None:
                self.sent.append(data)

            def close(self) -> None:
                self.closed = True

        # Send a tiny valid JSON.
        data = json.dumps({"text": "hi", "voice": "en_US-amy-medium"}).encode() + b"\n"
        conn = FakeConn([data])
        _serve_client(conn)
        self.assertTrue(conn.closed)
        self.assertTrue(any(b"error" in s or b"ok" in s for s in conn.sent))


if __name__ == "__main__":
    unittest.main()
