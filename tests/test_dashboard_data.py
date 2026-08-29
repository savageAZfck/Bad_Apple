"""Tests for badapple_dashboard_data._voice_activity's symlink hardening.

/tmp is world-writable, so anything else running as this user could
pre-create /tmp/badapple_voice_debug.log as a symlink to a file the
dashboard should never surface. _voice_activity must refuse to follow it
rather than reading through it.
"""

import os
import unittest
from unittest.mock import patch

import badapple_dashboard_data


class VoiceActivitySymlinkTests(unittest.TestCase):
    def setUp(self):
        self.log_path = "/tmp/badapple_voice_debug.log"
        self.target_path = "/tmp/badapple_test_symlink_target.log"
        for p in (self.log_path, self.target_path):
            if os.path.islink(p) or os.path.exists(p):
                os.remove(p)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for p in (self.log_path, self.target_path):
            if os.path.islink(p) or os.path.exists(p):
                os.remove(p)

    def test_reads_a_real_log_file(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("2026-01-01T00:00:00Z deliver: hello world\n")
        events = badapple_dashboard_data._voice_activity()
        self.assertTrue(any(e["type"] == "command" and e["text"] == "hello world" for e in events))

    def test_refuses_to_follow_a_symlink(self):
        with open(self.target_path, "w", encoding="utf-8") as f:
            f.write("consume: isFinal=true transcript='secret data' awaitingNextUtterance=false\n")
        os.symlink(self.target_path, self.log_path)
        with patch("builtins.print") as mock_print:
            events = badapple_dashboard_data._voice_activity()
        self.assertEqual(events, [])
        # Should log the refusal rather than silently succeed or crash.
        self.assertTrue(mock_print.called)


if __name__ == "__main__":
    unittest.main()
