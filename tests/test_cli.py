#!/usr/bin/env python3
"""Unit tests for the badapple CLI."""

import os
import subprocess
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._binary = Path(__file__).parent.parent / "target" / "release" / "badapple"
        if not cls._binary.is_file():
            raise unittest.SkipTest("release badapple binary not built")

    def _run(self, *args: str) -> tuple[int, str, str]:
        env = os.environ.copy()
        env.pop("BADAPPLE_VOICE", None)
        env.pop("BADAPPLE_SPEAK", None)
        result = subprocess.run(
            [str(self._binary), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr

    def test_help(self) -> None:
        """badapple --help prints the usage banner."""
        rc, out, _ = self._run("--help")
        self.assertEqual(rc, 0)
        self.assertIn("badapple", out)
        self.assertIn("--doctor", out)

    def test_doctor(self) -> None:
        """badapple --doctor prints a support diagnostic report."""
        rc, out, _ = self._run("--doctor")
        self.assertEqual(rc, 0)
        self.assertIn("=== Bad Apple Doctor ===", out)
        self.assertIn("[host]", out)
        self.assertIn("[binaries]", out)


if __name__ == "__main__":
    unittest.main()
