"""Tests for badapple_ambient_memory fact extraction."""
import os
import tempfile
import unittest
from pathlib import Path

import badapple_ambient_memory as am
from badapple_extras import MemoryGraph


class TestAmbientMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["BADAPPLE_DATA_DIR"] = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()
        os.environ.pop("BADAPPLE_DATA_DIR", None)

    def test_extract_facts_from_editor_title(self) -> None:
        facts = am.extract_ambient_facts("Code", "README.md - Bad Apple - Code")
        self.assertIn("User is using Code.", facts)
        self.assertIn("User is editing README.md in project Bad Apple with Code.", facts)
        self.assertIn("User's current project is Bad Apple.", facts)

    def test_extract_facts_from_browser_title(self) -> None:
        facts = am.extract_ambient_facts("Safari", "Bad Apple - GitHub")
        self.assertIn("User has GitHub open for Bad Apple.", facts)

    def test_extract_facts_from_xcode(self) -> None:
        facts = am.extract_ambient_facts("Xcode", "Bad Apple — main.swift")
        self.assertIn("User is working on main.swift in Bad Apple.", facts)
        self.assertIn("User is developing software.", facts)

    def test_memory_receives_facts(self) -> None:
        ctx = {"app": "Code", "window": "README.md - Bad Apple - Code", "timestamp": "2025-01-01T00:00:00", "screen_path": None}
        ambient = am.AmbientMemory(self.memory, Path(self.tmp.name))
        ambient._on_snapshot(ctx)
        results = self.memory.search("What is my current project")
        self.assertTrue(any("Bad Apple" in r for r in results))

    def _get_memory(self) -> MemoryGraph:
        from tests._test_utils import fake_encoder

        return MemoryGraph(Path(self.tmp.name), encoder=fake_encoder)

    @property
    def memory(self) -> MemoryGraph:
        if not hasattr(self, "_memory"):
            self._memory = self._get_memory()
        return self._memory


if __name__ == "__main__":
    unittest.main()
