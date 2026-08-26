#!/usr/bin/env python3
"""Unit tests for the FSEvents workspace watcher."""

import tempfile
import time
import unittest
from pathlib import Path

import badapple_workspace_watcher


class MockKnowledge:
    def __init__(self) -> None:
        self.indexed: list[list[Path]] = []

    def index_paths(self, paths: list[Path]) -> int:
        self.indexed.append(paths)
        return len(paths)


class WorkspaceWatcherTests(unittest.TestCase):
    def test_should_index_respects_extensions_and_size(self) -> None:
        """_should_index accepts known text files and rejects large or unknown files."""
        w = badapple_workspace_watcher.WorkspaceWatcher(MockKnowledge())
        with tempfile.TemporaryDirectory() as tmp:
            py = Path(tmp) / "test.py"
            py.write_text("x")
            self.assertTrue(w._should_index(py))

            bin_file = Path(tmp) / "big.bin"
            bin_file.write_bytes(b"x" * (badapple_workspace_watcher.MAX_FILE_SIZE + 1))
            self.assertFalse(w._should_index(bin_file))

            skip = Path(tmp) / ".git" / "config"
            skip.parent.mkdir()
            skip.write_text("x")
            self.assertFalse(w._should_index(skip))

    def test_set_workspace_resolves_and_restarts_observer(self) -> None:
        """set_workspace stores the resolved workspace path."""
        w = badapple_workspace_watcher.WorkspaceWatcher(MockKnowledge())
        with tempfile.TemporaryDirectory() as tmp:
            w.set_workspace(Path(tmp))
            self.assertEqual(w.workspace, Path(tmp).resolve())

    def test_start_stop_lifecycle(self) -> None:
        """start() and stop() manage the background worker without crashing."""
        w = badapple_workspace_watcher.WorkspaceWatcher(MockKnowledge())
        with tempfile.TemporaryDirectory() as tmp:
            w.set_workspace(Path(tmp))
            w.start()
            self.assertTrue(w._running)
            time.sleep(0.2)
            w.stop()
            self.assertFalse(w._running)

    def test_manual_enqueue_indexes_files(self) -> None:
        """_enqueue and the worker index a small Python file."""
        knowledge = MockKnowledge()
        w = badapple_workspace_watcher.WorkspaceWatcher(knowledge)
        with tempfile.TemporaryDirectory() as tmp:
            py = Path(tmp) / "hello.py"
            py.write_text("print('hello')")
            w.start()
            w._enqueue(py)
            # Wait for the worker to drain the batch.
            for _ in range(50):
                if knowledge.indexed:
                    break
                time.sleep(0.1)
            w.stop()
            self.assertTrue(len(knowledge.indexed) >= 1)
            self.assertIn(py, knowledge.indexed[0])


if __name__ == "__main__":
    unittest.main()
