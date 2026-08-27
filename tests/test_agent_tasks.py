"""Tests for badapple_agent_tasks."""
import os
import tempfile
import unittest
from pathlib import Path

import badapple_agent_tasks as at


class TestAgentTasks(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["BADAPPLE_DATA_DIR"] = self.tmp.name
        self.mgr = at.AgentTaskManager(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()
        os.environ.pop("BADAPPLE_DATA_DIR", None)

    def test_create_and_get(self) -> None:
        task = self.mgr.create("find all TODOs")
        self.assertEqual(task.status, "queued")
        self.assertTrue(task.task_id)
        fetched = self.mgr.get(task.task_id)
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertEqual(fetched.goal, "find all TODOs")

    def test_add_step_and_cancel(self) -> None:
        task = self.mgr.create("test task", max_steps=5)
        step = at.AgentStep(thought="list files", tool="list_directory", args={"path": "/"}, result="...")
        self.mgr.add_step(task.task_id, step)
        fetched = self.mgr.get(task.task_id)
        assert fetched is not None
        self.assertEqual(len(fetched.steps), 1)
        self.assertTrue(self.mgr.cancel(task.task_id))
        self.assertEqual(self.mgr.get(task.task_id).status, "cancelled")  # type: ignore[union-attr]

    def test_persistence(self) -> None:
        task = self.mgr.create("persist me")
        step = at.AgentStep(thought="hi", tool="get_current_time", args={}, result="now")
        self.mgr.add_step(task.task_id, step)
        # New manager instance loads the same data.
        mgr2 = at.AgentTaskManager(Path(self.tmp.name))
        fetched = mgr2.get(task.task_id)
        assert fetched is not None
        self.assertEqual(fetched.goal, "persist me")
        self.assertEqual(len(fetched.steps), 1)


if __name__ == "__main__":
    unittest.main()
