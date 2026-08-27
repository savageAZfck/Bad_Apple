"""Red-team edge-case tests for Bad Apple.

Verifies path traversal, invalid input, and resource exhaustion guards.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import badapple_agent_tasks
import badapple_dashboard as dashboard
import badapple_mcp_marketplace as mcp
import badapple_model_manager
import badapple_vram_governor as vg
from badapple_mlx_server import MLXServer


class TestAgentTaskPathSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.mgr = badapple_agent_tasks.AgentTaskManager(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_create_generates_safe_id(self) -> None:
        t = self.mgr.create("test goal")
        self.assertTrue(badapple_agent_tasks.SAFE_ID_RE.match(t.task_id))

    def test_delete_rejects_traversal(self) -> None:
        self.assertFalse(self.mgr.delete("../etc/passwd"))
        self.assertFalse(self.mgr.delete("/etc/passwd"))
        self.assertFalse(self.mgr.delete("foo/bar"))

    def test_delete_rejects_long_id(self) -> None:
        self.assertFalse(self.mgr.delete("a" * 100))

    def test_malicious_file_name_is_ignored(self) -> None:
        bad_path = self.mgr.tasks_dir / "../etc_passwd.json"
        bad_path.write_text('{"task_id": "../etc/passwd", "goal": "x", "status": "queued"}')
        mgr2 = badapple_agent_tasks.AgentTaskManager(Path(self.tmp.name))
        self.assertEqual(mgr2.get("../etc/passwd"), None)


class TestMcpNameValidation(unittest.TestCase):
    def test_invalid_names_rejected(self) -> None:
        for name in ("../evil", "/etc/passwd", "a/b", "", "x" * 100):
            self.assertFalse(mcp._is_safe_name(name), f"{name!r} should be unsafe")

    def test_valid_names_accepted(self) -> None:
        for name in ("filesystem", "my-server_2", "time"):
            self.assertTrue(mcp._is_safe_name(name), f"{name!r} should be safe")

    def test_add_server_rejects_invalid_name(self) -> None:
        result = mcp.add_mcp_server("../x", "echo hi")
        self.assertIn("alphanumeric", result.lower())


class TestVramGovernorSafety(unittest.TestCase):
    def test_recommend_for_memory_handles_zero(self) -> None:
        self.assertEqual(vg.recommend_for_memory(0), "fast_0.5b")

    @mock.patch.object(vg, "_available_gb", return_value=0.0)
    def test_can_fit_model_with_zero_memory(self, _mock) -> None:
        ok, _ = vg.can_fit_model(100)
        self.assertFalse(ok)


class TestModelRefValidation(unittest.TestCase):
    def test_validate_model_ref(self) -> None:
        mgr = badapple_model_manager.ModelManager(Path(tempfile.gettempdir()))

        class Server:
            def __init__(self, m):
                self.model_manager = m

        srv = Server(mgr)

        def validate(ref: str) -> bool:
            return MLXServer._validate_model_ref(srv, ref)

        self.assertTrue(validate("main_9b"))
        self.assertTrue(validate("mlx-community/Qwen3.5-32B-MLX-4bit"))
        self.assertFalse(validate("../evil"))
        self.assertFalse(validate("/etc/passwd"))
        self.assertFalse(validate(""))


class TestDashboardCsrf(unittest.TestCase):
    def test_token_generated_and_validated(self) -> None:
        token = dashboard._get_csrf_token()
        self.assertTrue(len(token) > 20)
        self.assertTrue(dashboard._check_csrf_token({"X-CSRF-Token": token}))
        self.assertFalse(dashboard._check_csrf_token({"X-CSRF-Token": "wrong"}))
        self.assertFalse(dashboard._check_csrf_token({}))


if __name__ == "__main__":
    unittest.main()
