"""Mock-based tests for MLXServer without loading any model weights."""
import unittest
from unittest import mock

from badapple_mlx_server import MLXServer


class TestMLXServerAdmit(unittest.TestCase):
    @mock.patch("badapple_vram_governor.can_fit_model", return_value=(False, 4.0))
    def test_admit_rejects_when_no_memory(self, _mock) -> None:
        srv = mock.MagicMock()
        srv._memory_for_model.return_value = 8.0
        result = MLXServer.admit_model(srv, "main_9b")
        self.assertFalse(result["ok"])
        self.assertIn("needed_gb", result)
        self.assertIn("available_gb", result)

    @mock.patch("badapple_vram_governor.can_fit_model", return_value=(True, 16.0))
    def test_admit_accepts_when_memory_fits(self, _mock) -> None:
        srv = mock.MagicMock()
        srv._memory_for_model.return_value = 6.0
        result = MLXServer.admit_model(srv, "main_9b")
        self.assertTrue(result["ok"])


class TestMLXServerModelValidation(unittest.TestCase):
    def test_validate_model_ref(self) -> None:
        import badapple_model_manager

        mgr = badapple_model_manager.ModelManager()

        class FakeServer:
            def __init__(self):
                self.model_manager = mgr

        srv = FakeServer()
        self.assertTrue(MLXServer._validate_model_ref(srv, "main_9b"))
        self.assertFalse(MLXServer._validate_model_ref(srv, "../etc/passwd"))
        self.assertFalse(MLXServer._validate_model_ref(srv, "/tmp/m"))


if __name__ == "__main__":
    unittest.main()
