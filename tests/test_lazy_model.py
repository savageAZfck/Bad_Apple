"""Tests for the lazy main-model loading and fast-tier fast-model routing."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import badapple_fast_model
import badapple_tier


class TestLazyModel(unittest.TestCase):
    def tearDown(self) -> None:
        for var in ("BADAPPLE_LAZY_MAIN_MODEL", "BADAPPLE_FAST_MODEL", "BADAPPLE_DATA_DIR"):
            os.environ.pop(var, None)

    def test_fast_tier_routes_chitchat_to_fast_model_when_enabled(self) -> None:
        router = badapple_tier.TieringRouter(fast_model_enabled=True)
        for prompt in ("hello", "how are you", "thanks", "ping"):
            tier, payload = router.select_tier(prompt)
            self.assertEqual(tier, "fast")
            self.assertIsNotNone(payload)
            self.assertTrue(payload.get("fast_model"))

    def test_fast_tier_routes_chitchat_without_fast_model(self) -> None:
        router = badapple_tier.TieringRouter(fast_model_enabled=False)
        tier, payload = router.select_tier("hello")
        self.assertEqual(tier, "fast")
        self.assertIsNotNone(payload)
        self.assertFalse(payload.get("fast_model"))

    def test_fast_model_path_falls_back_to_default_when_lazy(self) -> None:
        os.environ["BADAPPLE_LAZY_MAIN_MODEL"] = "1"
        # No explicit fast model configured; should probe the default cache path.
        path = badapple_fast_model.fast_model_path()
        if os.environ.get("BADAPPLE_FAST_MODEL"):
            self.assertEqual(path, os.environ["BADAPPLE_FAST_MODEL"])
        else:
            # Either a cached 0.5B path or None if the model is not downloaded.
            self.assertIsNoneOrCached(path)

    def assertIsNoneOrCached(self, path: str | None) -> None:
        if path is None:
            return
        self.assertTrue(Path(path).is_dir())

    @mock.patch("badapple_mlx_server.MLXServer._init_prompt_cache")
    @mock.patch("badapple_mlx_server.BadAppleKnowledge")
    @mock.patch("badapple_mlx_server.load")
    def test_ensure_main_model_loads_on_demand(self, mock_load, mock_knowledge, mock_init_prompt_cache) -> None:
        from tests._test_utils import fake_encoder

        os.environ["BADAPPLE_LAZY_MAIN_MODEL"] = "1"
        os.environ["BADAPPLE_P2P"] = "0"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            os.environ["BADAPPLE_DATA_DIR"] = tmp
            os.environ["BADAPPLE_SLICKS_KEY_PATH"] = str(Path(tmp) / "slicks.key")
            with open(os.environ["BADAPPLE_SLICKS_KEY_PATH"], "wb") as f:
                f.write(b"x" * 32)

            # Import here so env vars are respected.
            import badapple_mlx_server

            badapple_mlx_server.MAIN_MODEL = "dummy-model"
            mock_load.return_value = (object(), object())
            mock_knowledge.return_value._encode_texts = fake_encoder

            server = badapple_mlx_server.MLXServer(
                secret=b"x" * 32,
                system_prompt="You are Bad Apple.",
            )
            self.addCleanup(server.model_manager.shutdown)
            self.assertIsNone(server.model)
            self.assertIsNone(server.tokenizer)

            server._ensure_main_model()
            self.assertIsNotNone(server.model)
            self.assertIsNotNone(server.tokenizer)
            mock_load.assert_called_once_with("dummy-model")


if __name__ == "__main__":
    unittest.main()
