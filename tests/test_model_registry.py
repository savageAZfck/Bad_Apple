#!/usr/bin/env python3
"""Unit tests for the model registry and recommendation logic."""

import tempfile
import unittest
from pathlib import Path

import badapple_model_registry


def _set_hub_root(path: Path) -> Path:
    """Temporarily point the registry at a fake or empty HF cache."""
    original = badapple_model_registry.HUB_ROOT
    badapple_model_registry.HUB_ROOT = path
    return original


class RecommendedModelsTests(unittest.TestCase):
    def test_recommended_models_are_well_formed(self) -> None:
        """Every recommended entry must have the required fields."""
        for m in badapple_model_registry.RECOMMENDED_MODELS:
            self.assertTrue(m["id"])
            self.assertTrue(m["name"])
            self.assertGreater(m["size_gb"], 0)
            self.assertGreater(m["memory_gb"], 0)


class ModelRegistryTests(unittest.TestCase):
    def test_registry_with_empty_data_dir(self) -> None:
        """A fresh data directory with an empty cache starts with no models."""
        with tempfile.TemporaryDirectory() as hub, tempfile.TemporaryDirectory() as data:
            orig = _set_hub_root(Path(hub))
            try:
                reg = badapple_model_registry.ModelRegistry(Path(data))
                self.assertEqual(reg._state["models"], [])
                self.assertIn("No models found", reg.list_models())
            finally:
                badapple_model_registry.HUB_ROOT = orig

    def test_scan_finds_cached_models(self) -> None:
        """scan() discovers a fake cached model directory."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_hub = Path(tmp) / "hub"
            model_dir = fake_hub / "models--mlx-community--Qwen2.5-0.5B-Instruct-4bit"
            (model_dir / "blobs").mkdir(parents=True)
            snapshot = model_dir / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
            (snapshot / "tokenizer.json").write_text('"{}"')
            (snapshot / "tokenizer_config.json").write_text("{}")
            (snapshot / "model.safetensors").write_text("fake")

            orig = _set_hub_root(fake_hub)
            try:
                with tempfile.TemporaryDirectory() as data:
                    reg = badapple_model_registry.ModelRegistry(Path(data))
                    result = reg.scan()
                    self.assertIn("Found 1 local model", result)
                    self.assertEqual(len(reg._state["models"]), 1)
                    self.assertIn("qwen2.5-0.5b", reg._state["models"][0]["id"].lower())
                    info = reg.info("qwen2.5-0.5b")
                    self.assertTrue("Qwen2ForCausalLM" in info or "qwen2.5" in info.lower())
            finally:
                badapple_model_registry.HUB_ROOT = orig

    def test_recommend_lists_defaults(self) -> None:
        """recommend() always returns the curated recommended list."""
        with tempfile.TemporaryDirectory() as hub, tempfile.TemporaryDirectory() as data:
            orig = _set_hub_root(Path(hub))
            try:
                reg = badapple_model_registry.ModelRegistry(Path(data))
                rec = reg.recommend()
                self.assertIn("Qwen 3.5 9B", rec)
                self.assertIn("Qwen2.5-0.5B", rec)
            finally:
                badapple_model_registry.HUB_ROOT = orig

    def test_set_current_validates_against_cache(self) -> None:
        """set_current() only switches to a model present in the registry."""
        with tempfile.TemporaryDirectory() as hub, tempfile.TemporaryDirectory() as data:
            orig = _set_hub_root(Path(hub))
            try:
                reg = badapple_model_registry.ModelRegistry(Path(data))
                self.assertFalse(reg.set_current("not-in-cache"))
                self.assertNotEqual(reg._state.get("current"), "not-in-cache")
            finally:
                badapple_model_registry.HUB_ROOT = orig


if __name__ == "__main__":
    unittest.main()
