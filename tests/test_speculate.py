#!/usr/bin/env python3
"""Unit tests for speculative draft discovery."""

import json
import tempfile
import unittest
from pathlib import Path

import badapple_speculate


class SpeculateTests(unittest.TestCase):
    def test_scan_excludes_vision_models_and_oversized_models(self) -> None:
        """scan_draft_candidates skips vision models and models above max size."""
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "hub"
            # Small text model
            text_dir = hub / "models--mlx-community--Qwen2.5-0.5B-Instruct-4bit"
            text_snap = text_dir / "snapshots" / "abc"
            text_snap.mkdir(parents=True)
            (text_snap / "config.json").write_text(json.dumps({
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 896,
                "num_hidden_layers": 24,
            }))
            (text_snap / "model.safetensors").write_text("x" * 100_000)

            # Vision model
            vision_dir = hub / "models--mlx-community--Qwen2-VL-2B-Instruct-4bit"
            vision_snap = vision_dir / "snapshots" / "abc"
            vision_snap.mkdir(parents=True)
            (vision_snap / "config.json").write_text(json.dumps({
                "architectures": ["Qwen2VLForConditionalGeneration"],
                "vision_config": {},
            }))
            (vision_snap / "model.safetensors").write_text("x" * 100_000)

            # Oversized model
            big_dir = hub / "models--mlx-community--Qwen3.5-32B-MLX-4bit"
            big_snap = big_dir / "snapshots" / "abc"
            big_snap.mkdir(parents=True)
            (big_snap / "config.json").write_text(json.dumps({
                "architectures": ["Qwen2ForCausalLM"],
            }))
            (big_snap / "model.safetensors").write_text("x" * 2_000_000_000)

            orig = badapple_speculate._hub_root
            badapple_speculate._hub_root = lambda: hub
            try:
                candidates = badapple_speculate.scan_draft_candidates(max_size_gb=1.0)
                self.assertEqual(len(candidates), 1)
                self.assertIn("0.5B", candidates[0]["id"])
            finally:
                badapple_speculate._hub_root = orig

    def test_find_draft_candidate_prefers_qwen(self) -> None:
        """find_draft_candidate returns the best cached small model."""
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "hub"
            model_dir = hub / "models--mlx-community--Qwen2.5-0.5B-Instruct-4bit"
            snap = model_dir / "snapshots" / "abc"
            snap.mkdir(parents=True)
            (snap / "config.json").write_text(json.dumps({
                "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 24,
            }))
            (snap / "model.safetensors").write_text("x" * 100_000)

            orig = badapple_speculate._hub_root
            badapple_speculate._hub_root = lambda: hub
            try:
                candidate = badapple_speculate.find_draft_candidate(max_size_gb=1.0)
                self.assertIsNotNone(candidate)
                self.assertIn("0.5B", candidate)
            finally:
                badapple_speculate._hub_root = orig

    def test_no_candidate_returns_none(self) -> None:
        """find_draft_candidate returns None when the cache is empty."""
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "hub"
            hub.mkdir(parents=True)
            orig = badapple_speculate._hub_root
            badapple_speculate._hub_root = lambda: hub
            try:
                candidate = badapple_speculate.find_draft_candidate(max_size_gb=1.0)
                self.assertIsNone(candidate)
            finally:
                badapple_speculate._hub_root = orig


if __name__ == "__main__":
    unittest.main()
