"""Tests for badapple_vram_governor."""
import unittest
from unittest import mock

import badapple_vram_governor as vg


class TestVramGovernor(unittest.TestCase):
    def test_recommend_for_query_prefers_big_for_reasoning(self) -> None:
        rec = vg.recommend_model_for_query("deep reasoning about quantum physics", 64)
        self.assertEqual(rec, "main_70b")

    def test_recommend_for_query_falls_back_to_9b(self) -> None:
        rec = vg.recommend_model_for_query("deep reasoning", 12)
        self.assertEqual(rec, "main_9b")

    def test_recommend_for_query_fast_for_greeting(self) -> None:
        rec = vg.recommend_model_for_query("hi, what time is it?", 64)
        self.assertEqual(rec, "fast_0.5b")

    @mock.patch.object(vg, "_available_gb", return_value=4.0)
    def test_can_fit_model_refuses_too_large(self, _mock) -> None:
        ok, avail = vg.can_fit_model(64)
        self.assertFalse(ok)
        self.assertLessEqual(avail, 64)

    def test_recommend_for_memory(self) -> None:
        self.assertEqual(vg.recommend_for_memory(64), "main_70b")
        self.assertEqual(vg.recommend_for_memory(30), "main_32b")
        self.assertEqual(vg.recommend_for_memory(12), "main_9b")
        self.assertEqual(vg.recommend_for_memory(1.5), "fast_0.5b")


if __name__ == "__main__":
    unittest.main()
