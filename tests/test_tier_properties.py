#!/usr/bin/env python3
"""Property-style tests for the fast-tier router."""

import random
import string
import unittest

from badapple_tier import TieringRouter


class TierPropertyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = TieringRouter(fast_model_enabled=True)

    def _random_prompt(self, length: int) -> str:
        chars = string.ascii_letters + string.digits + string.punctuation + " \t\n"
        return "".join(random.choice(chars) for _ in range(length))

    def test_huge_prompts_never_crash(self) -> None:
        for length in (1000, 10_001, 50_000):
            prompt = self._random_prompt(length)
            tier, payload = self.router.select_tier(prompt)
            self.assertIn(tier, ("control", "fast", "vision", "tool", "reasoning"))
            self.assertTrue(payload is None or isinstance(payload, dict))

    def test_math_expressions_only_return_fast_when_safe(self) -> None:
        safe = ["2+2", "10 / 3 + (4 - 1)", "3.14 * 2"]
        for expr in safe:
            tier, payload = self.router.select_tier(f"what is {expr}")
            self.assertEqual(tier, "fast", f"{expr} should be fast")
            self.assertIsNotNone(payload)
            self.assertIn("=", payload["text"])

    def test_dangerous_math_expressions_are_not_evaled(self) -> None:
        for prompt in (
            "what is __import__('os').system('rm -rf /')",
            "what is open('/etc/passwd').read()",
            "what is (1).__class__",
        ):
            tier, _payload = self.router.select_tier(prompt)
            # These should fall through because the math regex does not match identifiers.
            self.assertNotEqual(tier, "fast", f"{prompt!r} must not be fast-tiered")

    def test_identity_and_capability_fast_paths(self) -> None:
        for prompt in ("who are you", "what can you do", "who made you"):
            tier, payload = self.router.select_tier(prompt)
            self.assertEqual(tier, "fast")
            self.assertIn("text", payload)

    def test_greetings_are_fast(self) -> None:
        tier, _payload = self.router.select_tier("hello")
        self.assertEqual(tier, "fast")

    def test_vision_queries_route_to_vision(self) -> None:
        for prompt in ("describe this image", "what is on my screen"):
            tier, _ = self.router.select_tier(prompt)
            self.assertEqual(tier, "vision")

    def test_control_commands_are_control(self) -> None:
        for prompt in ("emergency stop", "kill switch", "bad apple status"):
            tier, _ = self.router.select_tier(prompt)
            self.assertEqual(tier, "control")


if __name__ == "__main__":
    unittest.main()
