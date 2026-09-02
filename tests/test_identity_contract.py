#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from badapple_extras import BAD_APPLE_IDENTITY_CONTRACT, PersonaPack
from badapple_tier import TieringRouter


class IdentityContractTests(unittest.TestCase):
    def test_contract_defines_os_identity_and_model_boundary(self) -> None:
        self.assertIn("local AI operating system layer", BAD_APPLE_IDENTITY_CONTRACT)
        self.assertIn("Qwen, MLX", BAD_APPLE_IDENTITY_CONTRACT)
        self.assertIn("not merely a text LLM", BAD_APPLE_IDENTITY_CONTRACT)
        self.assertIn("AI wrapper", BAD_APPLE_IDENTITY_CONTRACT)

    def test_persona_prompt_always_starts_with_os_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt_file = root / "prompt.txt"
            prompt_file.write_text("Use a concise test persona.", encoding="utf-8")
            pack = PersonaPack(root, prompt_file)
            prompt = pack.get_system_prompt()
            self.assertIn("You are Bad Apple, a local AI operating system layer for macOS", prompt)
            self.assertIn("PERSONA", prompt)
            self.assertIn("Use a concise test persona.", prompt)

    def test_wrapper_question_is_deterministic_fast_path(self) -> None:
        router = TieringRouter(fast_model_enabled=False)
        tier, payload = router.select_tier("Are you an AI wrapper or a text LLM?")
        self.assertEqual(tier, "fast")
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertIn("local AI operating system layer", payload["text"])
        self.assertIn("not an AI wrapper", payload["text"])

    def test_identity_fast_responses_name_the_os(self) -> None:
        router = TieringRouter(fast_model_enabled=False)
        tier, payload = router.select_tier("Who are you?")
        self.assertEqual(tier, "fast")
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(
            "operating system" in payload["text"] or "AI OS" in payload["text"],
            payload["text"],
        )

    def test_developer_workspace_is_a_first_class_capability(self) -> None:
        router = TieringRouter(fast_model_enabled=False)
        for question in (
            "Can you code?",
            "Are you a sovereign dev workspace?",
            "Are you a sovereign developer workspace?",
            "Why do you say you can't code?",
        ):
            tier, payload = router.select_tier(question)
            self.assertEqual(tier, "fast")
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertIn("sovereign local developer workspace", payload["text"])
            self.assertIn("write", payload["text"])
            self.assertIn("debug", payload["text"])
            self.assertNotIn("can't code", payload["text"].lower())
            self.assertNotIn("cannot code", payload["text"].lower())

        tier, _ = router.select_tier("Write code to parse a JSON file")
        self.assertEqual(tier, "reasoning")

    def test_swift_prompt_manager_contains_same_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        prompt_source = (root / "src/platform/apple_desktop/BadAppleConversation.swift").read_text(encoding="utf-8")
        engine_source = (root / "src/platform/apple_desktop/BadAppleEngine.swift").read_text(encoding="utf-8")
        self.assertIn("local AI operating system layer for macOS", prompt_source)
        self.assertIn("Qwen, MLX", prompt_source)
        self.assertIn("not merely a text LLM", prompt_source)
        self.assertIn("sovereign developer workspace", prompt_source)
        self.assertIn("not an AI wrapper", engine_source)
        self.assertIn("language-model component inside me", engine_source)
        self.assertIn("I can inspect, write, refactor, build, test, and debug code", engine_source)


if __name__ == "__main__":
    unittest.main()
