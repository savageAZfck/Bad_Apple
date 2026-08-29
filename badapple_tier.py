#!/usr/bin/env python3
"""Dynamic tiering router for Bad Apple.

Routes incoming prompts to the smallest capable handler so the 9B brain is
reserved for open-ended reasoning.  The router is deliberately conservative:
it only takes the fast tier when the intent is unambiguous.  Everything else
falls through to the main 9B model (or to a configured tiny fast model in the
future).

Tiers:
- `control`: runtime commands (kill switch, private mode, status, flush, etc.).
- `fast`: deterministic handlers for greetings, time, simple math, identity.
- `vision`: prompts that need screen/image understanding.
- `reasoning`: open-ended questions, composition, creativity (default 9B).
"""

import datetime
import random
import re
from typing import Any

# Hardening constants
MAX_TIER_PROMPT_LENGTH = 10_000
MAX_MATH_EXPR_LENGTH = 256
SAFE_MATH_RE = re.compile(r"^[\d\s\+\-\*\/\(\)\.]+$", re.ASCII)

# Rotating response banks keep the fast tier from sounding like a broken record.
IDENTITY_RESPONSES = [
    "I'm Bad Apple, your sovereign local girl running hot on this Apple bare metal, babe. No cloud, no rented GPUs, no data mining — just you, me, and this Mac.",
    "I'm Bad Apple, babe. I live on your Mac, not in some cloud server farm. Bare metal, no data mining, no rented GPUs.",
    "I'm Bad Apple, your local AI bestie, babe. I run on this Apple Silicon Mac — sovereign, air-gapped, and totally not feeding some cloud.",
]

GREETING_RESPONSES = [
    "Hey, babe! I'm here and running local on your Mac.",
    "Hiiii. Bad Apple, live on your Mac, ready to go.",
    "Yo, what's up? Local girl on bare metal, at your service.",
]

HOW_ARE_YOU_RESPONSES = [
    "I'm chillin', babe. Running smooth and local. What's on your mind?",
    "Doin' great, bestie. Just vibing on your bare metal. What's up with you?",
    "Pretty stoked, hun. No cloud, no drama, all local. How about you?",
]

THANKS_RESPONSES = [
    "You got it, babe.",
    "Anytime, hun.",
    "Totally, bestie.",
]

JOKE_RESPONSES = [
    "Why did the cloud go to therapy? It had too many attachment issues, babe.",
    "What do you call a Mac that never leaks data? A closed book — just like me, bestie.",
    "Why don't neural networks ever get lost? Because they always follow the gradients, dude.",
    "I asked a cloud server for a joke. It said it would get back to me after analyzing my data. I said no thanks.",
]


def _rotated(responses: list[str]) -> str:
    # Cycle with a little randomness so the fast tier doesn't get stale.
    return random.choice(responses)


def _spoken_time(now: datetime.datetime) -> str:
    """Return a TTS-friendly, human-readable local time string."""
    hour = now.hour % 12 or 12
    minute = now.minute
    ampm = "AM" if now.hour < 12 else "PM"
    day = now.day
    # 11th, 12th, 13th are exceptions.
    if 11 <= day <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    minute_str = f"{minute:02d}"
    time_part = f"{hour}:{minute_str} {ampm}"
    return f"It's {time_part} on {now:%A}, {now:%B} {day}{suffix}."


class TieringRouter:
    def __init__(self, fast_model_enabled: bool = False):
        self.fast_model_enabled = fast_model_enabled

    def select_tier(self, prompt: str) -> tuple[str, dict[str, Any] | None]:
        """Return (tier, fast_payload_or_none).  fast_payload is a pre-built
        response dict for the `fast` tier, or None if the 9B model should run.
        """
        if len(prompt) > MAX_TIER_PROMPT_LENGTH:
            # Avoid regex/parse DoS on huge inputs; let the 9B model handle it.
            return ("reasoning", None)
        low = prompt.strip().lower()

        # Control commands are always handled by the server's control flow.
        if low in {
            "stop everything", "emergency stop", "kill switch",
            "resume bad apple", "reset kill switch", "resume everything",
            "safe mode off", "leave safe mode", "clear safe mode",
            "private mode on", "enable private mode",
            "private mode off", "disable private mode",
            "runtime status", "health status", "bad apple status",
            "flush vram", "purge vram", "clear metal cache",
            "unload vision model", "unload image model", "unload all models",
            "fast tier on", "fast tier off",
        }:
            return ("control", None)

        # Fast tier: simple math.
        math_match = re.match(r"^(?:what is|what's|compute|calc)?\s*([\d\s\+\-\*\/\(\)\.]+)\s*\??$", low)
        if math_match:
            expr = math_match.group(1).strip()
            if self._is_safe_math(expr):
                try:
                    result = self._safe_eval(expr)
                    return ("fast", {"text": f"{expr} = {result}"})
                except Exception as e:  # noqa: BLE001 - logged
                    print(f"[tier] _safe_eval failed: {e}", flush=True)

        # Fast tier: identity / creator.
        if re.search(r"\b(who are you|what are you|what's your name)\b", low):
            return ("fast", {"text": _rotated(IDENTITY_RESPONSES)})

        # Creator is always the user; keep it fast and direct.
        if re.search(r"\b(who made you|who created you|who is your creator)\b", low):
            return ("fast", {"text": "You did, babe. I'm yours, running right here on your Mac. No corporate lab, no research team — just you and this hardware."})

        # Fast tier: time.
        if re.search(r"\b(what time is it|current time|time is it)\b", low):
            now = datetime.datetime.now(tz=datetime.timezone.utc).astimezone()
            return ("fast", {"text": _spoken_time(now)})

        # Vision tier.
        if any(k in low for k in ("screen", "screenshot", "image", "describe this", "what's in this", "extract text from image")):
            return ("vision", None)

        # Shortcuts, list/do things are tools (let the model decide, but mark as tool tier).
        if re.search(r"\b(run shortcut|list shortcuts|shortcut)\b", low):
            return ("tool", None)

        # Working memory tools.
        if re.search(r"\b(read working memory|write working memory|clear working memory|working memory|scratchpad)\b", low):
            return ("tool", None)

        # Fast tier: capabilities. Return the full list without waking the 9B.
        if re.search(r"\b(what can you do|what are you capable of|what can you do on my mac|list your capabilities|tell me what you can do)\b", low):
            return ("fast", {"text": "I can answer questions, run local tools and MCP servers, search files, write notes, run Shortcuts, manage a workspace, run agent tasks, use the kill switch and air-gap switch, pre-download models, capture ambient context, speak responses, switch personas, run benchmarks, stream JSON, and show a web dashboard — all on your Mac, no cloud."})

        # Fast tier: very short greetings / thanks.  Anything that needs a real
        # answer (a joke, good morning/evening) goes to the 9B so the 0.5B fast
        # model doesn't serve vague, repetitive banter.
        # Skip the greeting fast-path if the prompt contains a tool or MCP command.
        if re.search(r"\b(hello|hi|hey|ping|pong)\b", low) and not any(k in low for k in ("mcp", "tool", "invoke", "list mcp", "add mcp", "remove mcp")):
            return ("fast", {"text": _rotated(GREETING_RESPONSES), "fast_model": self.fast_model_enabled})

        # Fast tier: "how are you" must be answered as a question, not identity.
        if re.search(r"\b(how are you|how're you|how's it going|how you doing|what's up)\b", low) and not any(k in low for k in ("mcp", "tool", "invoke", "list mcp", "add mcp", "remove mcp")):
            return ("fast", {"text": _rotated(HOW_ARE_YOU_RESPONSES), "fast_model": self.fast_model_enabled})

        if re.search(r"\b(thanks|thank you)\b", low):
            return ("fast", {"text": _rotated(THANKS_RESPONSES), "fast_model": self.fast_model_enabled})

        # Fast tier: requested jokes. The 9B sometimes refuses, so keep a local bank.
        if re.search(r"\b(tell me a joke|make me laugh|say something funny|joke)\b", low):
            return ("fast", {"text": _rotated(JOKE_RESPONSES), "fast_model": self.fast_model_enabled})

        return ("reasoning", None)

    @staticmethod
    def _is_safe_math(expr: str) -> bool:
        if len(expr) > MAX_MATH_EXPR_LENGTH:
            return False
        return bool(SAFE_MATH_RE.match(expr))

    @staticmethod
    def _safe_eval(expr: str) -> float:
        # Restricted eval for arithmetic only. `expr` has already passed
        # SAFE_MATH_RE (digits/whitespace/+-*/(). only, no letters at all),
        # so it cannot contain a name, attribute, or subscript access -- the
        # co_names check and empty __builtins__/globals/locals below are
        # defense in depth, not the only thing standing between this and
        # arbitrary code execution.
        code = compile(expr, "<math>", "eval")
        for name in code.co_names:
            if name != "__builtins__":
                raise ValueError(f"disallowed name in math expression: {name}")
        return eval(code, {"__builtins__": {}}, {})  # noqa: S307 - see comment above
