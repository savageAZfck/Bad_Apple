#!/usr/bin/env python3
"""Lightweight, deterministic fact extraction from user messages.

Uses regex patterns to pull self-contained facts out of free-form text and
rewrites them into third-person statements that the memory graph can embed
and retrieve.
"""

import re

# (pattern, group indices that form the fact, template with {g0}, {g1}, ...)
_FACT_PATTERNS = [
    # Name
    (re.compile(r"\bmy name is\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0], "User's name is {g0}."),
    (re.compile(r"\bmy name'?s\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0], "User's name is {g0}."),
    (re.compile(r"\bi am\s+([a-z][a-z0-9 ]{1,40})(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0], "User's name is {g0}."),
    (re.compile(r"\bcall me\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0], "User's name is {g0}."),
    # Likes / loves / prefers / hates
    (re.compile(r"\bi (like|love|enjoy|prefer|hate|dislike)\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0, 1], "User {g0}s {g1}."),
    (re.compile(r"\bmy favorite\s+(.+?)\s+is\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0, 1], "User's favorite {g0} is {g1}."),
    # Work / job
    (re.compile(r"\bi work at\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0], "User works at {g0}."),
    (re.compile(r"\bi work for\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0], "User works at {g0}."),
    (re.compile(r"\bmy job is\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0], "User's job is {g0}."),
    (re.compile(r"\bi am a\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0], "User is a {g0}."),
    # Location
    (re.compile(r"\bi live in\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0], "User lives in {g0}."),
    (re.compile(r"\bi'm from\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0], "User is from {g0}."),
    # Age / birthday
    (re.compile(r"\bi am\s+(\d{1,3})\s+years? old", re.IGNORECASE), [0], "User is {g0} years old."),
    (re.compile(r"\bmy birthday is\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0], "User's birthday is {g0}."),
    # Family / pets
    (re.compile(r"\bmy (wife|husband|partner|girlfriend|boyfriend|spouse) is\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0, 1], "User's {g0} is {g1}."),
    (re.compile(r"\bi have a\s+(dog|cat|pet|son|daughter|kid|child)\s+(?:named|called)\s+(.+?)(?:\.|$|\b(?:and|but|or)\b)", re.IGNORECASE), [0, 1], "User has a {g0} named {g1}."),
    # Explicit remember is intentionally omitted; the other patterns handle
    # "remember that I like pizza" directly and produce a cleaner fact.
]


def _clean(fragment: str) -> str:
    """Trim and remove trailing punctuation/artifacts."""
    fragment = fragment.strip().rstrip(".!?,;:\"'")
    # Limit to a reasonable sentence fragment.
    if len(fragment) > 240:
        fragment = fragment[:240].rstrip() + "..."
    return fragment


def extract_facts(text: str) -> list[str]:
    """Return a list of normalized fact strings."""
    facts = []
    seen = set()
    for pattern, groups, template in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            parts = [_clean(match.group(g + 1)) for g in groups]
            if any(not p for p in parts):
                continue
            # Lowercase the verb only for like/love/prefer/hate to avoid odd "Likes"
            rendered = template
            for i, p in enumerate(parts):
                rendered = rendered.replace(f"{{g{i}}}", p)
            # Normalize first-person leftovers like "I" inside the fact
            rendered = re.sub(r"\bi\b", "User", rendered, flags=re.IGNORECASE)
            rendered = re.sub(r"\bmy\b", "User's", rendered, flags=re.IGNORECASE)
            rendered = re.sub(r"\bme\b", "User", rendered, flags=re.IGNORECASE)
            rendered = re.sub(r"\bmyself\b", "User", rendered, flags=re.IGNORECASE)
            rendered = rendered.strip()
            if rendered and rendered.lower() not in seen:
                facts.append(rendered)
                seen.add(rendered.lower())
    return facts
