"""Shared test helpers to avoid loading heavy models repeatedly."""
from __future__ import annotations

import re


class _Vec(list):
    """list subclass with .tolist() so MemoryGraph can use it like a numpy array."""

    def tolist(self) -> list:
        return list(self)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def fake_encoder(texts: list[str]) -> list[_Vec]:
    """Return a 384-d bag-of-words-like embedding keyed on word hashes.

    Enough for unit tests: documents that share words get a high dot product
    without loading the real BGE model.
    """
    dim = 384
    out = []
    for text in texts:
        vec = [0.0] * dim
        for w in _words(text):
            h = hash(w) % dim
            vec[h] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        out.append(_Vec(x / norm for x in vec))
    return out
