#!/usr/bin/env python3
"""Local, on-device text translation for Bad Apple.

Uses the small, multilingual `facebook/m2m100_418M` model with the
`transformers` library. The model downloads once and is cached locally.
No cloud after first use.
"""

import os
import time
from pathlib import Path

DEFAULT_MODEL = os.environ.get(
    "BADAPPLE_TRANSLATION_MODEL",
    "facebook/m2m100_418M",
)


def _cache_dir() -> Path:
    d = Path(os.environ.get("BADAPPLE_TRANSLATION_CACHE", "/var/lib/bad_apple/translation_cache")).expanduser()
    import tempfile
    for candidate in (d, Path("~/.bad_apple/translation_cache").expanduser(), Path(tempfile.gettempdir()) / "badapple_translation_cache"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except PermissionError:
            continue
    return d


_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            print("[translate] loading translation model...", flush=True)
            t0 = time.time()
            _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, cache_dir=str(_cache_dir()))
            _model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL, cache_dir=str(_cache_dir()))
            print(f"[translate] translation model loaded in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            return f"Translation model load error: {e}"
    return _model, _tokenizer


def translate(
    text: str,
    target: str = "en",
    source: str = "en",
    max_length: int = 256,
) -> str:
    """Translate text between languages. Use ISO 639-1 codes (en, fr, de, es, ...)."""
    if not text:
        return "Error: no text to translate"
    loaded = _load()
    if isinstance(loaded, str):
        return loaded
    model, tokenizer = loaded
    try:
        tokenizer.src_lang = source
        encoded = tokenizer(text, return_tensors="pt")
        generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target), max_length=max_length)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    except Exception as e:
        return f"Translation error: {e}"
