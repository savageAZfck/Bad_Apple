#!/usr/bin/env python3
"""Local, on-device speech-to-text for Bad Apple.

Uses `mlx-audio` Whisper models. No cloud after the model is cached.
"""

import os
import tempfile
from pathlib import Path


def _available() -> bool:
    try:
        return True
    except Exception:  # noqa: BLE001 - catch-all wrapper
        return False


DEFAULT_STT_MODEL = os.environ.get(
    "BADAPPLE_STT_MODEL",
    "mlx-community/whisper-large-v3-turbo-asr-fp16",
)


def _cache_dir() -> Path:
    default = Path(os.environ.get("BADAPPLE_STT_CACHE", "/var/lib/bad_apple/stt_cache")).expanduser()
    for candidate in (default, Path("~/.bad_apple/stt_cache").expanduser(), Path(tempfile.gettempdir()) / "badapple_stt_cache"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except PermissionError:
            continue
    return default


def transcribe(
    audio_path: str,
    model_name: str = DEFAULT_STT_MODEL,
    language: str = "en",
    max_tokens: int = 8192,
) -> str:
    """Transcribe a local audio file (wav, mp3, m4a, etc.) to text."""
    if not _available():
        return "Error: mlx-audio is not installed."
    p = Path(audio_path).expanduser()
    if not p.is_file():
        return f"Error: {p} not found"
    try:
        from mlx_audio.stt.generate import generate_transcription
        _cache_dir().mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(_cache_dir())
        segments = generate_transcription(
            model=model_name,
            audio=str(p),
            output_path=str(_cache_dir() / "transcript"),
            format="txt",
            verbose=False,
            language=language,
            max_tokens=max_tokens,
        )
        return getattr(segments, "text", str(segments)).strip()
    except Exception as e:  # noqa: BLE001 - catch-all wrapper
        return f"Transcription error: {e}"
