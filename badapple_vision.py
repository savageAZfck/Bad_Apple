#!/usr/bin/env python3
"""Local, on-device vision/screen understanding using an MLX VLM.

Everything stays on the Mac: screen capture, image encoding, and the
language-vision model run through `mlx-vlm`. No cloud, no network once the
model weights are cached.
"""

import base64
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import List, Optional

import mlx.core as mx

_VLM_LOADED = False
try:
    from mlx_vlm import load as _vlm_load
    from mlx_vlm import generate as _vlm_generate
    _VLM_LOADED = True
except Exception:
    _vlm_load = None
    _vlm_generate = None

DEFAULT_VLM_MODEL = os.environ.get("BADAPPLE_VLM_MODEL", "mlx-community/Qwen2-VL-2B-Instruct-4bit")


def _model_cache_dir() -> Path:
    return Path(os.environ.get("BADAPPLE_VLM_CACHE", "/var/lib/bad_apple/vlm_cache")).expanduser()


class VisionHost:
    """Lazy-loaded VLM host for screen and image understanding."""

    def __init__(self, model_name: str = DEFAULT_VLM_MODEL):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._cache_dir = _model_cache_dir()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if self._model is not None:
            return
        if not _VLM_LOADED:
            raise RuntimeError("mlx-vlm is not installed")
        print(f"[vision] loading VLM {self.model_name}...", flush=True)
        t0 = time.time()
        # mlx-vlm load returns (model, processor)
        self._model, self._processor = _vlm_load(self.model_name)
        print(f"[vision] VLM loaded in {time.time() - t0:.1f}s", flush=True)

    def describe(self, image_path: Path, prompt: str = "Describe this image.", max_tokens: int = 256) -> str:
        self._load()
        return self._generate(image_path, prompt, max_tokens)

    def extract_text(self, image_path: Path, max_tokens: int = 256) -> str:
        """Return the text visible in an image or screenshot."""
        self._load()
        prompt = (
            "Extract and return only the text visible in this image. "
            "Preserve line breaks and structure as closely as possible. "
            "Do not describe the image. If no text is visible, say 'No text found.'"
        )
        return self._generate(image_path, prompt, max_tokens)

    def _generate(self, image_path: Path, prompt: str, max_tokens: int) -> str:
        try:
            result = _vlm_generate(
                self._model,
                self._processor,
                image=str(image_path),
                prompt=prompt,
                max_tokens=max_tokens,
                temp=0.0,
                verbose=False,
            )
            text = getattr(result, "text", None) or str(result)
            return text.strip()
        except Exception as e:
            traceback.print_exc()
            return f"Vision error: {e}"


def _console_user() -> Optional[str]:
    try:
        r = subprocess.run(
            ["stat", "-f", "%Su", "/dev/console"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def _run_as_user(cmd: List[str], user: Optional[str] = None, input_text: Optional[str] = None, timeout: int = 30):
    target = user or _console_user()
    if target and target != "root":
        full = ["sudo", "-n", "-u", target] + cmd
    else:
        full = cmd
    try:
        return subprocess.run(
            full,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return type("TimeoutResult", (), {"returncode": -1, "stdout": "", "stderr": f"timed out after {timeout}s"})()


def capture_screen(path: Optional[Path] = None) -> Path:
    """Capture the full main screen to a PNG using macOS screencapture."""
    if path is None:
        path = Path(tempfile.gettempdir()) / "badapple_screen.png"
    # -x suppresses the screenshot sound; -C captures the cursor; -S is main screen.
    result = _run_as_user(["screencapture", "-x", "-S", str(path)], timeout=15)
    if result.returncode != 0:
        raise RuntimeError(f"screencapture failed: {result.stderr or result.stdout}")
    return path


# Shared, lazy instance.
_vision_host: Optional[VisionHost] = None


def get_vision_host(model_name: str = DEFAULT_VLM_MODEL) -> VisionHost:
    global _vision_host
    if _vision_host is None:
        _vision_host = VisionHost(model_name)
    return _vision_host


def set_vision_host(host: VisionHost):
    global _vision_host
    _vision_host = host
