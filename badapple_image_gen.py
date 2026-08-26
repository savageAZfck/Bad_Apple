#!/usr/bin/env python3
"""Local, on-device image generation for Bad Apple using MFLUX.

Runs `mflux-generate-flux2` with the lightweight FLUX.2-klein-4B model.
The model is downloaded once and cached locally. No cloud after caching.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = os.environ.get("BADAPPLE_IMAGE_MODEL", "flux2-klein-4b")


def _mflux_cmd() -> str:
    base = Path(sys.executable).parent
    cmd = base / "mflux-generate-flux2"
    if cmd.is_file():
        return str(cmd)
    return "mflux-generate-flux2"


def _output_dir() -> Path:
    data_dir = Path(os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple"))
    out_dir = data_dir / "generated_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def generate(
    prompt: str,
    output: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    quantize: int = 4,
    seed: Optional[int] = None,
    low_ram: bool = True,
) -> str:
    """Generate an image from a prompt and return the output path."""
    exe = _mflux_cmd()
    if not Path(exe).is_file():
        return "Error: mflux is not installed."
    if not prompt:
        return "Error: prompt is required"

    if output is None:
        output = str(_output_dir() / f"badapple_gen_{(seed or 0)}_{os.getpid()}.png")

    cmd = [
        exe,
        "--model", DEFAULT_MODEL,
        "--prompt", prompt,
        "--output", output,
        "--width", str(width),
        "--height", str(height),
        "--steps", str(steps),
        "--quantize", str(quantize),
        "--no-metadata",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if low_ram:
        cmd.append("--low-ram")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode != 0:
            return f"Image generation failed:\n{result.stderr or result.stdout}"
        # The CLI prints a lot; just report the file path.
        return f"Generated image: {output}"
    except subprocess.TimeoutExpired:
        return "Image generation timed out"
    except Exception as e:
        return f"Image generation error: {e}"
