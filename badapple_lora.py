#!/usr/bin/env python3
"""On-device personal fine-tuning (LoRA) for Bad Apple.

Wraps `mlx-lm` so the user can train a small, local, private LoRA adapter on
their own prompt/completion pairs. No data leaves the device.

Training fires `mlx_lm.lora` in a subprocess. The resulting adapter is saved to
`~/.bad_apple/lora_adapters/<name>` and can be used later for inference with
`mlx_lm.generate --adapter-path` or fused into the base model.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_BASE_MODEL = os.environ.get(
    "BADAPPLE_LORA_BASE_MODEL",
    "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit",
)
LORA_ROOT = Path(os.environ.get("BADAPPLE_DATA_DIR", "/var/lib/bad_apple")).expanduser()
LORA_DATA_DIR = Path(os.environ.get("BADAPPLE_LORA_DATA_DIR", str(LORA_ROOT / "lora_data"))).expanduser()
LORA_ADAPTERS_DIR = Path(os.environ.get("BADAPPLE_LORA_ADAPTERS_DIR", str(LORA_ROOT / "lora_adapters"))).expanduser()


def _ensure_dirs():
    try:
        LORA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        LORA_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        pass


def list_datasets() -> List[str]:
    _ensure_dirs()
    if not LORA_DATA_DIR.is_dir():
        return []
    return sorted([d.name for d in LORA_DATA_DIR.iterdir() if d.is_dir()])


def list_adapters() -> List[str]:
    _ensure_dirs()
    if not LORA_ADAPTERS_DIR.is_dir():
        return []
    return sorted([d.name for d in LORA_ADAPTERS_DIR.iterdir() if d.is_dir()])


def get_dataset_path(name: str) -> Path:
    _ensure_dirs()
    p = LORA_DATA_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_adapter_path(name: str) -> Path:
    _ensure_dirs()
    p = LORA_ADAPTERS_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_example(dataset: str, messages: List[Dict[str, str]]) -> str:
    """Append one chat-formatted example to a dataset."""
    p = get_dataset_path(dataset)
    with open(p / "train.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"messages": messages}, ensure_ascii=True) + "\n")
    return f"Added example to {dataset} (train.jsonl)"


def train(
    dataset: str,
    adapter: str,
    base_model: str = DEFAULT_BASE_MODEL,
    iters: int = 100,
    learning_rate: float = 1e-4,
    rank: int = 8,
    alpha: int = 16,
    num_layers: int = 8,
    batch_size: int = 1,
    max_seq_length: int = 512,
) -> str:
    """Run `mlx_lm.lora` on the given dataset and save the adapter."""
    _ensure_dirs()
    data_path = get_dataset_path(dataset)
    adapter_path = get_adapter_path(adapter)
    train_file = data_path / "train.jsonl"
    if not train_file.is_file():
        return f"Error: no training data at {train_file}"

    # mlx-lm always tries to load valid.jsonl and test.jsonl; give them one
    # dummy example so the files parse (training on a tiny dataset anyway).
    dummy = json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}) + "\n"
    for name in ("valid", "test"):
        fpath = data_path / f"{name}.jsonl"
        need = True
        if fpath.is_file():
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    first = next((l for l in f if l.strip()), None)
                need = first is None
            except Exception:
                pass
        if need:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(dummy)

    import sys
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", base_model,
        "--train",
        "--data", str(data_path),
        "--adapter-path", str(adapter_path),
        "--iters", str(iters),
        "--learning-rate", str(learning_rate),
        "--num-layers", str(num_layers),
        "--batch-size", str(batch_size),
        "--max-seq-length", str(max_seq_length),
        "--mask-prompt",
        "--save-every", str(max(10, iters)),
        "--grad-checkpoint",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=iters * 30,  # rough per-iteration ceiling
        )
        if result.returncode != 0:
            return f"LoRA training failed:\n{result.stderr or result.stdout}"
        return (
            f"LoRA adapter '{adapter}' trained on dataset '{dataset}' and saved to "
            f"{adapter_path}.\n\n{result.stdout[-800:]}"
        )
    except subprocess.TimeoutExpired:
        return f"LoRA training timed out after {iters * 30}s"
    except Exception as e:
        return f"LoRA training error: {e}"


def generate_with_adapter(
    adapter: str,
    prompt: str,
    base_model: str = DEFAULT_BASE_MODEL,
    max_tokens: int = 120,
) -> str:
    """Generate with a saved adapter using `mlx_lm.generate`."""
    adapter_path = get_adapter_path(adapter)
    if not adapter_path.is_dir():
        return f"Error: adapter '{adapter}' not found at {adapter_path}"

    import sys
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", base_model,
        "--adapter-path", str(adapter_path),
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0.7",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return f"LoRA generation failed:\n{result.stderr or result.stdout}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "LoRA generation timed out"
    except Exception as e:
        return f"LoRA generation error: {e}"


def get_summary() -> str:
    return (
        f"Datasets: {', '.join(list_datasets()) or 'none'}\n"
        f"Adapters: {', '.join(list_adapters()) or 'none'}"
    )
