#!/usr/bin/env python3
"""Convert the bge-small-en-v1.5 embedding model to CoreML for Neural Engine use.

STATUS: blocked by tooling incompatibility, not yet a working conversion.
See docs/ANE_OFFLOAD_ATTEMPT.md for the full investigation: this gets past
an initial BertEmbeddings tracing failure (worked around below via explicit
position_ids/token_type_ids) but currently fails during MIL conversion with
`NotImplementedError: PyTorch convert function for op 'new_ones' not
implemented`, coming from a newer transformers version's dynamic attention
mask construction. Kept here as a documented starting point, not a
guaranteed-working tool -- do not assume it produces a usable .mlpackage
without re-verifying against whatever coremltools/transformers versions are
installed when you pick this back up.

Bad Apple's semantic cache and RAG retrieval currently run BAAI/bge-small-en-v1.5
through PyTorch on CPU (badapple_extras.SemanticCache._encode). That's pure CPU
work sitting next to an MLX GPU workload doing the heavy lifting for the main
9B model -- the Neural Engine sits idle the entire time. This script is meant
to convert the same weights to a CoreML model that can run on the ANE, so
embedding lookups stop competing with the CPU/GPU for cycles used by the main
brain -- once the conversion incompatibility above is resolved.

Usage:
    .venv/bin/pip install coremltools
    .venv/bin/python tools/convert_embedding_to_coreml.py

Intended output (once working):
    tools/artifacts/bge_small_en_v1_5.mlpackage
"""

from __future__ import annotations

import sys
from pathlib import Path

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_SEQ_LEN = 128  # SemanticCache truncates at 512, but cache queries are short;
                    # 128 covers the vast majority and keeps the ANE graph small.
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
OUTPUT_PATH = ARTIFACTS_DIR / "bge_small_en_v1_5.mlpackage"


def main() -> int:
    import coremltools as ct
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {MODEL_NAME} (PyTorch)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, return_dict=False)
    model.eval()

    print(f"Tracing with a fixed {MAX_SEQ_LEN}-token input shape...")
    dummy_text = "trace input"
    encoded = tokenizer(
        [dummy_text],
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    # Passing position_ids/token_type_ids explicitly bypasses BertEmbeddings'
    # internal dynamic buffer slicing + int() cast of the sequence length,
    # which is what trips up coremltools' tracer on newer transformers
    # versions (a known "only 0-dimensional arrays..." conversion failure).
    seq_len = encoded["input_ids"].shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.zeros_like(encoded["input_ids"])

    class _TracableWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids, position_ids):
            return self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )

    wrapped = _TracableWrapper(model)
    wrapped.eval()
    example_inputs = (encoded["input_ids"], encoded["attention_mask"], token_type_ids, position_ids)

    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example_inputs, strict=False)

    print("Converting to CoreML (targeting CPU + Neural Engine)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=example_inputs[0].shape, dtype=ct.converters.mil.mil.types.int32),
            ct.TensorType(name="attention_mask", shape=example_inputs[1].shape, dtype=ct.converters.mil.mil.types.int32),
            ct.TensorType(name="token_type_ids", shape=example_inputs[2].shape, dtype=ct.converters.mil.mil.types.int32),
            ct.TensorType(name="position_ids", shape=example_inputs[3].shape, dtype=ct.converters.mil.mil.types.int32),
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS13,
    )

    mlmodel.short_description = "bge-small-en-v1.5 sentence embeddings (Bad Apple ANE offload)"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(OUTPUT_PATH))
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Fixed sequence length: {MAX_SEQ_LEN} tokens (shorter inputs are padded, longer are truncated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
