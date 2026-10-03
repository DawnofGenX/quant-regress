"""Build real models at a requested precision.

Only dynamic ``quantize_dynamic`` INT8 is supported. That is deliberate:

* it needs no extra dependency (no autoawq / optimum / onnxruntime),
* it emits CPU-only kernels, which is what CI runners have, and
* it is the exact method that produced the motivating INT8 accuracy collapse.

Attempting to move a quantized model to CUDA raises at forward time, so the
device is pinned to CPU and we say so rather than letting it fail obscurely.
"""

from __future__ import annotations

from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SUPPORTED = ("fp32", "int8")


class AdapterError(RuntimeError):
    """Raised with an actionable message when a model cannot be built."""


def _quantize_int8(model):
    """Dynamic INT8 on Linear layers; embeddings/LayerNorm stay FP32."""
    return torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )


def build_model_factory(
    model_name: str,
    cache_dir: str | None = None,
    max_new_tokens: int = 32,
) -> Callable[[str], object]:
    if not model_name or not model_name.strip():
        raise AdapterError(
            "no model specified. Pass model: to the action, e.g. "
            "model: meta-llama/Llama-3.2-1B-Instruct"
        )
    name = model_name.strip()

    def factory(precision: str):
        p = precision.lower().strip()
        if p not in SUPPORTED:
            raise ValueError(
                f"unsupported precision {precision!r}; supported: {SUPPORTED}"
            )
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        try:
            tok = AutoTokenizer.from_pretrained(name, **kwargs)
            model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        except Exception as exc:
            raise AdapterError(
                f"could not load {name!r}: {type(exc).__name__}: {exc}. "
                "Check the model id, and that the runner has disk space "
                "(a 1B model is ~2 GB, a 7B model is ~14 GB)."
            ) from exc
        if p == "int8":
            model = _quantize_int8(model)
        # quantize_dynamic emits CPU-only kernels.
        model = model.to("cpu").eval()
        model.tokenizer = tok  # attached for the harness
        model.max_new_tokens = max_new_tokens
        return model

    return factory
