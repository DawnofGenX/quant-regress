"""Build real models at a requested precision.

Only dynamic ``quantize_dynamic`` INT8 is supported. That is deliberate:

* it needs no extra dependency (no autoawq / optimum / onnxruntime),
* it emits CPU-only kernels, which is what CI runners have, and
* it is the exact method that produced the motivating INT8 accuracy collapse.

Attempting to move a quantized model to CUDA raises at forward time, so the
device is pinned to CPU and we say so rather than letting it fail obscurely.

**Not every architecture can actually be quantized by this method**, and the
failure is silent — see ``_quantizable_weight_modules`` below.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

SUPPORTED = ("fp32", "int8")

#: Module types that carry weights but are intentionally left in FP32 by dynamic
#: quantization. Normalization layers are cheap and quantizing them is
#: counter-productive, so their presence is NOT a sign of an unsupported model.
_FP32_BY_DESIGN = ("LayerNorm", "GroupNorm", "RMSNorm", "Embedding", "LayerNorm2d")

#: A model whose *body* yields fewer quantized modules than this is treated as
#: not-quantizable rather than quantized-and-fine. Calibrated against real
#: checkpoints: BERT/DeBERTa/Llama swap 72-224 modules; GPT-2 swaps exactly 1
#: (an out-of-block lm_head) because its blocks use ``transformers.Conv1D``.
_MIN_QUANTIZED_MODULES = 8


class AdapterError(RuntimeError):
    """Raised with an actionable message when a model cannot be built."""


def _quantizable_weight_modules(model) -> dict[str, int]:
    """Weight-bearing module types that ``quantize_dynamic`` CANNOT swap.

    ``quantize_dynamic`` only replaces module types that appear in its own
    ``get_default_dynamic_quant_module_mappings()`` table. Passing an extra type
    in ``qconfig_spec`` merely *marks* those modules; ``convert()`` still cannot
    replace them. So adding a type to the set is not enough — the type has to be
    in torch's mapping too.

    Returns ``{type_name: count}`` for weight-bearing modules that are absent
    from that table, excluding types left in FP32 on purpose.
    """
    mapping = get_default_dynamic_quant_module_mappings()
    blocked: dict[str, int] = {}
    for _, mod in model.named_modules():
        cls = type(mod)
        own_params = sum(p.numel() for p in mod.parameters(recurse=False))
        if own_params == 0:
            continue
        if cls in mapping:
            continue
        if any(cls.__name__ == ok or cls.__name__.endswith(ok) for ok in _FP32_BY_DESIGN):
            continue
        if "quantized" in cls.__module__:
            continue
        name = cls.__name__
        blocked[name] = blocked.get(name, 0) + 1
    return blocked


def _count_quantized(model) -> int:
    """Modules actually swapped for a quantized implementation."""
    mapping = get_default_dynamic_quant_module_mappings()
    total = 0
    for _, mod in model.named_modules():
        cls = type(mod)
        if cls in mapping.values() and cls.__module__.startswith("torch.ao.nn.quantized"):
            total += 1
    return total


def _quantize_int8(model):
    """Dynamic INT8 on Linear layers; embeddings/LayerNorm stay FP32.

    Raises :class:`AdapterError` when the architecture's weight-bearing blocks
    cannot be swapped by ``quantize_dynamic`` (GPT-2 and friends use
    ``transformers.pytorch_utils.Conv1D``). Quantizing anyway would produce an
    "int8" arm that is really fp32, and the harness would then compare a model
    against itself and report a 0.00-point drop — a silent pass on an
    unquantized model, which is the worst possible failure for a gate.
    """
    blocked = _quantizable_weight_modules(model)
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    swapped = _count_quantized(quantized)

    if swapped < _MIN_QUANTIZED_MODULES and blocked:
        detail = ", ".join(f"{n} x {k}" for k, n in sorted(blocked.items()))
        raise AdapterError(
            f"cannot quantize this architecture with dynamic INT8: its weight "
            f"layers use module types torch's quantize_dynamic cannot replace "
            f"({detail}). Only {swapped} module(s) were quantized, so the "
            f"'int8' arm would be fp32 in the transformer body and the "
            f"comparison would measure a model against itself. "
            f"Use an nn.Linear-based architecture (BERT/DeBERTa/Llama/Mistral), "
            f"or quantize it outside this tool and pass a pre-quantized model."
        )
    return quantized


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
