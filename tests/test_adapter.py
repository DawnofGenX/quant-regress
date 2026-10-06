"""The adapter must build a genuinely quantized int8 model, or the tool lies.

The regression this file guards is a SILENT one. ``quantize_dynamic`` only
replaces module types present in its own
``get_default_dynamic_quant_module_mappings()`` table. GPT-2's transformer blocks
use ``transformers.pytorch_utils.Conv1D``, not ``nn.Linear``, so passing
``{torch.nn.Linear}`` marked nothing in the body and left the model entirely in
fp32 apart from one out-of-block ``lm_head``. The harness then compared a model
against itself and reported ``drop_points: 0.0`` with ``verdict: pass`` — a clean
bill of health for a model nobody quantized.

Two things are therefore asserted here:

1. an architecture whose weight layers cannot be swapped is REFUSED with an
   actionable message, rather than silently reported as quantized; and
2. an architecture that can be quantized actually has its body quantized
   (verified by counting swapped modules, not by trusting the call to have done
   something).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from quant_regress.adapter import (
    AdapterError,
    _count_quantized,
    _quantizable_weight_modules,
    build_model_factory,
)

#: GPT-2's blocks are Conv1D, so dynamic INT8 cannot reach the transformer body.
GPT2 = "hf-internal-testing/tiny-random-gpt2"
#: BERT-style blocks are nn.Linear, which torch's mapping does cover.
BERT = "hf-internal-testing/tiny-random-bert"


def _load(name: str):
    from transformers import AutoModel

    return AutoModel.from_pretrained(name)


# --------------------------------------------------------------------------
# The bug: a no-op quantization must be refused, not reported.
# --------------------------------------------------------------------------


def test_conv1d_architecture_is_detected_as_unquantizable():
    """GPT-2's weight-bearing Conv1D modules are invisible to quantize_dynamic."""
    blocked = _quantizable_weight_modules(_load(GPT2))
    assert "Conv1D" in blocked, (
        "expected Conv1D to be reported as a weight-bearing type torch cannot "
        f"replace; got {blocked}"
    )
    # Normalization must NOT be flagged — it is intentionally left in fp32.
    assert "LayerNorm" not in blocked


def test_linear_architecture_has_no_blocked_weight_modules():
    """BERT-style models must not be flagged, or the guard would be useless."""
    blocked = _quantizable_weight_modules(_load(BERT))
    assert blocked == {}, f"BERT should be fully quantizable, got {blocked}"


def test_unsupported_architecture_raises_instead_of_silently_no_op():
    """The discriminating case: fp32 must build, int8 must REFUSE loudly."""
    factory = build_model_factory(model_name=GPT2)

    # fp32 is fine — nothing is being quantized.
    assert factory("fp32") is not None

    with pytest.raises(AdapterError) as exc:
        factory("int8")
    msg = str(exc.value)
    # The message must be actionable, not just an exception.
    assert "Conv1D" in msg, "error must name the blocking module type"
    assert "itself" in msg or "fp32" in msg, "error must explain the false-pass risk"


def test_supported_architecture_really_quantizes():
    """BERT-style: the body must actually be quantized, verified by counting."""
    factory = build_model_factory(model_name=BERT)
    int8 = factory("int8")
    swapped = _count_quantized(int8)
    assert swapped >= 8, (
        f"only {swapped} module(s) quantized — the gate would compare a model "
        "against itself"
    )


# --------------------------------------------------------------------------
# Unchanged contracts.
# --------------------------------------------------------------------------


def test_fp32_contains_no_packed_params():
    int8_modules = list(build_model_factory(model_name=BERT)("int8").modules())
    assert any("packed" in type(m).__name__.lower() for m in int8_modules)


def test_factory_rejects_unknown_precision():
    factory = build_model_factory(model_name=BERT)
    with pytest.raises(ValueError, match="unsupported precision"):
        factory("int4")


def test_empty_model_name_is_rejected_with_actionable_message():
    with pytest.raises(AdapterError, match="model"):
        build_model_factory(model_name="")


def test_unsupported_architecture_is_exit_code_2_not_a_crash():
    """A refusal must reach the CLI as configuration error (2), never 0 or 1.

    Exit 0 would be the catastrophic outcome: a green build for a model that was
    never quantized.
    """
    from quant_regress.cli import EXIT_USAGE, main

    import json as _json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        ev = Path(d) / "e.jsonl"
        ev.write_text(
            "\n".join(_json.dumps({"id": f"c{i}", "prompt": "x", "expected": "y"})
                      for i in range(3)),
            encoding="utf-8",
        )
        rc = main(["--eval", str(ev), "--model", GPT2, "--max-drop-points", "2"])
    assert rc == EXIT_USAGE, f"expected exit {EXIT_USAGE} (misconfigured), got {rc}"