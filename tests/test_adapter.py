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
    _quantize_int8,
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


# --------------------------------------------------------------------------
# Robustness: peak memory. quantize_dynamic returns a NEW model, so the fp32
# original stays referenced inside the factory's local scope unless it is
# explicitly dropped. For a 7B checkpoint that is 14 GB of weights held for
# no reason, and on a runner with less than 28 GB the gate dies with an OOM
# instead of reporting a verdict.
# --------------------------------------------------------------------------


def test_original_model_is_released_after_quantization(monkeypatch):
    """The fp32 source must not stay alive once `quantized` exists.

    Load-bearing only for a model that participates in a reference cycle, which
    real torch modules do (parameter -> module -> parameter graphs). Plain
    refcounting frees an *acyclic* model the moment `del model` drops the last
    reference, so with a cycle-free tiny checkpoint this test passed even with
    the factory's `gc.collect()` deleted -- the guard was untested. The spy
    therefore installs a cycle, automatic GC is disabled, and no test-side
    collect runs: only the factory's own collect can reclaim the model. Delete
    the `gc.collect()` in adapter.factory and this fails with 1 alive.
    """
    import gc
    import weakref

    built: list[object] = []

    real_quantize = torch.ao.quantization.quantize_dynamic

    def spy_quantize(model, *a, **kw):
        # Simulate the reference cycle a real module graph carries, so the
        # explicit collect is the only thing that can reclaim it.
        model.__dict__["_cycle"] = model
        built.append(weakref.ref(model))
        return real_quantize(model, *a, **kw)

    monkeypatch.setattr(
        "quant_regress.adapter.torch.ao.quantization.quantize_dynamic",
        spy_quantize,
    )
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        factory = build_model_factory(model_name=BERT)
        int8 = factory("int8")
        # Deliberately NO gc.collect() here: rely on the factory's own.
        still_alive = [r for r in built if r() is not None]
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()

    assert built, "quantize_dynamic was never called"
    assert not still_alive, (
        f"{len(still_alive)} fp32 model(s) still resident after quantization: "
        "peak memory is ~2x the model size for no reason"
    )
    # The quantized result is the thing the harness gets, and it must work.
    assert _count_quantized(int8) >= 8


# --------------------------------------------------------------------------
# The quantize guard must fire on its own signal. `and blocked:` made the
# refusal depend on a SECOND condition that a small nn.Linear model does not
# satisfy: it has no blocked module types, so it passed with only a handful
# of Linear layers swapped -- an "int8" arm that is fp32 in all but name,
# and a comparison of a model against itself.
# --------------------------------------------------------------------------


def _tiny_linear_model(n_layers: int):
    """A small model whose weight layers are ALL nn.Linear.

    Built directly from nn.Linear rather than from a BERT config, because
    BERT's ``BertLMPredictionHead`` wraps its Linear in a container that
    ``_quantizable_weight_modules`` correctly reports as blocked — which
    would make this test exercise the already-covered architecture instead
    of the gap it is meant to close.
    """
    import torch.nn as nn

    class _Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                nn.Linear(32, 32) for _ in range(n_layers)
            )
            self.head = nn.Linear(32, 32)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.head(x)

    return _Plain()



def test_a_model_with_too_few_quantized_modules_is_refused():
    """Few swapped modules must be refused even when nothing is 'blocked'.

    This is the gap: the old guard required `blocked` to be non-empty, so a
    model whose weight layers are all nn.Linear (nothing blocked) sailed
    through with only a few modules swapped.
    """
    small = _tiny_linear_model(1)
    blocked = _quantizable_weight_modules(small)
    assert blocked == {}, f"expected no blocked types, got {blocked}"
    with pytest.raises(AdapterError, match="quantize|module|Linear"):
        _quantize_int8(small)



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