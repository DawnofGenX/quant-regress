"""The adapter must build a genuinely different int8 model, or the tool lies."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from quant_regress.adapter import build_model_factory, AdapterError


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 2)

    def forward(self, **kw):
        return type("O", (), {"logits": self.linear(torch.zeros(1, 8))})()


def test_int8_model_is_actually_quantized():
    factory = build_model_factory(model_name="hf-internal-testing/tiny-random-gpt2")
    fp32 = factory("fp32")
    int8 = factory("int8")
    # In modern torch, quantize_dynamic keeps the Linear class name but packs
    # weights into a LinearPackedParams child module.
    assert not any("packed" in type(m).__name__.lower() for m in fp32.modules())
    assert any("packed" in type(m).__name__.lower() for m in int8.modules())


def test_factory_rejects_unknown_precision():
    factory = build_model_factory(model_name="hf-internal-testing/tiny-random-gpt2")
    with pytest.raises(ValueError, match="unsupported precision"):
        factory("int4")


def test_empty_model_name_is_rejected_with_actionable_message():
    with pytest.raises(AdapterError, match="model"):
        build_model_factory(model_name="")
