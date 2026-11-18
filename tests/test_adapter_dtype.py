"""The precision label must be a measurement, not an assumption.

transformers 5.x loads a checkpoint in the dtype its own config declares:
``Qwen/Qwen2.5-0.5B-Instruct`` and ``HuggingFaceTB/SmolLM2-135M-Instruct``
arrive as bfloat16. The adapter's "fp32" arm used to be a bare ``return model``,
so the baseline of every comparison was really bf16 while the report said
fp32 — and dynamic INT8 then died mid-run with ``RuntimeError: expected scalar
type Float but found BFloat16``. Both halves are asserted here on a local
bfloat16 checkpoint, so the test needs no network.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from quant_regress import adapter as adapter_mod
from quant_regress.adapter import build_model_factory
from quant_regress.harness import _measured_dtype


@pytest.fixture()
def bf16_checkpoint(tmp_path, monkeypatch):
    """A tiny local checkpoint whose config declares bfloat16."""
    cfg = transformers.AutoConfig.for_model(
        "llama",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        torch_dtype="bfloat16",
    )
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    model = model.to(torch.bfloat16)
    model.save_pretrained(str(tmp_path))
    # The tokenizer is irrelevant to dtype and would need a network fetch.
    monkeypatch.setattr(
        adapter_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: object()
    )
    return str(tmp_path)


def test_fp32_arm_is_really_float32_on_a_bfloat16_checkpoint(bf16_checkpoint):
    factory = build_model_factory(model_name=bf16_checkpoint)
    fp32 = factory("fp32")
    dtypes = {p.dtype for p in fp32.parameters()}
    assert dtypes == {torch.float32}, (
        f"the baseline claims fp32 but measured {dtypes} — the checkpoint loads "
        "as bfloat16, so an untouched model is not an fp32 baseline"
    )


def test_int8_arm_builds_on_a_bfloat16_checkpoint(bf16_checkpoint):
    factory = build_model_factory(model_name=bf16_checkpoint)
    int8 = factory("int8")  # used to raise RuntimeError: expected Float/BFloat16
    assert _measured_dtype(int8) == "torch.qint8", _measured_dtype(int8)


def test_measured_dtype_reads_the_quantized_weights_not_the_embeddings():
    plain = torch.nn.Sequential(torch.nn.Linear(4, 4))
    assert _measured_dtype(plain) == "torch.float32"
    assert _measured_dtype(plain.half()) == "torch.float16"
    quantized = torch.ao.quantization.quantize_dynamic(
        plain, {torch.nn.Linear}, dtype=torch.qint8
    )
    assert _measured_dtype(quantized) == "torch.qint8"
    assert _measured_dtype(None) == ""
