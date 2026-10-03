"""The harness must run a real eval at multiple precisions and report accuracy."""
from __future__ import annotations

import json

import pytest

from quant_regress.evalset import EvalSet
from quant_regress.harness import QuantHarness, Verdict


class _Logits:
    def __init__(self, score):
        self.logits = [[score, 1.0 - score]]


class _FakeModel:
    """Deterministic stand-in that answers correctly, or wrongly once quantized.

    Mirrors the real failure mode this tool exists to catch — a quantized model
    that still runs and still produces output, just worse output. Precision is
    captured at construction time, because that is what the factory knows and
    what the comparison is actually about.
    """

    def __init__(self, precision: str, correct: bool = True):
        self.precision = precision
        self.correct = correct
        self.calls = 0

    def eval(self):
        return self

    def answer(self, prompt: str) -> str:
        self.calls += 1
        return "yes" if self.correct else "no"


def _eval(tmp_path, n=6, expected="yes"):
    rows = [{"id": str(i), "prompt": f"q{i}?", "expected": expected} for i in range(n)]
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return EvalSet.load(p)


def test_fp32_and_int8_are_both_measured(tmp_path):
    es = _eval(tmp_path)

    h = QuantHarness(model_factory=lambda precision: _FakeModel(precision))
    res = h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"])

    assert res.baseline.precision == "fp32"
    assert res.baseline.accuracy == 1.0
    assert len(res.candidates) == 1
    assert res.candidates[0].precision == "int8"


def test_verdict_fails_when_accuracy_drops_past_threshold(tmp_path):
    es = _eval(tmp_path, n=4)

    def factory(precision):
        return _FakeModel(precision, correct=(precision == "fp32"))

    h = QuantHarness(model_factory=factory)
    res = h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"],
                    max_drop_points=5.0)

    assert res.verdict is Verdict.FAIL
    assert res.worst_drop_points > 5.0


def test_verdict_passes_when_drop_is_within_tolerance(tmp_path):
    es = _eval(tmp_path, n=6)

    h = QuantHarness(model_factory=lambda precision: _FakeModel(precision, correct=True))
    res = h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"],
                    max_drop_points=5.0)
    assert res.verdict is Verdict.PASS


def test_every_measured_precision_runs_the_whole_set(tmp_path):
    """Guards against a partial run silently reporting inflated accuracy."""
    es = _eval(tmp_path, n=5)

    seen = []

    def factory(precision):
        seen.append(precision)
        return _FakeModel(precision)

    h = QuantHarness(model_factory=factory, max_new_tokens=4)
    res = h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"])
    assert seen == ["fp32", "int8"]
    assert res.baseline.total == 5 and res.candidates[0].total == 5


def test_logits_only_model_uses_labels_not_a_stringified_logit(tmp_path):
    """Regression: stringifying a logit can never equal a textual expectation.

    That bug made every logits-only model score 0% for the wrong reason, which
    silently defeated the regression gate. Reading through id2label makes "no"
    come out as "no" rather than as "0.1".
    """
    es = _eval(tmp_path, n=3)

    class _Cfg:
        id2label = {0: "no", 1: "yes"}

    class _LogitsOnly:
        def __init__(self, precision):
            self.precision = precision
            self.config = _Cfg()

        def eval(self):
            return self

        def __call__(self, **kw):
            return _Logits(0.9)  # argmax -> index 0 -> "no"

    h = QuantHarness(model_factory=lambda p: _LogitsOnly(p))
    model = _LogitsOnly("fp32")
    # The label map is what produces the text, not the raw logit value.
    assert h._predict(model, None, "q?") == "no"

    res = h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"])
    # "no" != "yes", so accuracy is 0 — reached through the label map.
    assert res.baseline.accuracy == 0.0


def test_logits_only_model_without_labels_raises_actionable_error(tmp_path):
    es = _eval(tmp_path, n=2)

    class _Bare:
        def eval(self):
            return self

        def __call__(self, **kw):
            return _Logits(0.9)

    h = QuantHarness(model_factory=lambda p: _Bare())
    with pytest.raises(RuntimeError, match="answer\\(\\)"):
        h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"])


def test_unsupported_precision_is_rejected(tmp_path):
    es = _eval(tmp_path, n=2)
    h = QuantHarness(model_factory=lambda p: _FakeModel(p))
    with pytest.raises(ValueError, match="unsupported precision"):
        h.compare(es, baseline_precision="fp32", candidate_precisions=["int4"])