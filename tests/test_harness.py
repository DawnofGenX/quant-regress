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


# --------------------------------------------------------------------------
# A gate that cannot fail is worse than no gate. Each of these was MEASURED to
# let a real 100-point INT8 regression through (or to invert it), silently.
# --------------------------------------------------------------------------


def test_nan_threshold_is_rejected_because_it_always_passes(tmp_path):
    """`worst_drop > nan` is False for every drop, so PASS always wins."""
    es = _eval(tmp_path, n=4)

    def factory(precision):
        return _FakeModel(precision, correct=(precision == "fp32"))

    h = QuantHarness(model_factory=factory)
    with pytest.raises(ValueError, match="NaN"):
        h.compare(es, candidate_precisions=["int8"], max_drop_points=float("nan"))


def test_infinite_threshold_is_rejected_because_it_always_passes(tmp_path):
    """No finite drop can exceed +inf, so the gate would never fail."""
    es = _eval(tmp_path, n=4)
    h = QuantHarness(model_factory=lambda p: _FakeModel(p, correct=(p == "fp32")))
    with pytest.raises(ValueError, match="inf"):
        h.compare(es, candidate_precisions=["int8"], max_drop_points=float("inf"))


def test_negative_threshold_is_rejected_because_it_inverts_the_gate(tmp_path):
    """A negative threshold makes the test 'the quantized model must be worse'.

    That PASSes a 100-point collapse and FAILs a healthy quantized model -- the
    opposite of a gate. Verified: before the fix, -1.0 returned FAIL for the
    collapse, i.e. it would have failed the build for the wrong reason.
    """
    es = _eval(tmp_path, n=4)
    h = QuantHarness(model_factory=lambda p: _FakeModel(p, correct=(p == "fp32")))
    with pytest.raises(ValueError, match=">= 0"):
        h.compare(es, candidate_precisions=["int8"], max_drop_points=-1.0)


def test_zero_threshold_is_allowed_and_fails_on_any_regression(tmp_path):
    """0 is coherent policy -- "any regression fails" -- so it must be accepted."""
    es = _eval(tmp_path, n=4)
    h = QuantHarness(model_factory=lambda p: _FakeModel(p, correct=(p == "fp32")))
    res = h.compare(es, candidate_precisions=["int8"], max_drop_points=0.0)
    assert res.verdict is Verdict.FAIL


def test_non_numeric_threshold_is_rejected(tmp_path):
    es = _eval(tmp_path, n=2)
    h = QuantHarness(model_factory=lambda p: _FakeModel(p))
    with pytest.raises(ValueError):
        h.compare(es, candidate_precisions=["int8"], max_drop_points="two")


def test_empty_candidate_list_is_rejected_rather_than_reporting_pass(tmp_path):
    """With no candidates, worst_drop falls back to default=0.0 and PASSes.

    That is a green build in which no quantized model was ever measured.
    """
    es = _eval(tmp_path, n=4)
    h = QuantHarness(model_factory=lambda p: _FakeModel(p))
    with pytest.raises(ValueError, match="no candidate precisions"):
        h.compare(es, candidate_precisions=[])


def test_empty_candidate_list_is_rejected_before_any_model_is_built(tmp_path):
    """The refusal must be cheap -- no baseline run, then a thrown-away result."""
    built: list[str] = []
    es = _eval(tmp_path, n=4)
    h = QuantHarness(model_factory=lambda p: built.append(p) or _FakeModel(p))
    with pytest.raises(ValueError):
        h.compare(es, candidate_precisions=[])
    assert built == [], f"models were built before validating arguments: {built}"


# --------------------------------------------------------------------------
# The real HF generate path.
#
# Every other test here uses a model with .answer(), so the generate branch of
# _predict was untested — which is how it shipped decoding the prompt as the
# model's answer. The fakes below reproduce generate()'s contract (it returns
# prompt ids FOLLOWED BY new ids) without needing a model download.
# --------------------------------------------------------------------------


class _FakeSeq:
    """A 2-D [batch, seq] tensor: out[0] yields a 1-D row, as torch does."""

    def __init__(self, row_ids):
        self._row = _FakeBatch(row_ids)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _FakeSeq(self._row._ids[key])
        return self._row


class _FakeBatch:
    """A 1-D tensor: supports len(), slicing, numel() and .to()."""

    def __init__(self, ids):
        self._ids = list(ids)

    def to(self, *_a, **_kw):
        return self

    @property
    def shape(self):
        return (1, len(self._ids))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _FakeBatch(self._ids[key])
        return self._ids[key]

    def __len__(self):
        return len(self._ids)

    def numel(self):
        return len(self._ids)


class _FakeGenerateModel:
    """Mimics an HF causal LM: generate() returns prompt ids + continuation ids."""

    def __init__(self, precision: str = "fp32", continuation: str = " Paris"):
        self.precision = precision
        self.continuation = continuation
        self.tokenizer = None

    def eval(self):
        return self

    def generate(self, **kwargs):
        n_prompt = len(kwargs["input_ids"])
        # 2 prompt tokens + 1 continuation token, as a real model would.
        return _FakeSeq(list(range(n_prompt)) + [99])


class _FakeTokenizer:
    """Real decode semantics: ids are rendered as words, so a leaked prompt is visible.

    An earlier version ignored its input and always returned " Paris", which made
    the prompt-leak assertions pass on the BUGGY code — a test that cannot fail.
    Now token id 0 renders as "PROMPT" and 99 as "Paris", so returning the
    prompt ids is detectable.
    """

    _VOCAB = {0: "PROMPT", 1: "PROMPT", 2: "Paris", 99: "Paris"}

    def __init__(self):
        self.calls = []
        self.decoded = None

    def __call__(self, text, return_tensors=None):
        self.calls.append(text)
        return {"input_ids": _FakeBatch([0, 0])}

    def decode(self, ids, skip_special_tokens=False):
        self.decoded = list(ids)
        return " ".join(self._VOCAB.get(int(i), str(i)) for i in ids)


def _gen_harness(**kw):
    tok = _FakeTokenizer()
    model = _FakeGenerateModel()
    return QuantHarness(model_factory=lambda p: model, **kw), model, tok


def test_generate_path_decodes_only_the_continuation_not_the_prompt():
    """The prompt must NOT come back as the model's answer.

    generate() returns prompt ids followed by new ids. Decoding the whole tensor
    therefore returns the question text, so exact-match scoring can never succeed
    and the gate reports a 0.00-point drop for a model it never measured.
    """
    h, model, tok = _gen_harness()
    out = h._predict(model, tok, "The capital of France is")
    assert out == "Paris", f"prediction should be the continuation only, got {out!r}"
    assert "PROMPT" not in out, "prompt tokens leaked into the prediction"
    # Only the continuation token (99) may be decoded.
    assert tok.decoded == [99], f"decoded {tok.decoded}, expected only [99]"


def test_generate_path_with_system_prompt_does_not_leak_the_instruction():
    """A system prompt must not become part of the answer either.

    Measured pre-fix: with a system prompt configured the decode returned
    "You are a precise assistant.\\nThe capital of France is..." -- the
    instruction, not the answer.
    """
    h, model, tok = _gen_harness(system_prompt="You are a precise assistant.")
    out = h._predict(model, tok, "The capital of France is")
    assert out == "Paris", f"prediction should be the continuation only, got {out!r}"
    assert "PROMPT" not in out, "system prompt tokens leaked into the prediction"
    assert tok.calls[-1].startswith("You are a precise assistant.")


def test_generate_path_handles_a_model_that_emits_nothing():
    """Zero new tokens must yield '' rather than an IndexError."""
    class _Empty(_FakeGenerateModel):
        def generate(self, **kwargs):
            return _FakeSeq(list(range(len(kwargs["input_ids"]))))

    h = QuantHarness(model_factory=lambda p: _Empty(p))
    assert h._predict(_Empty(), _FakeTokenizer(), "q?") == ""


def test_generate_path_is_reachable_end_to_end_and_can_score(tmp_path):
    """A real generate-path run must be able to score above zero.

    Guards the whole branch: with .answer() absent, the harness must still route
    to generate() and score a match. Pre-fix this scored 0/3 at both precisions
    and reported PASS.
    """
    rows = [{"id": str(i), "prompt": f"The capital of France {i} is", "expected": "Paris"}
            for i in range(3)]
    p = tmp_path / "gen.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    es = EvalSet.load(p)

    tok = _FakeTokenizer()
    model = _FakeGenerateModel()
    h = QuantHarness(model_factory=lambda precision: model)
    # Inject the tokenizer the way the adapter attaches one to a real model.
    model.tokenizer = tok

    res = h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"])
    assert res.baseline.accuracy == 1.0, (
        "the generate path scored 0 despite producing the expected answer"
    )
    assert res.verdict is Verdict.PASS