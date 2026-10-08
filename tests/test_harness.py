"""The harness must run a real eval at multiple precisions and report accuracy."""
from __future__ import annotations

import json
import weakref

import pytest

from quant_regress.evalset import Case, EvalSet, EvalSetError
from quant_regress.harness import QuantHarness, RegressError, Verdict


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

    The fake carries a tokenizer because the harness now tokenizes the real
    prompt on the logits path (it no longer feeds a hardcoded ``[[1]]``), so a
    tokenizer-less logits model is refused with an actionable error instead of
    being evaluated on an input the prompt never reached.
    """
    es = _eval(tmp_path, n=3)

    class _Cfg:
        id2label = {0: "no", 1: "yes"}

    class _LogitsTok:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": _FakeBatch([0] * (len(text) + 1))}

    class _LogitsOnly:
        def __init__(self, precision):
            self.precision = precision
            self.config = _Cfg()
            self.tokenizer = _LogitsTok()
            self.seen_lengths = []

        def eval(self):
            return self

        def __call__(self, **kw):
            self.seen_lengths.append(len(kw["input_ids"]))
            return _Logits(0.9)  # argmax -> index 0 -> "no"

    h = QuantHarness(model_factory=lambda p: _LogitsOnly(p))
    model = _LogitsOnly("fp32")
    # The label map is what produces the text, not the raw logit value.
    assert h._predict(model, model.tokenizer, "q?") == "no"
    # And the prompt really was tokenized, so the answer is not an artefact
    # of a hardcoded dummy input.
    assert set(model.seen_lengths) == {3}, model.seen_lengths

    res = h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"])
    # "no" != "yes", so accuracy is 0 — reached through the label map.
    assert res.baseline.accuracy == 0.0
    assert res.candidates[0].accuracy == 0.0


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


def test_logits_path_uses_prompt_not_dummy_input(tmp_path):
    """The logits path must tokenize the actual prompt, not a hardcoded dummy.

    Pre-fix: input_ids=[[1]] was passed regardless of prompt, so every case
    got the same prediction and the evaluation was meaningless.
    """
    rows = [
        {"id": "a", "prompt": "x?", "expected": "yes"},
        {"id": "b", "prompt": "a much longer prompt that tokenizes differently?", "expected": "yes"},
        {"id": "c", "prompt": "medium length prompt here?", "expected": "yes"},
    ]
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    es = EvalSet.load(p)

    class _Tok:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": _FakeBatch([0] * (len(text) + 1))}

    class _LogitsModel:
        def __init__(self, precision):
            self.precision = precision
            self.tokenizer = _Tok()
            self.seen_lengths = []

        def eval(self):
            return self

        def __call__(self, **kw):
            n = len(kw["input_ids"])
            self.seen_lengths.append(n)
            return _Logits(0.9 if n > 5 else 0.1)

    model = _LogitsModel("fp32")
    h = QuantHarness(model_factory=lambda p: model, labels=["no", "yes"])
    h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"])
    assert len(set(model.seen_lengths)) > 1, (
        f"all prompts produced the same input length {model.seen_lengths}, "
        f"so the prompt was not tokenized"
    )


def test_unsupported_precision_is_rejected(tmp_path):
    es = _eval(tmp_path, n=2)
    h = QuantHarness(model_factory=lambda p: _FakeModel(p))
    with pytest.raises(ValueError, match="unsupported precision"):
        h.compare(es, baseline_precision="fp32", candidate_precisions=["int4"])


def test_baseline_equal_to_candidate_is_rejected(tmp_path):
    """If baseline == candidate, the harness compares a model against itself,
    producing 0.0 drop and PASS without measuring any quantization regression."""
    es = _eval(tmp_path, n=4)
    h = QuantHarness(model_factory=lambda p: _FakeModel(p, correct=(p == "fp32")))
    with pytest.raises(ValueError, match="baseline.*candidate"):
        h.compare(es, baseline_precision="int8", candidate_precisions=["int8"])


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
# Phase 2 -- robustness. The gate runs real models on real CI runners, so a
# crash or an OOM is a broken build. Each guard here was MEASURED to be
# missing (the bug re-introduced, the test going red) before the fix landed.
# --------------------------------------------------------------------------


class _Tracked:
    """Answers deterministically; liveness is observable via a weakref.

    Two details make this able to detect a real leak:

    * the registry holds *weakrefs*, since a strong reference would pin the
      model forever and hide the bug; and
    * ``__init__`` builds a reference cycle (``child.parent is self``), which
      is what a real torch module graph is — a parent module holding
      submodules that hold a back-reference. Without the cycle, CPython's
      refcounting frees the model the instant ``_measure`` returns and the
      test passes even with the fix removed.

    Verified load-bearing both ways: with the cycle present and the
    ``del``/``gc.collect()`` removed, both models stay resident.
    """

    alive: list = []

    def __init__(self, precision: str):
        self.precision = precision
        self.child = _Cycle()
        self.child.parent = self  # a module graph holds its children and back
        _Tracked.alive.append(weakref.ref(self))

    def eval(self):
        return self

    def answer(self, prompt: str) -> str:
        return "yes"

    @classmethod
    def live_count(cls) -> int:
        return sum(1 for r in cls.alive if r() is not None)


class _Cycle:
    """Stand-in for a submodule pointing back at its parent."""

    parent: object = None


def test_models_are_freed_between_precisions(tmp_path):
    """Each precision's model must be released before the next is built.

    Two precisions means two models alive at the same time. For a 7B
    checkpoint that is ~28 GB of resident weights on a runner with far less,
    so the gate OOMs instead of reporting a verdict. The fix releases the
    previous model (and runs gc) as soon as its measurement finishes.

    Verified load-bearing: with the `del`/`gc.collect()` removed, both models
    stay alive simultaneously and this fails.
    """
    _Tracked.alive.clear()
    es = _eval(tmp_path, n=3)
    h = QuantHarness(model_factory=lambda p: _Tracked(p))
    h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"])

    assert len(_Tracked.alive) == 2, _Tracked.alive
    # The baseline is done by the time the candidate is measured, so only the
    # model currently in use may still be alive.
    assert _Tracked.live_count() <= 1, (
        f"{_Tracked.live_count()} models still resident after their "
        "measurements finished -- two precisions means two models in memory "
        "at once"
    )


def test_models_are_freed_when_a_measurement_raises(tmp_path):
    """A failure mid-run must not leak the model that was being measured.

    The scorer raises, so `_measure` exits early. Without a `finally`, that
    model is never released and repeated failures leak memory until the
    runner dies -- a failure mode that looks like a flake, not a bug.
    """
    _Tracked.alive.clear()
    es = _eval(tmp_path, n=3)

    def boom(pred, expected):
        raise RuntimeError("scorer exploded")

    h = QuantHarness(model_factory=lambda p: _Tracked(p), scorer=boom)
    with pytest.raises(RuntimeError):
        h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"])
    assert len(_Tracked.alive) >= 1
    assert _Tracked.live_count() == 0, (
        "the model being measured was not released when the run raised"
    )





class _AlwaysWrong:
    """0% at every precision — e.g. an eval set the model cannot answer."""

    def __init__(self, precision: str):
        self.precision = precision

    def eval(self):
        return self

    def answer(self, prompt: str) -> str:
        return "no"


def test_a_useless_model_passes_without_a_floor(tmp_path):
    """Documents the gap the floor exists to close. Baseline 0% -> PASS."""
    es = _eval(tmp_path, n=10)
    h = QuantHarness(model_factory=lambda p: _AlwaysWrong(p))
    res = h.compare(es, candidate_precisions=["int8"], max_drop_points=2.0)
    assert res.baseline.accuracy == 0.0
    assert res.worst_drop_points == 0.0
    assert res.verdict is Verdict.PASS, "this is the behaviour the floor overrides"


def test_min_accuracy_fails_a_useless_model(tmp_path):
    es = _eval(tmp_path, n=10)
    h = QuantHarness(model_factory=lambda p: _AlwaysWrong(p))
    res = h.compare(es, candidate_precisions=["int8"], max_drop_points=2.0,
                    min_accuracy=0.5)
    assert res.verdict is Verdict.FAIL
    assert res.baseline_below_floor is True
    assert "min-accuracy" in (res.floor_failure or "")


def test_min_accuracy_is_a_quality_failure_not_a_config_error(tmp_path):
    """FAIL (exit 1), not exit 2: the tool measured correctly; the answer is no."""
    es = _eval(tmp_path, n=10)
    h = QuantHarness(model_factory=lambda p: _AlwaysWrong(p))
    res = h.compare(es, candidate_precisions=["int8"], min_accuracy=0.9)
    # A harness-level FAIL. The CLI maps this to exit 1; exit 2 is reserved for
    # runs that could not measure at all.
    assert res.verdict is Verdict.FAIL
    assert res.floor_failure is not None


def test_min_accuracy_satisfied_still_gates_the_delta(tmp_path):
    """The floor must not mask a real regression."""
    es = _eval(tmp_path, n=6)
    h = QuantHarness(model_factory=lambda p: _FakeModel(p, correct=(p == "fp32")))
    res = h.compare(es, candidate_precisions=["int8"], min_accuracy=0.5)
    assert res.baseline_below_floor is False
    assert res.floor_failure is None
    assert res.verdict is Verdict.FAIL, "delta gate must still fire"


def test_min_accuracy_not_reached_exactly_is_allowed(tmp_path):
    """A baseline exactly AT the floor passes; the comparison is strict `<`."""
    rows = [{"id": str(i), "prompt": f"q{i}?", "expected": "yes"} for i in range(4)]
    import json as _json
    p = tmp_path / "e.jsonl"
    p.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")
    es = EvalSet.load(p)

    class _ThreeOfFour:
        def __init__(self, p_): self.p = p_
        def eval(self): return self
        def answer(self, prompt): return "yes"

    h = QuantHarness(model_factory=lambda p: _ThreeOfFour(p))
    res = h.compare(es, candidate_precisions=["int8"], min_accuracy=1.0)
    assert res.baseline.accuracy == 1.0
    assert res.baseline_below_floor is False
    assert res.verdict is Verdict.PASS


def test_nan_min_accuracy_is_rejected_because_it_would_never_fire(tmp_path):
    """Same silent-pass class as a NaN drop threshold."""
    es = _eval(tmp_path, n=4)
    h = QuantHarness(model_factory=lambda p: _AlwaysWrong(p))
    with pytest.raises(ValueError, match="NaN"):
        h.compare(es, candidate_precisions=["int8"], min_accuracy=float("nan"))


def test_percentage_style_min_accuracy_is_rejected(tmp_path):
    """A user typing 50 must be told it is a fraction, not silently accepted."""
    es = _eval(tmp_path, n=4)
    h = QuantHarness(model_factory=lambda p: _AlwaysWrong(p))
    with pytest.raises(ValueError, match="FRACTION"):
        h.compare(es, candidate_precisions=["int8"], min_accuracy=50)


def test_negative_min_accuracy_is_rejected(tmp_path):
    es = _eval(tmp_path, n=4)
    h = QuantHarness(model_factory=lambda p: _AlwaysWrong(p))
    with pytest.raises(ValueError, match=">= 0"):
        h.compare(es, candidate_precisions=["int8"], min_accuracy=-0.1)


def test_zero_min_accuracy_is_allowed(tmp_path):
    """0 means 'the baseline must answer at least once' — coherent, so allowed."""
    es = _eval(tmp_path, n=4)
    h = QuantHarness(model_factory=lambda p: _FakeModel(p, correct=(p == "fp32")))
    res = h.compare(es, candidate_precisions=["int8"], min_accuracy=0.0)
    assert res.baseline_below_floor is False


def test_report_records_the_floor_and_its_outcome(tmp_path):
    es = _eval(tmp_path, n=10)
    h = QuantHarness(model_factory=lambda p: _AlwaysWrong(p))
    res = h.compare(es, candidate_precisions=["int8"], min_accuracy=0.5)
    d = res.to_dict()
    assert d["min_accuracy"] == 0.5
    assert d["baseline_below_floor"] is True
    assert d["verdict"] == "fail"
    assert d["floor_failure"]


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

# --------------------------------------------------------------------------
# Phase 4 -- coverage for paths that were previously reachable but unguarded.
# Each of these could regress silently: the guard exists, nothing proved it
# still fires.
# --------------------------------------------------------------------------


class _Partial:
    """Scores only some of the cases, as a model that gives up mid-run would."""

    def __init__(self, precision: str, stop_after: int):
        self.precision = precision
        self.stop_after = stop_after

    def eval(self):
        return self

    def answer(self, prompt: str) -> str:
        return "yes"


class _ShortEval:
    """An eval set that shrank after some precisions already ran.

    The realistic shape of this defect: a set that yields fewer cases to one
    arm than to another -- for instance a lazy iterator that is exhausted, or
    a set mutated concurrently. The harness must refuse a partial number
    rather than report accuracy computed over a subset as though it were the
    whole set.
    """

    def __init__(self, first: int, then: int):
        self._n = first
        self._then = then
        self._passes = 0

    def __len__(self):
        return self._n

    def __iter__(self):
        self._passes += 1
        n = self._n if self._passes == 1 else self._then
        for i in range(n):
            yield Case(id=f"c{i}", prompt=f"p{i}", expected="yes")


def test_incomplete_run_is_refused_rather_than_reporting_a_partial_number():
    """A partial run must not be reported as a verdict.

    Without the check, a model that answers only part of the set produces a
    plausible-looking accuracy over the cases it did answer -- a number that
    looks like a measurement but is not one, and it would be compared against
    a full baseline as if the two were comparable.
    """
    ev = _ShortEval(first=6, then=3)
    h = QuantHarness(model_factory=lambda p: _FakeModel(p))
    with pytest.raises(RuntimeError, match="incomplete"):
        h.compare(ev, baseline_precision="fp32", candidate_precisions=["int8"])


def test_empty_eval_set_is_refused(tmp_path):
    """A gate with no cases cannot measure anything, so it must refuse.

    Refusing matters because the fallback -- reporting an empty run as a
    verdict -- would let any configuration green through CI.
    """
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(EvalSetError, match="empty"):
        EvalSet.load(p)

    # The harness refuses a zero-length set independently of the loader, so a
    # programmatic caller cannot bypass the check by constructing one directly.
    with pytest.raises(ValueError, match="empty"):
        QuantHarness(model_factory=lambda p: _FakeModel(p)).compare(
            EvalSet([]), candidate_precisions=["int8"]
        )


def test_a_custom_scorer_is_actually_used(tmp_path):
    """QuantHarness(scorer=...) must reach the scorer, not just accept it.

    A scorer that is stored and never called would make every custom scoring
    setup silently fall back to exact matching -- which then fails to match
    anything, so the run looks like a model problem rather than a wiring bug.
    """
    es = _eval(tmp_path, n=3)
    calls: list[tuple[str, str]] = []

    def custom(pred, expected):
        calls.append((pred, expected))
        return True  # everything is "correct"

    h = QuantHarness(model_factory=lambda p: _FakeModel(p), scorer=custom)
    res = h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"])

    assert calls, "the custom scorer was never called -- it was accepted and dropped"
    assert res.baseline.accuracy == 1.0, "the scorer's verdict did not reach the result"


def test_default_scorer_matches_after_normalisation(tmp_path):
    """The default is normalised equality, so case and spacing must not matter."""
    rows = [{"id": "a", "prompt": "x?", "expected": "  YES  "}]
    p = tmp_path / "e.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    es = EvalSet.load(p)

    class _Chatty:
        def __init__(self, precision):
            self.precision = precision

        def eval(self):
            return self

        def answer(self, prompt):
            return "Yes"

    h = QuantHarness(model_factory=lambda p: _Chatty(p))
    res = h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"])
    assert res.baseline.accuracy == 1.0, "exact scoring should normalise before comparing"
