"""Core harness: run a task eval at several precisions and compare accuracy.

The point of this module is to make the marginal-decision damage visible.
Token-level proxies (perplexity, KL) routinely miss it; task accuracy does
not, provided the whole set is actually run — which the harness enforces by
scoring every case at every precision and refusing to report a partial run.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from .evalset import EvalSet, normalise

SUPPORTED_PRECISIONS = ("fp32", "int8")


def _check_threshold(max_drop_points: float) -> float:
    """Reject thresholds that would make the gate unsatisfiable or inverting.

    The verdict is ``worst_drop > max_drop_points``. Every value rejected here
    was measured to break a real 100-point regression:

    * ``nan`` — every comparison with nan is False, so PASS always wins.
    * ``+inf`` — no finite drop can exceed infinity, so PASS always wins.
    * negative — the test inverts into "the quantized model must be at least as
      bad by N points", so an INT8 collapse PASSes and a healthy model FAILs.

    Zero is allowed deliberately: it means "any regression fails", a coherent
    policy for a gate whose whole purpose is to catch accuracy loss.
    """
    try:
        value = float(max_drop_points)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"max_drop_points must be a number, got {max_drop_points!r}"
        ) from exc
    if math.isnan(value):
        raise ValueError(
            "max_drop_points is NaN: every comparison against NaN is False, so "
            "the gate would report PASS for ANY regression. Pass a finite "
            "number, e.g. 2.0."
        )
    if value == math.inf:
        raise ValueError(
            "max_drop_points is +inf: no accuracy drop can exceed it, so the "
            "gate would always PASS. Pass a finite number, e.g. 2.0."
        )
    if value < 0:
        raise ValueError(
            f"max_drop_points must be >= 0, got {value}: a negative threshold "
            "inverts the gate, so an INT8 accuracy collapse would PASS and a "
            "healthy quantized model would FAIL. Use 0 to fail on any "
            "regression."
        )
    return value


def _check_min_accuracy(min_accuracy: float | None) -> float | None:
    """Validate the baseline-quality floor, or ``None`` when unset.

    NaN would make every ``accuracy < min_accuracy`` comparison False, so the
    floor would never fire -- the same silent-pass class as a NaN drop
    threshold. Values are fractions in [0, 1]; a percentage like ``50`` is
    rejected rather than silently read as 5000%, because a typo there would
    disable the floor instead of tightening it.
    """
    if min_accuracy is None:
        return None
    try:
        value = float(min_accuracy)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"min_accuracy must be a number, got {min_accuracy!r}"
        ) from exc
    if math.isnan(value):
        raise ValueError(
            "min_accuracy is NaN: every comparison against NaN is False, so the "
            "floor would never fire and a useless model would PASS. Pass a "
            "fraction such as 0.5 for 50%."
        )
    if value > 1.0:
        raise ValueError(
            f"min_accuracy is {value}, which is above 1.0: it is a FRACTION, "
            "not percentage points. Pass 0.5 for 50%, not 50."
        )
    if value < 0.0:
        raise ValueError(
            f"min_accuracy must be >= 0, got {value}. Use 0 to require only that "
            "the baseline answers at least once."
        )
    return value


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass
class PrecisionResult:
    precision: str
    correct: int
    total: int
    seconds: float
    misclassified: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total else 0.0


@dataclass
class ComparisonResult:
    baseline: PrecisionResult
    candidates: list[PrecisionResult]
    max_drop_points: float
    verdict: Verdict
    #: Minimum acceptable BASELINE accuracy as a fraction (0-1). ``None`` means
    #: no floor was requested and only the delta is gated.
    min_accuracy: float | None = None
    #: Human-readable reason when the floor was breached, else ``None``.
    floor_failure: str | None = None

    @property
    def baseline_below_floor(self) -> bool:
        """True when the baseline itself is too weak to trust the comparison."""
        return (
            self.min_accuracy is not None
            and self.baseline.accuracy < self.min_accuracy
        )

    @property
    def worst_drop_points(self) -> float:
        drops = [
            (self.baseline.accuracy - c.accuracy) * 100.0 for c in self.candidates
        ]
        return max(drops) if drops else 0.0

    def to_dict(self) -> dict:
        return {
            "baseline": {
                "precision": self.baseline.precision,
                "accuracy": round(self.baseline.accuracy, 4),
                "correct": self.baseline.correct,
                "total": self.baseline.total,
                "seconds": round(self.baseline.seconds, 3),
                # Named failures make a red build triageable: which cases got
                # worse is the first question anyone asks, and without the ids
                # the only answer is "run it again locally".
                "misclassified": list(self.baseline.misclassified),
            },
            "candidates": [
                {
                    "precision": c.precision,
                    "accuracy": round(c.accuracy, 4),
                    "correct": c.correct,
                    "total": c.total,
                    "seconds": round(c.seconds, 3),
                    "drop_points": round(
                        (self.baseline.accuracy - c.accuracy) * 100.0, 3
                    ),
                    "misclassified": list(c.misclassified),
                }
                for c in self.candidates
            ],
            "worst_drop_points": round(self.worst_drop_points, 3),
            "max_drop_points": self.max_drop_points,
            "min_accuracy": self.min_accuracy,
            "baseline_below_floor": self.baseline_below_floor,
            "floor_failure": self.floor_failure,
            "verdict": self.verdict.value,
        }


class RegressError(RuntimeError):
    """A model could not be made to produce an answer.

    Subclasses ``RuntimeError`` so the CLI's existing handler catches it and
    exits 2 -- the code meaning "bad configuration / could not measure".
    Letting the underlying ``TypeError``/``AttributeError`` escape instead
    makes it exit 1, the code meaning "accuracy regressed", which is a
    broken model impersonating a real regression.
    """


class QuantHarness:
    """Measures task accuracy for a model at each requested precision.

    ``model_factory(precision) -> model`` is injected so tests can run with a
    deterministic fake and CI can run with a real model.
    """

    def __init__(
        self,
        model_factory: Callable[[str], object],
        max_new_tokens: int = 32,
        scorer: Callable[[str, str], bool] | None = None,
        system_prompt: str | None = None,
        labels: Sequence[str] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ):
        if scorer is None:
            def scorer(pred: str, expected: str) -> bool:
                return normalise(pred) == expected
        self._factory = model_factory
        self.max_new_tokens = max_new_tokens
        self._scorer = scorer
        self.system_prompt = system_prompt
        # Label set for models that expose only logits and carry no id2label.
        self.labels = list(labels) if labels else []
        # Called as progress(done, total, precision) after each case. The CLI
        # passes a stderr writer; a library caller can leave it unset.
        self._progress = progress

    # -- public -----------------------------------------------------------
    def compare(
        self,
        eval_set: EvalSet,
        baseline_precision: str = "fp32",
        candidate_precisions: Sequence[str] = ("int8",),
        max_drop_points: float = 2.0,
        min_accuracy: float | None = None,
    ) -> ComparisonResult:
        """Measure every precision and decide whether the build should pass.

        ``max_drop_points`` gates the CHANGE from baseline to candidate.
        ``min_accuracy`` gates the BASELINE's own quality: measured, a model
        that is wrong at every precision yields a 0.00-point drop and a PASS,
        so without a floor a useless model produces a green build. It is a
        fraction in [0, 1] (0.5 means 50%), not percentage points.
        """
        for p in (baseline_precision, *candidate_precisions):
            if p not in SUPPORTED_PRECISIONS:
                raise ValueError(
                    f"unsupported precision {p!r}; supported: {SUPPORTED_PRECISIONS}"
                )
        if baseline_precision in candidate_precisions:
            raise ValueError(
                f"baseline precision {baseline_precision!r} must not appear in "
                f"candidate precisions {list(candidate_precisions)}: the harness "
                f"would compare a model against itself, producing a 0.0-point "
                f"drop and PASS without measuring any quantization regression."
            )
        if not len(eval_set):
            raise ValueError("eval set is empty")

        # A gate must not be able to PASS a regression. Every one of these
        # cases measured a real 100-point drop and returned PASS (or inverted
        # to FAIL), which is worse than no gate at all: the failure is silent.
        _check_threshold(max_drop_points)
        _check_min_accuracy(min_accuracy)

        # With no candidates there is nothing to compare, so `worst_drop_points`
        # falls back to its `default=0.0` and the run reports PASS having
        # measured only the baseline. That is a green build with no comparison
        # in it, so refuse instead.
        if not candidate_precisions:
            raise ValueError(
                "no candidate precisions given: nothing would be compared against "
                "the baseline, so the run would report PASS without measuring a "
                "quantized model. Pass e.g. candidate_precisions=['int8']."
            )

        baseline = self._measure(eval_set, baseline_precision, self._progress)
        candidates = [
            self._measure(eval_set, p, self._progress) for p in candidate_precisions
        ]

        # Every precision must have scored the whole set, or the comparison
        # is meaningless. Refuse rather than report a partial number.
        for r in (baseline, *candidates):
            if r.total != len(eval_set):
                raise RuntimeError(
                    f"incomplete run at {r.precision}: scored {r.total} of "
                    f"{len(eval_set)} cases"
                )

        worst = max(
            ((baseline.accuracy - c.accuracy) * 100.0 for c in candidates), default=0.0
        )

        # The floor is a QUALITY failure, not a configuration error, so it is a
        # FAIL (exit 1) rather than exit 2: the tool measured correctly and the
        # answer is "no". Exit 2 stays reserved for runs that could not measure.
        floor_failure = None
        if min_accuracy is not None and baseline.accuracy < min_accuracy:
            floor_failure = (
                f"baseline {baseline.precision} accuracy "
                f"{baseline.accuracy * 100:.1f}% is below the --min-accuracy "
                f"floor of {min_accuracy * 100:.1f}% "
                f"({baseline.correct}/{baseline.total} correct). Quantization "
                f"did not cause this, but a delta measured from a model this "
                f"weak cannot be trusted, and an eval set the model simply "
                f"cannot answer would otherwise PASS with a 0.00-point drop."
            )

        return ComparisonResult(
            baseline=baseline,
            candidates=candidates,
            max_drop_points=max_drop_points,
            verdict=Verdict.FAIL
            if worst > max_drop_points or floor_failure
            else Verdict.PASS,
            min_accuracy=min_accuracy,
            floor_failure=floor_failure,
        )

    # -- internals --------------------------------------------------------
    def _measure(
        self,
        eval_set: EvalSet,
        precision: str,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> PrecisionResult:
        """Score the whole set at one precision, then release the model.

        The model is released in a `finally`, so a scorer that raises mid-run
        cannot leak it. Two precisions means two models: a 7B checkpoint is
        ~14 GB of weights, and holding both at once OOMs the runner instead of
        producing a verdict — a failure that looks like a flake rather than a
        bug.
        """
        model = self._factory(precision)
        tok = None
        try:
            tok = getattr(model, "tokenizer", None)
            correct = 0
            mis: list[str] = []
            scored = 0
            t0 = time.perf_counter()
            for i, case in enumerate(eval_set, 1):
                pred = self._predict(model, tok, case.prompt)
                ok = self._scorer(pred, case.expected)
                correct += int(ok)
                scored += 1
                if not ok:
                    mis.append(case.id)
                if progress is not None:
                    progress(i, len(eval_set), precision)
            return PrecisionResult(
                precision=precision,
                correct=correct,
                # The count actually SCORED, not len(eval_set). The two differ
                # exactly when the set yields fewer cases than it declares,
                # and that difference is what the incomplete-run check in
                # compare() exists to catch -- recording len() here made that
                # check compare a number against itself, so it could never
                # fire and a partial run was reported as a verdict.
                total=scored,
                seconds=time.perf_counter() - t0,
                misclassified=mis,
            )
        finally:
            # Drop the local reference and force the collection cycle: the
            # caching allocator only returns blocks once the module graph has
            # no more references, and a dangling tensor in an exception
            # traceback is enough to keep the whole model resident.
            del model
            if tok is not None:
                del tok
            gc.collect()

    def _predict(self, model, tok, prompt: str) -> str:
        """Return the model's answer as text.

        Three supported model shapes, in priority order:

        1. A real HF model with ``generate`` + tokenizer — decode the output.
        2. A model exposing ``.answer(prompt)`` — used by fakes and by any model
           whose output is not free text (a classifier, a scorer).
        3. A bare callable returning ``.logits`` — decoded via the model's own
           ``id2label`` when it has one, else by argmax over a caller-supplied
           label set.

        Shape 3 deliberately does NOT stringify the raw logit. A logit is not an
        answer: ``str(0.95)`` can never equal ``"yes"``, so stringifying it made
        every such model score 0% and silently defeated the regression gate.

        On shape 1 only the CONTINUATION is decoded. ``generate()`` returns the
        input ids followed by the new ones, so decoding the whole tensor returns
        the prompt as the model's "answer" — measured: prompting
        ``"The capital of France is"`` decoded to
        ``"The capital of France is is us us us us us"``. With a system prompt
        configured, the leak is worse: the system prompt is returned too, so the
        prediction is the instruction rather than the answer. Under exact-match
        scoring that is not a cosmetic defect — it makes the generate path
        unsatisfiable, every case scores 0 at every precision, and the gate
        reports a 0.00-point drop and PASSes a model it never measured.

        Whether the continuation is *enough* to match is the eval-set author's
        call, not this function's: a prompt asking for a one-word answer and an
        ``expected`` of ``"Paris"`` now works, and a chatty model can still be
        matched by passing a custom ``scorer``.

        Any unexpected failure inside the model is re-raised as
        :class:`RegressError` (a ``RuntimeError``), so the CLI turns it into
        exit 2 instead of a traceback. A raw ``TypeError`` or
        ``AttributeError`` from a broken model currently escapes main() and
        exits 1 -- the code meaning "accuracy regressed".
        """
        try:
            return self._predict_one(model, tok, prompt)
        except RegressError:
            raise
        except Exception as exc:
            raise RegressError(
                f"model at {getattr(model, 'precision', '?')!r} could not "
                f"produce an answer for prompt {prompt[:60]!r}: "
                f"{type(exc).__name__}: {exc}. The gate could not measure this "
                f"model, so it will not guess a verdict. Check that the model "
                f"and its tokenizer load and match this eval set's format."
            ) from exc

    def _predict_one(self, model, tok, prompt: str) -> str:
        """The three supported model shapes, in priority order.

        Kept separate from :meth:`_predict` so the wrapper's except clause
        cannot swallow its own error type.
        """
        # 2. explicit text answer
        answer = getattr(model, "answer", None)
        if callable(answer):
            return str(answer(prompt))

        # 1. real HF generation path — continuation only
        if tok is not None and hasattr(model, "generate"):
            text = prompt if self.system_prompt is None else f"{self.system_prompt}\n{prompt}"
            inputs = tok(text, return_tensors="pt")
            inputs = {k: v.to(getattr(model, "device", "cpu")) for k, v in inputs.items()}
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                 do_sample=False)
            n_prompt = int(inputs["input_ids"].shape[-1])
            new_ids = out[0][n_prompt:]
            if new_ids.numel() == 0:
                return ""
            return tok.decode(new_ids, skip_special_tokens=True)

        # 3. logits -> label
        if tok is None:
            raise RegressError(
                "cannot evaluate a logits-only model without a tokenizer: "
                "the prompt cannot be tokenized. Attach a tokenizer to the "
                "model (model.tokenizer = ...) or use a model with .answer()."
            )
        text = prompt if self.system_prompt is None else f"{self.system_prompt}\n{prompt}"
        inputs = tok(text, return_tensors="pt")
        inputs = {k: v.to(getattr(model, "device", "cpu")) for k, v in inputs.items()}
        logits = model(**inputs).logits
        row = logits[0]
        idx = int(max(range(len(row)), key=lambda i: float(row[i])))
        id2label = getattr(model, "config", None)
        id2label = getattr(id2label, "id2label", None) if id2label else None
        if id2label:
            return str(id2label.get(idx, idx))
        if self.labels:
            return str(self.labels[idx]) if idx < len(self.labels) else str(idx)
        raise RegressError(
            "cannot map model output to text: the model exposes no .answer(), "
            "no generate()+tokenizer, and no id2label. Give the fake model an "
            "answer(prompt) method, or pass labels=[...] to QuantHarness."
        )
