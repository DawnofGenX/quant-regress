"""Core harness: run a task eval at several precisions and compare accuracy.

The point of this module is to make the marginal-decision damage visible.
Token-level proxies (perplexity, KL) routinely miss it; task accuracy does
not, provided the whole set is actually run — which the harness enforces by
scoring every case at every precision and refusing to report a partial run.
"""

from __future__ import annotations

import gc
import hashlib
import json
from collections import defaultdict
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Sequence, cast

from .evalset import EvalSet, Case, normalise

from .stats import paired_bootstrap_drop_ci

from .adapter import SUPPORTED as SUPPORTED_PRECISIONS


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
    per_category: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: The dtype actually measured, e.g. ``"torch.float32"``. Recorded because
    #: a checkpoint can load in bfloat16 and a mislabelled baseline is exactly
    #: the kind of quiet lie this gate exists to catch. Empty when every case
    #: came from the response cache and no model was loaded.
    model_dtype: str = ""

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
    points_per_case: float = 0.0
    tolerance_reachable: bool = True
    granularity_warning: str | None = None
    #: Paired bootstrap CI on the accuracy drop, keyed by candidate precision.
    drop_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: When True, a drop that is not statistically significant does not FAIL.
    require_significance: bool = False
    #: Human-readable note about significance (PASS explanation or caveat).
    significance_note: str | None = None

    def regressed_ids(self, c: PrecisionResult) -> list[str]:
        base_wrong = set(self.baseline.misclassified)
        return [i for i in c.misclassified if i not in base_wrong]

    def recovered_ids(self, c: PrecisionResult) -> list[str]:
        cand_wrong = set(c.misclassified)
        return [i for i in self.baseline.misclassified if i not in cand_wrong]

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

    def drop_ci_for(self, c: PrecisionResult) -> tuple[float, float] | None:
        return self.drop_ci.get(c.precision)

    def significant_drop(self, c: PrecisionResult) -> bool:
        iv = self.drop_ci_for(c)
        if iv is None:
            return False
        return iv[0] > 0

    @property
    def statistics(self) -> dict:
        return {
            "method": "paired percentile bootstrap",
            "resamples": 10000,
            "alpha": 0.05,
            "seed": 0,
        }

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
                # Same audit fields as a candidate: the baseline is an arm too,
                # and a table that states a precision for it without the dtype
                # that was actually measured is how "fp32" came to mean bf16.
                "dtype": self.baseline.model_dtype,
                "categories": {
                    (k or "uncategorised"): {"correct": cc, "total": tt}
                    for k, (cc, tt) in self.baseline.per_category.items()
                },
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
                    "dtype": c.model_dtype,
                    "categories": {
                        (k or "uncategorised"): {"correct": cc, "total": tt}
                        for k, (cc, tt) in c.per_category.items()
                    },
                    "regressed": self.regressed_ids(c),
                    "recovered": self.recovered_ids(c),
                    "drop_ci_95": [
                        round(self.drop_ci_for(c)[0], 3) if self.drop_ci_for(c) is not None else 0.0,
                        round(self.drop_ci_for(c)[1], 3) if self.drop_ci_for(c) is not None else 0.0,
                    ],
                    "drop_significant": self.significant_drop(c),
                }
                for c in self.candidates
            ],
            "worst_drop_points": round(self.worst_drop_points, 3),
            "max_drop_points": self.max_drop_points,
            "min_accuracy": self.min_accuracy,
            "baseline_below_floor": self.baseline_below_floor,
            "floor_failure": self.floor_failure,
            "points_per_case": self.points_per_case,
            "tolerance_reachable": self.tolerance_reachable,
            "granularity_warning": self.granularity_warning,
            "statistics": self.statistics,
            "require_significance": self.require_significance,
            "significance_note": self.significance_note,
            "verdict": self.verdict.value,
        }


def _has_chat_template(tok) -> bool:
    """True when the tokenizer actually carries a chat template.

    ``apply_chat_template`` is defined on EVERY transformers tokenizer, so
    its existence says nothing. What distinguishes a chat model is
    ``chat_template`` being SET to a non-None string: calling the method
    without one raises ``ValueError``, which in this tool would become
    exit 2 for a model that loads fine (measured on
    ``hf-internal-testing/tiny-random-bert``).

    A test double without the attribute at all is a base tokenizer, which
    is the correct reading for a fake.
    """
    if tok is None:
        return False
    template = getattr(tok, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        return False
    return callable(getattr(tok, "apply_chat_template", None))


class RegressError(RuntimeError):
    """A model could not be made to produce an answer.

    Subclasses ``RuntimeError`` so the CLI's existing handler catches it and
    exits 2 -- the code meaning "bad configuration / could not measure".
    Letting the underlying ``TypeError``/``AttributeError`` escape instead
    makes it exit 1, the code meaning "accuracy regressed", which is a
    broken model impersonating a real regression.
    """


class _ResponseCache:
    """On-disk cache of model predictions, keyed by (model, precision, prompt).

    A re-run of the same eval against the same model never re-generates:
    the prediction is looked up instead. This is the main lever for the
    per-PR cost called out in the backlog (E5): generation dominates wall
    time, and most CI re-runs change nothing about the model or the eval.
    """

    def __init__(self, directory: str | None, model_name: str | None):
        self._dir = directory
        self._model = model_name or ""
        self._data: dict[str, str] = {}
        self._path: str | None = None
        if directory:
            os.makedirs(directory, exist_ok=True)
            self._path = os.path.join(directory, "responses.json")
            if os.path.exists(self._path):
                try:
                    with open(self._path, encoding="utf-8") as f:
                        self._data = json.load(f)
                except (OSError, json.JSONDecodeError):
                    self._data = {}

    def _key(self, precision: str, prompt: str) -> str:
        raw = f"{self._model}x00{precision}x00{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, precision: str, prompt: str) -> str | None:
        if self._path is None:
            return None
        return self._data.get(self._key(precision, prompt))

    def put(self, precision: str, prompt: str, prediction: str) -> None:
        if self._path is None:
            return
        self._data[self._key(precision, prompt)] = prediction
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
        except OSError:
            pass


def _measured_dtype(model) -> str:
    """The dtype of the weights that actually carry the precision claim.

    Linear weights are what ``quantize_dynamic`` swaps and what ``.half()`` /
    ``.bfloat16()`` cast, so they are the honest signal: a quantized arm reports
    ``torch.qint8``, fp16 reports ``torch.float16``. Embeddings and norms stay
    float32 by design and would report the wrong thing. Duck-typed so the
    harness keeps working when the model object is a test double.
    """
    if model is None:
        return ""
    named_modules = getattr(model, "named_modules", None)
    walk = cast(Iterable, named_modules() if callable(named_modules) else ())
    for _, mod in walk:
        # A dynamically quantized Linear keeps its dtype on the packed params;
        # its ``weight`` is a bound method that dequantizes to float32, so
        # reading weight.dtype off it would report the wrong arm.
        packed = getattr(mod, "_packed_params", None)
        packed_dtype = getattr(packed, "dtype", None)
        if packed_dtype is not None:
            return str(packed_dtype)
        weight = getattr(mod, "weight", None)
        if weight is None or not hasattr(weight, "dtype"):
            continue
        if type(mod).__name__.endswith("Linear"):
            return str(weight.dtype)
    parameters = getattr(model, "parameters", None)
    params_iter = cast(Iterable, parameters() if callable(parameters) else ())
    params = [
        p for p in params_iter if hasattr(p, "dtype")
    ]
    if params:
        biggest = max(params, key=lambda p: p.numel())
        return str(biggest.dtype)
    return ""


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
        model_name: str | None = None,
        response_cache_dir: str | None = None,
    ):
        self._default_scorer = scorer is None
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
        self._cache = _ResponseCache(response_cache_dir, model_name)

    # -- public -----------------------------------------------------------
    def _matches_default(self, pred: str, case: Case) -> bool:
        got = normalise(pred)
        return got == case.expected or got in case.accept

    def compare(
        self,
        eval_set: EvalSet,
        baseline_precision: str = "fp32",
        candidate_precisions: Sequence[str] = ("int8",),
        max_drop_points: float = 2.0,
        min_accuracy: float | None = None,
        *,
        require_significance: bool = False,
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

        cases = list(eval_set)
        baseline_wrong = set(baseline.misclassified)
        baseline_hits = [0 if c.id in baseline_wrong else 1 for c in cases]
        drop_ci: dict[str, tuple[float, float]] = {}
        for cand in candidates:
            cand_wrong = set(cand.misclassified)
            cand_hits = [0 if c.id in cand_wrong else 1 for c in cases]
            drop_ci[cand.precision] = paired_bootstrap_drop_ci(
                baseline_hits, cand_hits, resamples=10000, alpha=0.05, seed=0
            )

        points_per_case = 100.0 / len(eval_set)
        tolerance_reachable = (
            max_drop_points == 0.0
            or max_drop_points >= points_per_case - 1e-9
        )
        granularity_warning = None
        if not tolerance_reachable:
            granularity_warning = (
                f"tolerance is finer than the eval set can express: {len(eval_set)} cases "
                f"means one case is {points_per_case:.2f} points, but "
                f"--max-drop-points is {max_drop_points:g}; in effect ANY new error fails "
                f"the run. Widen the set or raise the tolerance to >= {points_per_case:.2f}."
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

        # Significance handling.
        over_tol = [
            c for c in candidates
            if (baseline.accuracy - c.accuracy) * 100.0 > max_drop_points
        ]
        all_over_not_sig = bool(over_tol) and all(
            not (drop_ci.get(c.precision, (0.0, 0.0))[0] > 0) for c in over_tol
        )

        # Triggering candidate = worst drop.
        worst_cand = None
        if candidates:
            worst_cand = max(
                candidates, key=lambda c: (baseline.accuracy - c.accuracy) * 100.0
            )

        flipped = (
            require_significance
            and floor_failure is None
            and bool(over_tol)
            and all_over_not_sig
        )

        if floor_failure is not None:
            verdict = Verdict.FAIL
        elif worst > max_drop_points:
            if flipped:
                verdict = Verdict.PASS
            else:
                verdict = Verdict.FAIL
        else:
            verdict = Verdict.PASS

        significance_note: str | None = None
        if flipped:
            assert worst_cand is not None
            c = worst_cand
            iv = drop_ci[c.precision]
            drop_val = (baseline.accuracy - c.accuracy) * 100.0
            significance_note = (
                f"the drop of {drop_val:.1f} points is inside the sampling "
                f"noise: 95% CI {iv[0]:.1f} to {iv[1]:.1f} points for "
                f"{c.precision}, so --require-significance reports PASS. "
                f"Widen the eval set to resolve a smaller drop."
            )
        elif (
            floor_failure is None
            and worst > max_drop_points
            and worst_cand is not None
            and not (drop_ci.get(worst_cand.precision, (0.0, 0.0))[0] > 0)
        ):
            c = worst_cand
            iv = drop_ci[c.precision]
            drop_val = (baseline.accuracy - c.accuracy) * 100.0
            significance_note = (
                f"note: the drop of {drop_val:.1f} points is inside the "
                f"sampling noise: 95% CI {iv[0]:.1f} to {iv[1]:.1f} points "
                f"for {c.precision}. Widen the eval set to resolve a smaller drop."
            )

        return ComparisonResult(
            baseline=baseline,
            candidates=candidates,
            max_drop_points=max_drop_points,
            verdict=verdict,
            min_accuracy=min_accuracy,
            floor_failure=floor_failure,
            points_per_case=points_per_case,
            tolerance_reachable=tolerance_reachable,
            granularity_warning=granularity_warning,
            drop_ci=drop_ci,
            require_significance=require_significance,
            significance_note=significance_note,
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
        # Skip the (expensive) model load entirely when every case is already
        # cached: a fully cached re-run then never touches the network or the
        # weights, which is the biggest remaining cost lever.
        cached_all = (
            self._cache.get(precision, c.prompt) is not None
            for c in eval_set
        )
        all_cached = all(cached_all)
        if all_cached:
            model = None
        else:
            model = self._factory(precision)
        tok = None
        model_dtype = ""
        try:
            if model is not None:
                tok = getattr(model, "tokenizer", None)
                model_dtype = _measured_dtype(model)
            correct = 0
            mis: list[str] = []
            scored = 0
            cat_counts = defaultdict(lambda: [0, 0])
            t0 = time.perf_counter()
            for i, case in enumerate(eval_set, 1):
                cached = self._cache.get(precision, case.prompt)
                if cached is not None:
                    pred = cached
                else:
                    pred = self._predict(model, tok, case.prompt)
                    self._cache.put(precision, case.prompt, pred)
                ok = self._matches_default(pred, case) if self._default_scorer else self._scorer(pred, case.expected)
                correct += int(ok)
                scored += 1
                cat_counts[case.category][1] += 1
                if ok:
                    cat_counts[case.category][0] += 1
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
                model_dtype=str(model_dtype),
                per_category={k: (v[0], v[1]) for k, v in cat_counts.items()},
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

    def _prompt_text(self, tok, prompt: str) -> str:
        """Render the prompt the way the model itself expects to read it.

        An instruction-tuned checkpoint carries its own chat template, which
        is how ``--system-prompt`` is supposed to reach it. Pasting
        ``f"{system_prompt}\\n{prompt}"`` instead produces input the model was
        never trained on: it answers with conversational filler rather than
        the expected token, so the accuracy being measured is the accuracy
        of a *broken prompt*, and the marginal-decision damage this tool
        exists to detect is invisible.

        ``apply_chat_template`` is HF's standard hook for this and it is what
        a tokenizer exposes exactly when the model is a chat model, so using
        it is not a heuristic -- it is asking the model how it wants to be
        addressed.

        The test for "is a chat model" is ``chat_template`` being SET, not
        ``apply_chat_template`` being callable. Every transformers tokenizer
        has the method -- including base models like BERT that have no
        template at all -- and calling it there raises
        ``ValueError: Cannot use chat template functions because
        tokenizer.chat_template is not set``. That mistake turned every
        non-chat model into exit 2, and the fake-based tests could not see
        it because a fake only has the attribute if the test gives it one.
        """
        if _has_chat_template(tok):
            conversation: list[dict[str, str]] = []
            if self.system_prompt is not None:
                conversation.append(
                    {"role": "system", "content": self.system_prompt}
                )
            conversation.append({"role": "user", "content": prompt})
            return tok.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        if self.system_prompt is None:
            return prompt
        return f"{self.system_prompt}\n{prompt}"

    def _predict_one(self, model, tok, prompt: str) -> str:
        """The three supported model shapes, in priority order.

        Kept separate from :meth:`_predict` so the wrapper's except clause
        cannot swallow its own error type.
        """
        # 2. explicit text answer
        answer = getattr(model, "answer", None)
        if callable(answer):
            return str(answer(prompt))

        text = self._prompt_text(tok, prompt)

        # 1. real HF generation path — continuation only
        if tok is not None and hasattr(model, "generate"):
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
