"""Core harness: run a task eval at several precisions and compare accuracy.

The point of this module is to make the marginal-decision damage visible.
Token-level proxies (perplexity, KL) routinely miss it; task accuracy does
not, provided the whole set is actually run — which the harness enforces by
scoring every case at every precision and refusing to report a partial run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from .evalset import EvalSet, normalise

SUPPORTED_PRECISIONS = ("fp32", "int8")


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
                }
                for c in self.candidates
            ],
            "worst_drop_points": round(self.worst_drop_points, 3),
            "max_drop_points": self.max_drop_points,
            "verdict": self.verdict.value,
        }


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

    # -- public -----------------------------------------------------------
    def compare(
        self,
        eval_set: EvalSet,
        baseline_precision: str = "fp32",
        candidate_precisions: Sequence[str] = ("int8",),
        max_drop_points: float = 2.0,
    ) -> ComparisonResult:
        for p in (baseline_precision, *candidate_precisions):
            if p not in SUPPORTED_PRECISIONS:
                raise ValueError(
                    f"unsupported precision {p!r}; supported: {SUPPORTED_PRECISIONS}"
                )
        if not len(eval_set):
            raise ValueError("eval set is empty")

        baseline = self._measure(eval_set, baseline_precision)
        candidates = [self._measure(eval_set, p) for p in candidate_precisions]

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
        return ComparisonResult(
            baseline=baseline,
            candidates=candidates,
            max_drop_points=max_drop_points,
            verdict=Verdict.FAIL if worst > max_drop_points else Verdict.PASS,
        )

    # -- internals --------------------------------------------------------
    def _measure(self, eval_set: EvalSet, precision: str) -> PrecisionResult:
        model = self._factory(precision)
        tok = getattr(model, "tokenizer", None)
        correct = 0
        mis: list[str] = []
        t0 = time.perf_counter()
        for case in eval_set:
            pred = self._predict(model, tok, case.prompt)
            ok = self._scorer(pred, case.expected)
            correct += int(ok)
            if not ok:
                mis.append(case.id)
        return PrecisionResult(
            precision=precision,
            correct=correct,
            total=len(eval_set),
            seconds=time.perf_counter() - t0,
            misclassified=mis,
        )

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
        """
        # 2. explicit text answer
        answer = getattr(model, "answer", None)
        if callable(answer):
            return str(answer(prompt))

        # 1. real HF generation path
        if tok is not None and hasattr(model, "generate"):
            text = prompt if self.system_prompt is None else f"{self.system_prompt}\n{prompt}"
            inputs = tok(text, return_tensors="pt")
            inputs = {k: v.to(getattr(model, "device", "cpu")) for k, v in inputs.items()}
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                 do_sample=False)
            return tok.decode(out[0], skip_special_tokens=True)

        # 3. logits -> label
        logits = model(**{"input_ids": [[1]], "attention_mask": [[1]]}).logits
        row = logits[0]
        idx = int(max(range(len(row)), key=lambda i: float(row[i])))
        id2label = getattr(model, "config", None)
        id2label = getattr(id2label, "id2label", None) if id2label else None
        if id2label:
            return str(id2label.get(idx, idx))
        if self.labels:
            return str(self.labels[idx]) if idx < len(self.labels) else str(idx)
        raise RuntimeError(
            "cannot map model output to text: the model exposes no .answer(), "
            "no generate()+tokenizer, and no id2label. Give the fake model an "
            "answer(prompt) method, or pass labels=[...] to QuantHarness."
        )
