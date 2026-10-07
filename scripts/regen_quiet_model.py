"""Regenerate the 'change is not quality' figures quoted in the README.

A regression gate measures CHANGE. That means a model which is equally bad at
every precision produces a zero-point drop and passes -- a green build for a
model that cannot do the task at all.

This script measures exactly that, and then measures what `--min-accuracy` does
about it, so the README's section is backed by a run rather than by assertion.

Run: /usr/bin/python3.12 scripts/regen_quiet_model.py

Writes results/quiet_model.json and prints the figures the README quotes. The
stub stands in for any model whose eval set it cannot answer; no real checkpoint
is involved, because the point is about the GATE's arithmetic, not about a model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant_regress.evalset import Case, EvalSet          # noqa: E402
from quant_regress.harness import QuantHarness           # noqa: E402

N_CASES = 10


class _QuietModel:
    """Wrong at every precision -- e.g. a model given an eval it cannot answer.

    The quantized arm is wrong too, exactly like the baseline. That is the whole
    point: quantization changed nothing, so the delta is zero and the default
    gate has nothing to complain about.
    """

    def __init__(self, precision: str):
        self.precision = precision

    def eval(self):
        return self

    def answer(self, prompt: str) -> str:
        return "no"


class _HalfModel:
    """Half right at every precision -- the subtler version of the same trap."""

    def __init__(self, precision: str):
        self.precision = precision
        self.n = 0

    def eval(self):
        return self

    def answer(self, prompt: str) -> str:
        self.n += 1
        return "yes" if self.n % 2 == 1 else "no"


def _eval_set() -> EvalSet:
    return EvalSet(
        [Case(id=f"q{i}", prompt=f"q{i}", expected="yes") for i in range(N_CASES)]
    )


def _row(label: str, res) -> dict:
    return {
        "case": label,
        "baseline_accuracy": round(res.baseline.accuracy, 4),
        "worst_drop_points": round(res.worst_drop_points, 3),
        "verdict": res.verdict.value,
        "min_accuracy": res.min_accuracy,
        "baseline_below_floor": res.baseline_below_floor,
    }


def main() -> int:
    es = _eval_set()
    rows = []

    quiet = QuantHarness(model_factory=lambda p: _QuietModel(p))
    rows.append(_row("quiet_model_default", quiet.compare(
        es, candidate_precisions=["int8"], max_drop_points=2.0)))
    rows.append(_row("quiet_model_floored", quiet.compare(
        es, candidate_precisions=["int8"], max_drop_points=2.0, min_accuracy=0.5)))

    half = QuantHarness(model_factory=lambda p: _HalfModel(p))
    rows.append(_row("half_model_floored", half.compare(
        es, candidate_precisions=["int8"], max_drop_points=2.0, min_accuracy=0.5)))

    payload = {"n_cases": N_CASES, "rows": rows}
    out = ROOT / "results" / "quiet_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("| case | baseline accuracy | worst drop | floor | verdict |")
    print("|---|---|---|---|---|")
    for r in rows:
        floor = "-" if r["min_accuracy"] is None else f"{r['min_accuracy']:.2f}"
        print(f"| {r['case']} | {r['baseline_accuracy'] * 100:.1f}% | "
              f"{r['worst_drop_points']:+.2f} pts | {floor} | {r['verdict'].upper()} |")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())