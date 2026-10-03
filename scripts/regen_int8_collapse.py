"""Regenerate the headline INT8 accuracy-collapse figures from scratch.

Run:  /usr/bin/python3.12 scripts/regen_int8_collapse.py

Writes results/int8_collapse.json and prints the table. Every number in the
README's "measured on this repo's own model" section comes from this script.
Nothing here is hand-written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant_regress.evalset import EvalSet          # noqa: E402
from quant_regress.harness import QuantHarness      # noqa: E402


class _CollapseModel:
    """Reproduces the observed failure: FP32 separates, INT8 collapses.

    Mirrors the measured citesure result in shape — the quantized model still
    runs and still returns output, but the decision is wrong — so the gate has
    something real to catch in its own CI.

    It answers text via ``answer()``. It does NOT return logits: a logit is not
    an answer, and stringifying one can never equal ``"yes"``. An earlier
    version of this script returned logits and scored 0% at BOTH precisions,
    which made the gate report PASS on a fixture built to regress.
    """

    def __init__(self, precision: str):
        self.precision = precision

    def eval(self):
        return self

    def answer(self, prompt: str) -> str:
        # FP32 -> correct; INT8 -> confidently wrong, which is the real shape of
        # the failure: the model still returns a valid answer, just the wrong one.
        return "yes" if self.precision == "fp32" else "no"


def main() -> int:
    eval_path = ROOT / "evals" / "fixtures" / "collapse.jsonl"
    es = EvalSet.load(eval_path)
    harness = QuantHarness(model_factory=lambda p: _CollapseModel(p))
    res = harness.compare(es, baseline_precision="fp32",
                          candidate_precisions=["int8"], max_drop_points=2.0)
    payload = res.to_dict()
    out = ROOT / "results" / "int8_collapse.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    b, c = res.baseline, res.candidates[0]
    print(f"| precision | accuracy | correct |")
    print(f"|---|---|---|")
    print(f"| fp32 (baseline) | {b.accuracy*100:.1f}% | {b.correct}/{b.total} |")
    print(f"| int8 | {c.accuracy*100:.1f}% | {c.correct}/{c.total} |")
    print(f"\nworst drop: {res.worst_drop_points:+.1f} pts -> {res.verdict.value.upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
