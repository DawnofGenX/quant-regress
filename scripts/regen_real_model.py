"""Measure a REAL model at fp32 and int8 — replacing the stub the headline used to rest on.

The README's lead table came from ``scripts/regen_int8_collapse.py``, whose model is
``_CollapseModel``: a stub that returns "yes" at fp32 and "no" at int8. That is a
demonstration of the gate's arithmetic, not a measurement of quantization, and the README
now says so. This script is the missing half: a real checkpoint, a real task, the same
harness the action uses.

What it does
------------
* loads a real instruction-tuned causal LM (small enough for CPU, real weights),
* runs a 20-item task with exactly checkable answers through the same
  ``QuantHarness``/adapter path the GitHub Action uses (fp32 baseline, int8 candidate),
* writes ``results/real_model.json`` and prints the table the README quotes.

Honest outcomes, all publishable
--------------------------------
The gate measures a CHANGE. Dynamic int8 on a small model either costs accuracy
(the failure the tool exists to catch), or it does not (a PASS — which is the
correct verdict, not a failed experiment), or the model cannot do the task at either
precision (the ``--min-accuracy`` floor from 1.0.3 catches that as FAIL). Whichever
happens, it is measured rather than asserted, and the script prints which one it was.

Run:
    /usr/bin/python3 scripts/regen_real_model.py
    QUANT_REGRESS_REAL_MODEL=<hf-id> /usr/bin/python3 scripts/regen_real_model.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant_regress.adapter import build_model_factory      # noqa: E402
from quant_regress.evalset import Case, EvalSet             # noqa: E402
from quant_regress.harness import QuantHarness              # noqa: E402

MODEL = os.environ.get("QUANT_REGRESS_REAL_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
MAX_NEW_TOKENS = 8
RESULTS = ROOT / "results" / "real_model.json"


def numeral(pred: str, expected: str) -> bool:
    """Score by the numeral the model produced, not by its exact wording.

    Two scoring bugs, both caught by looking at the raw completions rather than at the
    score, and both of which produced a 0/20 "measurement":

    1. Exact-string equality scored 0/20 because a model answers "The answer is 12." rather
       than "12" — that measures verbosity, not arithmetic.
    2. First-numeral extraction also scored 0/20, because a 135M model answers
       "7 + 5 = 12": the first numeral is the operand it echoed, not the result.

    The rule is therefore the LAST numeral in the completion, which is the answer in every
    one of these prompts ("7 + 5 = 12" -> 12).
    """
    found = re.findall(r"-?\d[\d,]*", pred.replace(",", ""))
    return bool(found) and found[-1] == expected.strip()

# Exactly checkable, no ambiguity about what a correct answer is: every prompt demands one
# numeral, and the scorer compares the normalised completion to the expected numeral.
ANSWERS = {
    "What is 7 plus 5?": "12",
    "What is 9 plus 6?": "15",
    "What is 13 plus 8?": "21",
    "What is 24 plus 17?": "41",
    "What is 50 minus 18?": "32",
    "What is 64 minus 27?": "37",
    "What is 100 minus 43?": "57",
    "What is 6 times 7?": "42",
    "What is 8 times 9?": "72",
    "What is 12 times 4?": "48",
    "A crate holds 15 bottles. Two crates hold how many bottles?": "30",
    "A shop sells 9 pens a day. How many pens in 7 days?": "63",
    "There are 40 seats and 27 are taken. How many are free?": "13",
    "A bus carries 52 people. 19 get off. How many remain?": "33",
    "Three boxes hold 14 apples each. How many apples in total?": "42",
    "A bill is 96 and split between 8 people. What does each pay?": "12",
    "What is 121 minus 64?": "57",
    "What is 18 plus 29?": "47",
    "What is 7 times 11?": "77",
    "A pool has 300 litres. 175 drain out. How many remain?": "125",
}


def build_eval_set() -> EvalSet:
    return EvalSet([
        Case(id=f"real-{i:03d}", prompt=f"{q} Answer with the number only.",
             expected=a)
        for i, (q, a) in enumerate(ANSWERS.items(), start=1)
    ])


def print_table(payload: dict) -> None:
    """Print the block the README quotes, so the committed artifact is fully
    auditable from the script's stdout alone.

    Includes per-arm wall-clock seconds and the baseline-vs-floor relationship,
    because a drop measured from a baseline below the ``--min-accuracy`` floor
    cannot support a collapse conclusion — the README must say which side of the
    floor the baseline sits on, not just the delta.
    """
    b = payload["baseline"]
    c = payload["candidates"][0]
    print("| precision | accuracy | correct | seconds |")
    print("|---|---|---|---|")
    print(f"| fp32 (baseline) | {b['accuracy'] * 100:.1f}% | {b['correct']}/{b['total']} | {b['seconds']:.1f}s |")
    print(f"| int8 | {c['accuracy'] * 100:.1f}% | {c['correct']}/{c['total']} | {c['seconds']:.1f}s |")
    print(f"\nworst drop: {payload['worst_drop_points']:+.1f} pts -> {payload['verdict'].upper()}")
    floor = payload.get("min_accuracy")
    if floor is not None:
        rel = "below" if payload.get("baseline_below_floor") else "above"
        print(f"baseline {b['accuracy'] * 100:.1f}% is {rel} the --min-accuracy floor ({floor})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--measure", action="store_true",
        help="load the model and regenerate results/real_model.json (CPU, several minutes). "
             "Without it the committed result is printed — that is the fast path the claims "
             "test uses to pin the README to the artifact.",
    )
    args = ap.parse_args(argv)

    # claim:real-model — the fast path is what tests/test_claims_are_reproducible.py
    # executes. It prints the committed artifact (results/real_model.json) and
    # nothing else, so the README's real-model table is pinned to the committed
    # run: mutate a figure in the README and the claims test fails, because the
    # number no longer appears in this output. `--measure` is the expensive
    # reproduction (loads the checkpoint, ~minutes on CPU) that a human runs to
    # regenerate the artifact; it is deliberately NOT what the test runs, so the
    # suite stays fast and deterministic.
    if not args.measure:
        if not RESULTS.exists():
            print(f"no committed result at {RESULTS.relative_to(ROOT)} — run with --measure first",
                  file=sys.stderr)
            return 2
        payload = json.loads(RESULTS.read_text(encoding="utf-8"))
        print(f"model: {payload.get('model')}")
        print("(printed from the committed run; pass --measure to regenerate)")
        print()
        print_table(payload)
        return 0

    started = time.time()
    print(f"model: {MODEL}")
    print(f"cases: {len(ANSWERS)} · precisions: fp32, int8 · max_new_tokens: {MAX_NEW_TOKENS}")

    factory = build_model_factory(MODEL, max_new_tokens=MAX_NEW_TOKENS)
    harness = QuantHarness(
        model_factory=factory, max_new_tokens=MAX_NEW_TOKENS,
        scorer=numeral,          # see numeral(): exact-string matching measured verbosity
    )
    es = build_eval_set()

    # min_accuracy=0.5: a model that cannot do the task at baseline must not produce a
    # green build just because quantization changed nothing. That is what 1.0.3 added.
    res = harness.compare(
        es,
        baseline_precision="fp32",
        candidate_precisions=["int8"],
        max_drop_points=2.0,
        min_accuracy=0.5,
    )

    payload = res.to_dict()
    payload["model"] = MODEL
    payload["task"] = "20 single-answer arithmetic word problems, scored on the numeral produced"
    payload["max_new_tokens"] = MAX_NEW_TOKENS
    payload["wall_seconds"] = round(time.time() - started, 1)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print_table(payload)
    if payload["baseline"]["accuracy"] < 0.5:
        print("NOTE: the baseline itself is below the 0.5 floor — read this as 'this model "
              "cannot do the task', not as a pass on quality.")
    print(f"\nwrote {RESULTS.relative_to(ROOT)} in {payload['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
