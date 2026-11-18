"""Measure a real model across selectable precisions and eval sets.

The published tables under ``results/`` are produced by this script. It runs a
real checkpoint through the same ``QuantHarness``/adapter path the GitHub Action
uses, over an eval set you choose, and writes a JSON artifact plus the table
the README quotes.

The eval set is selectable (``--set``), as are the precisions (``--precisions``),
the output path (``--out``), the baseline-quality floor (``--min-accuracy``) and
a case cap for smoke runs (``--max-cases``). Nothing here is hard-coded to a
single task: point ``--set`` at any JSONL eval set and the measurement runs.

Honest outcomes, all publishable
--------------------------------
The gate measures a CHANGE. Dynamic int8 on a small model either costs accuracy
(the failure the tool exists to catch), or it does not (a PASS -- which is the
correct verdict, not a failed experiment), or the model cannot do the task at
either precision (the ``--min-accuracy`` floor catches that as FAIL). Whichever
happens, it is measured rather than asserted, and the script prints which one
it was.

Run:
    /home/hermes/citesure/.venv/bin/python scripts/regen_real_model.py --measure
    QUANT_REGRESS_REAL_MODEL=<hf-id> /home/hermes/citesure/.venv/bin/python \\
        scripts/regen_real_model.py --measure --set evals/fixtures/collapse.jsonl
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

_DEFAULT_SET_CANDIDATE = ROOT / "evals" / "sets" / "arith-tiered.jsonl"
_DEFAULT_SET = (
    _DEFAULT_SET_CANDIDATE
    if _DEFAULT_SET_CANDIDATE.exists()
    else ROOT / "evals" / "fixtures" / "collapse.jsonl"
)
DEFAULT_SET = _DEFAULT_SET
DEFAULT_OUT = ROOT / "results" / "real_model.json"


def numeral(pred: str, expected: str) -> bool:
    """Score by the numeral the model produced, not by its exact wording.

    Two scoring bugs, both caught by looking at the raw completions rather than at the
    score, and both of which produced a 0/20 "measurement":

    1. Exact-string equality scored 0/20 because a model answers "The answer is 12." rather
       than "12" -- that measures verbosity, not arithmetic.
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
    cannot support a collapse conclusion -- the README must say which side of the
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
        help="load the model and regenerate the output (CPU, several minutes). "
             "Without it the committed result is printed -- that is the fast path.",
    )
    ap.add_argument(
        "--set", default=str(DEFAULT_SET),
        help="path to a JSONL eval set (id, prompt, expected).",
    )
    ap.add_argument(
        "--precisions", default="fp32,int8",
        help="comma-separated precisions; the first is the baseline.",
    )
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="where to write (or read, in fast mode) the JSON result.",
    )
    ap.add_argument(
        "--min-accuracy", type=float, default=None,
        help="fail if the BASELINE scores below this fraction (0.5 = 50%%).",
    )
    ap.add_argument(
        "--max-cases", type=int, default=None,
        help="truncate the eval set to this many cases (smoke runs).",
    )
    args = ap.parse_args(argv)

    set_path = Path(args.set)
    if not set_path.is_absolute():
        set_path = ROOT / set_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    # Fast path: print the committed artifact when it already exists and an
    # explicit regeneration was not requested. Missing artifact is an error,
    # not an invitation to spend ten minutes of CPU by accident.
    if not args.measure:
        if not out_path.exists():
            shown = out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path
            print(f"no committed result at {shown} -- run with --measure first",
                  file=sys.stderr)
            return 2
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"model: {payload.get('model')}")
        print("(printed from the committed run; pass --measure to regenerate)")
        print()
        print_table(payload)
        return 0

    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    if len(precisions) < 2:
        print("--precisions must contain at least a baseline and one candidate",
              file=sys.stderr)
        return 2

    started = time.time()
    print(f"model: {MODEL}")
    es = EvalSet.load(set_path)
    if args.max_cases is not None and args.max_cases < len(es):
        es = EvalSet(list(es)[: args.max_cases])
    print(f"cases: {len(es)} · precisions: {', '.join(precisions)} · max_new_tokens: {MAX_NEW_TOKENS}")

    factory = build_model_factory(MODEL, max_new_tokens=MAX_NEW_TOKENS)
    harness = QuantHarness(
        model_factory=factory, max_new_tokens=MAX_NEW_TOKENS,
        scorer=numeral,
    )

    res = harness.compare(
        es,
        baseline_precision=precisions[0],
        candidate_precisions=precisions[1:],
        max_drop_points=2.0,
        min_accuracy=args.min_accuracy,
    )

    payload = res.to_dict()
    payload["model"] = MODEL
    try:
        payload["eval"] = str(set_path.relative_to(ROOT))
    except ValueError:
        payload["eval"] = str(set_path)
    payload["points_per_case"] = round(100 / len(es), 3)
    payload["max_new_tokens"] = MAX_NEW_TOKENS
    payload["wall_seconds"] = round(time.time() - started, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print_table(payload)
    if args.min_accuracy is not None and payload["baseline"]["accuracy"] < args.min_accuracy:
        print("NOTE: the baseline itself is below the --min-accuracy floor -- read this as 'this model "
              "cannot do the task', not as a pass on quality.")
    print(f"\nwrote {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path} in {payload['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
