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
from quant_regress.evalset import EvalSet                   # noqa: E402
from quant_regress.harness import QuantHarness              # noqa: E402

MODEL = os.environ.get("QUANT_REGRESS_REAL_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
#: Generous on purpose. 8 tokens truncated every word problem that reasons
#: before answering, so the old "measurement" scored truncation, not arithmetic:
#: the same model went from 33% to a much higher baseline purely by letting it
#: finish a sentence. The scorer takes the LAST numeral, so extra prose is free.
DEFAULT_MAX_NEW_TOKENS = 32

DEFAULT_SET = ROOT / "evals" / "sets" / "arith-tiered.jsonl"

#: One committed artifact per published table. Without --measure the script
#: prints all of them, so a single no-argument run yields every number the
#: README claims and tests/test_claims_are_reproducible.py can check them.
ARTIFACTS = (
    ROOT / "results" / "real_model_arith.json",
    ROOT / "results" / "real_model_classify.json",
)


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


def pick_scorer(eval_set, choice: str):
    """Return (scorer, label). ``auto`` keeps exact scoring honest per task type.

    A numeral scorer on a sentiment set scores 0/60 at every precision and the
    gate reports a perfect PASS -- the exact false-green this tool exists to
    prevent. ``auto`` therefore uses the numeral scorer only when every expected
    answer is a numeral, and otherwise the harness default (exact match plus the
    set's declared aliases).
    """
    if choice == "numeral":
        return numeral, "numeral (last number in the completion)"
    if choice == "default":
        return None, "default (exact match + declared aliases)"
    numeric = all(re.fullmatch(r"-?\d+", c.expected.strip()) for c in eval_set)
    if numeric:
        return numeral, "numeral (last number in the completion)"
    return None, "default (exact match + declared aliases)"


def _progress(done: int, total: int, precision: str) -> None:
    """Keep a redirected log alive: without this a 20-minute CPU run looks dead."""
    if done == total or done % 10 == 0:
        print(f"  {precision}: {done}/{total}", file=sys.stderr, flush=True)


def print_table(payload: dict) -> None:
    """Print the block the README quotes, so the committed artifact is fully
    auditable from the script's stdout alone.

    Every precision in the run gets a row, and every row carries the dtype the
    arm ACTUALLY measured -- a table that says fp32 while the weights are bf16
    is the exact failure this column exists to prevent. Per-arm seconds and the
    baseline-vs-floor relationship are included because a drop measured from a
    baseline below the ``--min-accuracy`` floor cannot support a collapse
    conclusion.
    """
    b = payload["baseline"]
    print(f"set: {payload.get('eval', '?')} · model: {payload.get('model', '?')}")
    print("| precision | accuracy | correct | drop | 95% CI (pts) | measured dtype | seconds |")
    print("|---|---|---|---|---|---|---|")
    print(f"| {b['precision']} (baseline) | {b['accuracy'] * 100:.1f}% | "
          f"{b['correct']}/{b['total']} | - | - | {b.get('dtype') or 'n/a'} | "
          f"{b['seconds']:.1f}s |")
    for c in payload["candidates"]:
        drop = (b["accuracy"] - c["accuracy"]) * 100.0
        ci = c.get("drop_ci_95")
        ci_str = f"{ci[0]:.1f} to {ci[1]:.1f}" if ci else "n/a"
        print(f"| {c['precision']} | {c['accuracy'] * 100:.1f}% | "
              f"{c['correct']}/{c['total']} | {drop:+.1f} pts | {ci_str} | "
              f"{c.get('dtype') or 'n/a'} | {c['seconds']:.1f}s |")
    stats = payload.get("statistics") or {}
    if stats:
        print(f"\n95% CI: {stats.get('method')}, {stats.get('resamples')} resamples, "
              f"alpha {stats.get('alpha')} (paired on case identity, not on accuracy)")
    print(f"\nworst drop: {payload['worst_drop_points']:+.1f} pts -> {payload['verdict'].upper()}")
    floor = payload.get("min_accuracy")
    if floor is not None:
        rel = "below" if payload.get("baseline_below_floor") else "above"
        print(f"baseline {b['accuracy'] * 100:.1f}% is {rel} the --min-accuracy floor ({floor})")


def _difficulty_groups(eval_set) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for case in eval_set:
        label = getattr(case, "difficulty", "") or ""
        if label:
            groups.setdefault(label, []).append(case.id)
    return groups


def print_breakdown(payload: dict, eval_set=None) -> None:
    """Category and difficulty rows.

    "int8 collapses everywhere" and "int8 collapses only on multi-step cases"
    are different findings, and the aggregate table cannot tell them apart.
    Printed here so the README's per-group claims are auditable from stdout
    exactly like the headline numbers.
    """
    arms = [payload["baseline"], *payload["candidates"]]
    if any(arm.get("categories") for arm in arms):
        print("\n| arm | category | correct |")
        print("|---|---|---|")
        for arm in arms:
            for name, c in sorted((arm.get("categories") or {}).items()):
                print(f"| {arm['precision']} | {name} | {c['correct']}/{c['total']} |")
    groups = _difficulty_groups(eval_set) if eval_set is not None else {}
    if groups:
        print("\n| arm | difficulty | correct |")
        print("|---|---|---|")
        for arm in arms:
            wrong = set(arm.get("misclassified") or [])
            for name, ids in groups.items():
                ok = sum(1 for i in ids if i not in wrong)
                print(f"| {arm['precision']} | {name} | {ok}/{len(ids)} |")


def _load_set_for(payload: dict):
    rel = payload.get("eval")
    if not rel:
        return None
    try:
        return EvalSet.load(ROOT / rel)
    except Exception:  # a printed summary must never die on a missing set
        return None


def main(argv: list[str] | None = None) -> int:
    # Line-buffered so a redirected log shows progress instead of a silent hour.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--measure", action="store_true",
        help="load the model and regenerate the output (CPU, several minutes). "
             "Without it the committed result is printed -- that is the fast path.",
    )
    ap.add_argument(
        "--set", default=None,
        help=f"path to a JSONL eval set (default: {DEFAULT_SET.name}).",
    )
    ap.add_argument(
        "--precisions", default="fp32,int8",
        help="comma-separated precisions; the first is the baseline.",
    )
    ap.add_argument(
        "--out", default=None,
        help="where to write the JSON result (default: results/real_model_<set>.json).",
    )
    ap.add_argument(
        "--min-accuracy", type=float, default=None,
        help="fail if the BASELINE scores below this fraction (0.5 = 50%%).",
    )
    ap.add_argument(
        "--max-cases", type=int, default=None,
        help="truncate the eval set to this many cases (smoke runs).",
    )
    ap.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help="decode budget per case (too small measures truncation, not accuracy).",
    )
    ap.add_argument(
        "--scorer", default="auto", choices=["auto", "numeral", "default"],
        help="auto = numeral for all-numeric answers, exact+aliases otherwise.",
    )
    args = ap.parse_args(argv)

    set_path = Path(args.set or DEFAULT_SET)
    if not set_path.is_absolute():
        set_path = ROOT / set_path
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
    else:
        out_path = ROOT / "results" / f"real_model_{set_path.stem.split('-')[0]}.json"

    # Fast path: print the committed artifact when it already exists and an
    # explicit regeneration was not requested. Missing artifact is an error,
    # not an invitation to spend ten minutes of CPU by accident.
    if not args.measure:
        explicit = bool(args.out or args.set)
        paths = [out_path] if explicit else [p for p in ARTIFACTS if p.exists()]
        missing = [p for p in (paths or [out_path]) if not p.exists()]
        if missing:
            for path in missing:
                shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
                print(f"no committed result at {shown} -- run with --measure first",
                      file=sys.stderr)
            return 2
        for i, path in enumerate(paths):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if i:
                print()
            print("(printed from the committed run; pass --measure to regenerate)")
            print_table(payload)
            print_breakdown(payload, _load_set_for(payload))
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
    scorer, scorer_label = pick_scorer(es, args.scorer)
    print(f"cases: {len(es)} · precisions: {', '.join(precisions)} · "
          f"max_new_tokens: {args.max_new_tokens} · scorer: {scorer_label}")

    factory = build_model_factory(MODEL, max_new_tokens=args.max_new_tokens)
    harness = QuantHarness(
        model_factory=factory, max_new_tokens=args.max_new_tokens,
        scorer=scorer,
        progress=_progress,
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
    payload["max_new_tokens"] = args.max_new_tokens
    payload["scorer"] = scorer_label
    payload["wall_seconds"] = round(time.time() - started, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print_table(payload)
    print_breakdown(payload, es)
    if args.min_accuracy is not None and payload["baseline"]["accuracy"] < args.min_accuracy:
        print("NOTE: the baseline itself is below the --min-accuracy floor -- read this as 'this model "
              "cannot do the task', not as a pass on quality.")
    print(f"\nwrote {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path} in {payload['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
