"""Command-line entry point.

Exit codes are the contract with CI:

* ``0`` — within tolerance
* ``1`` — accuracy regressed past ``--max-drop-points``
* ``2`` — bad usage / configuration (missing model, unreadable eval set)

A JSON report is written when ``--report`` is given so a workflow can publish
it as an artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapter import AdapterError, build_model_factory
from .evalset import EvalSet, EvalSetError
from .harness import QuantHarness, Verdict

EXIT_OK, EXIT_REGRESSION, EXIT_USAGE = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quant-regress",
        description="Fail CI when quantization costs you task accuracy.",
    )
    p.add_argument("--eval", required=True, help="path to a JSONL task eval set")
    p.add_argument("--model", default=None, help="HF model id (required for a real run)")
    p.add_argument("--precisions", default="int8",
                   help="comma-separated candidate precisions (default: int8)")
    p.add_argument("--baseline-precision", default="fp32")
    p.add_argument("--max-drop-points", type=float, default=2.0,
                   help="fail if any candidate drops more than this many points")
    p.add_argument("--min-accuracy", type=float, default=None,
                   help="fail if the BASELINE scores below this fraction "
                        "(0.5 = 50%%). Without it, a model that is wrong at every "
                        "precision yields a 0.00-point drop and PASSes.")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--report", default=None, help="write a JSON report here")
    p.add_argument("--cache-dir", default=None)
    return p


def _markdown(res) -> str:
    b = res.baseline
    lines = [
        "| precision | accuracy | correct | drop |",
        "|---|---|---|---|",
        f"| {b.precision} (baseline) | {b.accuracy*100:.1f}% | {b.correct}/{b.total} | — |",
    ]
    for c in res.candidates:
        drop = (b.accuracy - c.accuracy) * 100.0
        lines.append(
            f"| {c.precision} | {c.accuracy*100:.1f}% | {c.correct}/{c.total} | "
            f"{drop:+.1f} pts |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        eval_set = EvalSet.load(args.eval)
    except EvalSetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not args.model:
        print(
            "error: --model is required (e.g. --model "
            "meta-llama/Llama-3.2-1B-Instruct)", file=sys.stderr
        )
        return EXIT_USAGE

    try:
        factory = build_model_factory(
            args.model, cache_dir=args.cache_dir,
            max_new_tokens=args.max_new_tokens,
        )
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    harness = QuantHarness(model_factory=factory, max_new_tokens=args.max_new_tokens)
    try:
        res = harness.compare(
            eval_set,
            baseline_precision=args.baseline_precision,
            candidate_precisions=[p.strip() for p in args.precisions.split(",") if p.strip()],
            max_drop_points=args.max_drop_points,
            min_accuracy=args.min_accuracy,
        )
    except (RuntimeError, ValueError) as exc:
        # ValueError covers the argument checks inside compare() -- an
        # unsatisfiable threshold, or no candidate precisions. Those are
        # configuration mistakes, so they must exit 2 like a missing model or
        # unreadable eval set, not traceback out of main().
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print(_markdown(res))
    print(f"\nworst drop: {res.worst_drop_points:+.2f} pts "
          f"(threshold {args.max_drop_points:+.2f}) -> {res.verdict.value.upper()}")
    if res.min_accuracy is not None:
        floor = res.min_accuracy * 100.0
        state = "BELOW" if res.baseline_below_floor else "ok"
        print(f"baseline accuracy: {res.baseline.accuracy * 100:.1f}% "
              f"(floor {floor:.1f}%) -> {state}")
    if res.floor_failure:
        print(f"\n{res.floor_failure}")

    if args.report:
        try:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(
                json.dumps(res.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            # The measurement is done and the verdict is in hand. An
            # unhandled OSError here would traceback out of main() and, on
            # CI, surface as exit 1 -- "accuracy regressed" -- for what is
            # actually a disk problem. Report it as a usage error instead.
            print(
                f"error: could not write the report to {args.report}: {exc}",
                file=sys.stderr,
            )
            return EXIT_USAGE

    return EXIT_OK if res.verdict is Verdict.PASS else EXIT_REGRESSION


if __name__ == "__main__":
    raise SystemExit(main())
