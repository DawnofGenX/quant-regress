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
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .adapter import AdapterError, build_model_factory
from .evalset import EvalSet, EvalSetError, normalise
from .harness import SUPPORTED_PRECISIONS, QuantHarness, Verdict

EXIT_OK, EXIT_REGRESSION, EXIT_USAGE = 0, 1, 2

#: Output formats the CLI can emit. `text` is the human table, `json` is the
#: machine report, `junit` is for Jenkins/GitLab/GitHub test-result panes.
OUTPUT_FORMATS = ("text", "json", "junit")

#: Scorers exposed on the CLI. Deliberately a fixed pair: accepting an
#: arbitrary expression would be a code-execution hole in a tool people run
#: on arbitrary PRs, and two options cover the real need. Anything more
#: belongs in the library API, where QuantHarness(scorer=...) already exists.
SCORERS = {
    "exact": lambda pred, expected: normalise(pred) == expected,
    "contains": lambda pred, expected: expected in normalise(pred),
}


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
    p.add_argument("--scorer", default="exact", choices=sorted(SCORERS),
                   help="how a prediction is matched against `expected`. "
                        "exact = normalised equality (default); contains = "
                        "the expected answer appears anywhere in the "
                        "prediction, for a chatty model.")
    p.add_argument("--system-prompt", default=None,
                   help="prepended to every prompt. Withheld from the "
                        "decoded answer, so a system prompt cannot leak into "
                        "the prediction.")
    p.add_argument("--labels", default=None,
                   help="comma-separated label names for a logits-only model "
                        "that has no id2label, e.g. 'no,yes'")
    p.add_argument("--hf-token", default=None,
                   help="HF access token for a gated model. Falls back to "
                        "$HF_TOKEN. Never logged.")
    p.add_argument("--output-format", default="text", choices=OUTPUT_FORMATS,
                   help="text = human table; json = also write --report; "
                        "junit = test-result XML for CI panes")
    p.add_argument("--junit-path", default=None,
                   help="where to write JUnit XML (default: alongside "
                        "--report, or quant-regress-junit.xml)")
    p.add_argument("--progress-every", type=int, default=25,
                   help="print a progress line every N cases to stderr "
                        "(0 disables). Stdout is never polluted.")
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


def _junit(res) -> ET.Element:
    """One <testcase> per precision; a failing arm carries <failure>.

    JUnit is what Jenkins, GitLab and the GitHub test-results pane read, so
    a regression shows up as a named failing test rather than a log line.
    """
    suite = ET.Element("testsuite", name="quant-regress")
    failures = 0
    arms = [(res.baseline.precision, res.baseline, None)]
    for c in res.candidates:
        drop = (res.baseline.accuracy - c.accuracy) * 100.0
        arms.append((c.precision, c, drop))

    for precision, result, drop in arms:
        case = ET.SubElement(suite, "testcase", name=precision, time=f"{result.seconds:.3f}")
        if drop is not None and drop > res.max_drop_points:
            failures += 1
            failed_ids = ",".join(result.misclassified[:20])
            ET.SubElement(
                case, "failure",
                message=f"accuracy dropped {drop:.2f} points past "
                        f"{res.max_drop_points:.2f}",
                type="accuracyRegression",
            ).text = (
                f"{result.correct}/{result.total} correct at {precision}; "
                f"baseline {res.baseline.accuracy * 100:.1f}%, "
                f"candidate {result.accuracy * 100:.1f}%. "
                f"Misclassified: {failed_ids or 'none recorded'}"
            )
    if res.floor_failure:
        failures += 1
        case = ET.SubElement(suite, "testcase", name="baseline_quality_floor")
        ET.SubElement(case, "failure", message="baseline below floor",
                      type="qualityFloor").text = res.floor_failure

    suite.set("tests", str(len(suite.findall("testcase"))))
    suite.set("failures", str(failures))
    return suite


def _progress_printer(every: int):
    """A stderr progress callback, or None when disabled.

    Goes to stderr so stdout stays a parseable table -- a workflow that
    greps the summary line must not see progress noise mixed in.
    """
    if every <= 0:
        return None

    def report(done: int, total: int, precision: str) -> None:
        if done == 1 or done % every == 0 or done == total:
            print(
                f"\r[{precision}] {done}/{total} cases",
                end="" if done != total else "\n",
                file=sys.stderr,
            )

    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Validate precisions BEFORE any model is built. A typo like "int4"
    # otherwise costs a full download to discover, and the error surfaces
    # after minutes of work instead of instantly.
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    for p in (args.baseline_precision, *precisions):
        if p not in SUPPORTED_PRECISIONS:
            print(
                f"error: unsupported precision {p!r}; supported: "
                f"{list(SUPPORTED_PRECISIONS)}",
                file=sys.stderr,
            )
            return EXIT_USAGE

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

    scorer = SCORERS[args.scorer]
    labels = (
        [s.strip() for s in args.labels.split(",") if s.strip()]
        if args.labels else None
    )
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    try:
        factory = build_model_factory(
            args.model, cache_dir=args.cache_dir,
            max_new_tokens=args.max_new_tokens, hf_token=hf_token,
        )
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    harness = QuantHarness(
        model_factory=factory,
        max_new_tokens=args.max_new_tokens,
        scorer=scorer,
        system_prompt=args.system_prompt,
        labels=labels,
        progress=_progress_printer(args.progress_every),
    )
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

    if args.output_format == "junit":
        junit_path = args.junit_path or (
            str(Path(args.report).with_suffix(".junit.xml"))
            if args.report else "quant-regress-junit.xml"
        )
        try:
            root = _junit(res)
            ET.indent(root)
            Path(junit_path).parent.mkdir(parents=True, exist_ok=True)
            ET.ElementTree(root).write(
                junit_path, encoding="utf-8", xml_declaration=True
            )
        except OSError as exc:
            print(f"error: could not write JUnit XML to {junit_path}: {exc}",
                  file=sys.stderr)
            return EXIT_USAGE
        print(f"junit report written to {junit_path}", file=sys.stderr)

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
