"""The published Action must translate its inputs into valid CLI argv.

The action's entrypoint used to bind `$1..$7` positionally. That is the
whole reason this test exists: the moment a flag is added to the CLI, the
positional list silently mis-binds and `--max-new-tokens` arrives as
`--min-accuracy`. A workflow then fails validation for no visible reason, or
worse, measures the wrong thing.

The contract pinned here: every action input reaches the CLI under the flag
it belongs to, and optional empties are dropped rather than forwarded as ""
(which would fail validation and, for --min-accuracy, disable the floor).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = ROOT / "entrypoint.sh"

#: Exactly what action.yml passes, in order. Kept in one place so a new
#: action input cannot be added here and forgotten in the action file.
ACTION_ARGS = [
    "--eval", "ev.jsonl",
    "--model", "meta-llama/Llama-3.2-1B-Instruct",
    "--precisions", "int8",
    "--baseline-precision", "fp32",
    "--max-drop-points", "2",
    "--min-accuracy", "",
    "--max-new-tokens", "32",
    "--scorer", "exact",
    "--output-format", "text",
    "--report", "quant-regress-report.json",
    "--hf-token", "",
    "--cache-dir", "",
]


@pytest.mark.skipif(
    not shutil.which("sh"), reason="a POSIX shell is required to run the entrypoint"
)
def _run_entrypoint(argv: list[str], tmp_path: Path) -> tuple[int, list[str]]:
    """Run entrypoint.sh with `python` replaced by an argv recorder."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recorder = bin_dir / "python"
    recorder.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$OUT_FILE\"\nexit 0\n",
        encoding="utf-8",
    )
    recorder.chmod(0o755)

    evalfile = tmp_path / "ev.jsonl"
    evalfile.write_text('{"id": "1", "prompt": "q?", "expected": "yes"}\n',
                        encoding="utf-8")

    # Copy the entrypoint next to a `python` that just records its argv, and
    # rewrite the one interpreter call so the real python is never invoked.
    src = ENTRYPOINT.read_text(encoding="utf-8")
    src = src.replace("python -m quant_regress.cli",
                      str(recorder) + " -m quant_regress.cli")
    script = tmp_path / "entrypoint.sh"
    script.write_text(src, encoding="utf-8")
    script.chmod(0o755)

    out = tmp_path / "argv.txt"
    proc = subprocess.run(
        ["sh", str(script), *argv],
        cwd=tmp_path, capture_output=True, text=True, env={"OUT_FILE": str(out)},
        timeout=60,
    )
    recorded = out.read_text(encoding="utf-8").splitlines() if out.exists() else []
    return proc.returncode, recorded


def test_action_inputs_reach_the_cli_under_the_right_flags(tmp_path):
    """Every action input must arrive under its own flag.

    Pinned against the recorded argv, so a positional mis-binding -- a flag
    arriving under the wrong name, or two flags swapped -- fails here.
    """
    rc, argv = _run_entrypoint(ACTION_ARGS, tmp_path)
    assert rc == 0, "the entrypoint rejected the action's own argument shape"
    pairs = dict(zip(argv[2::2], argv[3::2])) if len(argv) > 2 else {}
    assert pairs.get("--eval") == "ev.jsonl", argv
    assert pairs.get("--model") == "meta-llama/Llama-3.2-1B-Instruct", argv
    assert pairs.get("--precisions") == "int8", argv
    assert pairs.get("--baseline-precision") == "fp32", argv
    assert pairs.get("--max-drop-points") == "2", argv
    assert pairs.get("--max-new-tokens") == "32", argv
    assert pairs.get("--scorer") == "exact", argv
    assert pairs.get("--output-format") == "text", argv
    assert pairs.get("--report") == "quant-regress-report.json", argv
    # Each flag must appear exactly once: a duplicate means the positional
    # list and the new named path are both firing.
    for flag in ("--eval", "--model", "--max-new-tokens", "--max-drop-points"):
        assert argv.count(flag) == 1, f"{flag} appears {argv.count(flag)} times in {argv}"


def test_empty_optional_inputs_are_dropped_not_forwarded(tmp_path):
    """An empty --min-accuracy would fail validation ("" is not a fraction).

    The flag must stay opt-in: forwarding it as an empty string would turn an
    unset floor into a usage error, and forward it for --hf-token would make
    every unauthenticated run pass an empty token.
    """
    rc, argv = _run_entrypoint(ACTION_ARGS, tmp_path)
    assert rc == 0
    assert "--min-accuracy" not in argv, (
        f"an empty min-accuracy was forwarded and would fail validation: {argv}"
    )
    assert "--hf-token" not in argv, (
        f"an empty hf-token was forwarded: {argv}"
    )
    assert "--cache-dir" not in argv, argv


def test_optional_inputs_are_forwarded_when_supplied(tmp_path):
    rc, argv = _run_entrypoint(
        ["--eval", "ev.jsonl", "--model", "m", "--precisions", "int8",
         "--baseline-precision", "fp32", "--max-drop-points", "2",
         "--min-accuracy", "0.5", "--max-new-tokens", "16", "--scorer",
         "contains", "--output-format", "junit", "--report", "out/r.json",
         "--hf-token", "hf_SHOULD_NOT_LEAK", "--cache-dir", "/tmp/cache"],
        tmp_path,
    )
    assert rc == 0
    assert "0.5" in argv and "--min-accuracy" in argv, argv
    assert "--hf-token" in argv and "hf_SHOULD_NOT_LEAK" in argv
    assert "--cache-dir" in argv and "/tmp/cache" in argv
    # The token must never be printed by the entrypoint.
    assert "SHOULD_NOT_LEAK" not in (tmp_path / "entrypoint.sh").read_text()


def test_unknown_argument_is_refused(tmp_path):
    rc, _ = _run_entrypoint(
        ["--eval", "ev.jsonl", "--model", "m", "--bogus", "x"], tmp_path
    )
    assert rc == 2, "an unknown argument must be a usage error, never ignored"


def test_missing_value_is_refused(tmp_path):
    rc, _ = _run_entrypoint(["--eval", "ev.jsonl", "--model", "m", "--hf-token"],
                            tmp_path)
    assert rc == 2, "a flag with no value must be refused, not read as empty"


def test_missing_eval_file_is_refused_before_any_work(tmp_path):
    """The check must fire before the CLI is invoked at all."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (bin_dir / "python").chmod(0o755)
    src = ENTRYPOINT.read_text(encoding="utf-8").replace(
        "python -m quant_regress.cli", f"{bin_dir}/python -m quant_regress.cli")
    script = tmp_path / "entrypoint.sh"
    script.write_text(src, encoding="utf-8")
    script.chmod(0o755)
    proc = subprocess.run(
        ["sh", str(script), "--eval", "nonexistent.jsonl", "--model", "m"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2, proc.stderr
    # GitHub renders ::error:: from STDOUT, so that is where it must land.
    assert "eval set not found" in (proc.stdout + proc.stderr), (
        f"the refusal must say what was wrong: {proc.stdout!r} {proc.stderr!r}"
    )


def test_action_yaml_passes_only_flags_the_entrypoint_understands(tmp_path):
    """action.yml and entrypoint.sh must agree on the flag vocabulary.

    Adding an action input the entrypoint does not parse would make it
    `unknown argument` and exit 2 for every user of that input -- a
    config error in our own release.
    """
    action = (ROOT / "action.yml").read_text(encoding="utf-8")

    # Collect the `- --flag` list items under `args:` only. A YAML value like
    # `- ${{ inputs.eval }}` is skipped. Note strip() leaves "- --eval", so
    # splitting on whitespace gives ["-", "--eval"] -- the flag is [1].
    action_flags: set[str] = set()
    in_args = False
    for line in action.splitlines():
        s = line.strip()
        if s.startswith("args:"):
            in_args = True
            continue
        if in_args and s and not s.startswith("-"):
            in_args = False
        if in_args and s.startswith("- --"):
            action_flags.add(s.split()[1])

    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    assert action_flags, "no flags found in action.yml -- did the format change?"
    # The entrypoint lists the accepted flags in one `|`-separated case arm.
    # Find that arm directly: the line containing at least three pipe-joined
    # `--flags`. Ship a bug in the parsing of our own release notes here and
    # a user setting a new input gets `unknown argument` and exit 2.
    case_arm = None
    for line in entrypoint.splitlines():
        tokens = [t for t in line.strip().split("|") if t.startswith("--")]
        if len(tokens) >= 3:
            case_arm = line.strip().rstrip(")")
            break
    assert case_arm, (
        "could not find the entrypoint's flag case arm -- did the entrypoint "
        "get reformatted so this check no longer applies?"
    )
    accepted_flags = {
        t.strip() for t in case_arm.split("|") if t.strip().startswith("--")
    }
    for flag in sorted(action_flags):
        assert flag in accepted_flags, (
            f"action.yml passes {flag} but entrypoint.sh does not accept it as a "
            f"named argument -- a user setting that input would get "
            f"`unknown argument` and exit 2. Accepted: {sorted(accepted_flags)}"
        )

    # And the reverse: every flag the entrypoint handles must exist on the CLI,
    # so a flag cannot be silently accepted by the shell and rejected by the
    # program it hands off to.
    import quant_regress.cli as cli

    parser_flags = {
        opt for action_ in cli.build_parser()._actions for opt in action_.option_strings
    }
    for flag in sorted(action_flags):
        assert flag in parser_flags, (
            f"action.yml passes {flag} but the CLI parser has no such option"
        )


def test_every_action_value_reaches_the_cli_with_its_value(tmp_path):
    """Every value action.yml passes must arrive at the CLI under its flag.

    This is the BEHAVIOURAL counterpart to the structural check above. The
    entrypoint can accept a flag, satisfy the source pattern, and still drop
    the value -- `--scorer) : ;;` matches the structural check while binding
    nothing. What a user relies on is that the value arrives, so that is
    asserted here against argv a recording stub actually saw.

    Pinned load-bearing: substituting `--scorer) : ;;` for the real binding
    fails this test while the structural check still passes.
    """
    rc, argv = _run_entrypoint(
        ["--eval", "ev.jsonl", "--model", "m", "--precisions", "int8",
         "--baseline-precision", "fp32", "--max-drop-points", "2",
         "--min-accuracy", "0.5", "--max-new-tokens", "16", "--scorer",
         "contains", "--output-format", "junit", "--report", "out/r.json",
         "--hf-token", "tok", "--cache-dir", "/tmp/cd"],
        tmp_path,
    )
    assert rc == 0, argv
    pairs = dict(zip(argv[2::2], argv[3::2]))
    expected = {
        "--eval": "ev.jsonl",
        "--model": "m",
        "--precisions": "int8",
        "--baseline-precision": "fp32",
        "--max-drop-points": "2",
        "--min-accuracy": "0.5",
        "--max-new-tokens": "16",
        "--scorer": "contains",
        "--output-format": "junit",
        "--report": "out/r.json",
        "--hf-token": "tok",
        "--cache-dir": "/tmp/cd",
    }
    for flag, want in expected.items():
        assert pairs.get(flag) == want, (
            f"{flag} did not reach the CLI with its value: wanted {want!r}, "
            f"the entrypoint built {pairs.get(flag)!r}. full argv: {argv}"
        )


# --------------------------------------------------------------------------
# The JUnit output the action publishes must be well-formed XML, or the CI
# pane that renders it shows nothing at all.
# --------------------------------------------------------------------------


def test_junit_output_is_valid_xml_with_a_named_failing_case(tmp_path, monkeypatch):
    import json
    import xml.etree.ElementTree as ET

    from quant_regress.cli import main

    rows = [{"id": str(i), "prompt": f"q{i}?", "expected": "yes"} for i in range(4)]
    ev = tmp_path / "ev.jsonl"
    ev.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    class _M:
        def __init__(self, precision):
            self.precision = precision

        def eval(self):
            return self

        def answer(self, prompt):
            return "yes" if self.precision == "fp32" else "no"

    monkeypatch.setattr(
        "quant_regress.cli.build_model_factory",
        lambda *a, **k: (lambda p: _M(p)),
    )
    junit = tmp_path / "junit.xml"
    rc = main(["--eval", str(ev), "--model", "m", "--output-format", "junit",
               "--junit-path", str(junit)])
    assert rc == 1, "a 100-point regression must exit 1"
    root = ET.parse(junit).getroot()
    assert root.tag == "testsuite"
    failed = [tc for tc in root.findall("testcase") if tc.find("failure") is not None]
    assert failed, "the regressing precision produced no <failure> element"
    # The failure must name the cases, or the red build cannot be triaged.
    assert any("misclassified" in (tc.find("failure").text or "").lower()
               or "correct" in (tc.find("failure").text or "").lower()
               for tc in failed), "the failure carries no case detail"
