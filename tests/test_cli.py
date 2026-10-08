"""The CLI must exit non-zero on regression so CI actually fails."""
from __future__ import annotations

import json

import pytest

from quant_regress.cli import main
from quant_regress.harness import QuantHarness


class _Fake:
    """Deterministic stand-in: answers correctly at fp32, wrongly at int8.

    Mirrors the real failure mode this tool exists to catch — a quantized model
    that still runs and still produces output, just worse output.
    """

    def __init__(self, precision: str, correct: bool = True):
        self.precision = precision
        self.correct = correct

    def eval(self):
        return self

    def answer(self, prompt: str) -> str:
        return "yes" if self.correct else "no"


def _eval(tmp_path, n=6):
    rows = [{"id": str(i), "prompt": f"q{i}?", "expected": "yes"} for i in range(n)]
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def test_exit_1_on_regression(tmp_path, monkeypatch):
    def factory(*a, **kw):
        return lambda precision: _Fake(precision, correct=(precision == "fp32"))

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--max-drop-points", "5"])
    assert rc == 1


def test_exit_0_when_within_tolerance(tmp_path, monkeypatch):
    def factory(*a, **kw):
        return lambda precision: _Fake(precision, correct=True)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--max-drop-points", "5"])
    assert rc == 0


def test_writes_json_report(tmp_path, monkeypatch):
    out = tmp_path / "report.json"

    def factory(*a, **kw):
        return lambda precision: _Fake(precision, correct=(precision == "fp32"))

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    main(["--eval", _eval(tmp_path), "--model", "fake-model",
          "--max-drop-points", "5", "--report", str(out)])
    data = json.loads(out.read_text())
    assert "baseline" in data and "candidates" in data and "verdict" in data


def test_report_is_written_even_when_the_path_is_a_directory(
    tmp_path, monkeypatch, capsys
):
    """An unwritable report path must not traceback after the work is done.

    The measurement completed; if `Path.write_text` raises an unhandled
    OSError the CLI dies with a traceback instead of reporting the verdict it
    already has. Exit 2 (could not record), never 1 (regressed).
    """
    blocker = tmp_path / "report.json"
    blocker.mkdir()  # a directory cannot be written as a file

    def factory(*a, **kw):
        return lambda precision: _Fake(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--report", str(blocker)])
    assert rc == 2, "an unwritable --report must exit 2, never 1"
    assert "report" in capsys.readouterr().err.lower()


def test_report_parent_directory_is_created(tmp_path, monkeypatch):
    """`--report out/nested/r.json` must not fail because out/ does not exist."""
    out = tmp_path / "out" / "nested" / "r.json"

    def factory(*a, **kw):
        return lambda precision: _Fake(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--report", str(out)])
    assert rc == 0
    assert out.exists(), "the report's parent directories were not created"


def test_missing_model_argument_exits_2(tmp_path, monkeypatch):
    def factory(*a, **kw):
        return lambda precision: _Fake(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path)])
    assert rc == 2


# --------------------------------------------------------------------------
# The exit code IS the contract with CI. A configuration mistake must never
# exit 1, because 1 means "accuracy regressed" -- a build that fails for the
# wrong reason teaches people to ignore it.
# --------------------------------------------------------------------------


def test_directory_as_eval_exits_2_not_1(tmp_path, monkeypatch):
    """A directory passes exists(); read_text() raised IsADirectoryError uncaught,
    which escaped main() and surfaced as exit 1."""
    d = tmp_path / "evals"
    d.mkdir()
    (d / "x.jsonl").write_text("{}\n", encoding="utf-8")

    def factory(*a, **kw):
        return lambda precision: _Fake(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", str(d), "--model", "fake-model"])
    assert rc == 2, "a directory must be a usage error, never 'accuracy regressed'"


def test_nan_threshold_exits_2_not_0(tmp_path, monkeypatch, capsys):
    """NaN threshold previously reported PASS (exit 0) for a real regression."""
    def factory(*a, **kw):
        return lambda precision: _Fake(precision, correct=(precision == "fp32"))

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--max-drop-points", "nan"])
    assert rc == 2
    assert "NaN" in capsys.readouterr().err


def test_negative_threshold_exits_2(tmp_path, monkeypatch, capsys):
    def factory(*a, **kw):
        return lambda precision: _Fake(precision, correct=(precision == "fp32"))

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--max-drop-points", "-1"])
    assert rc == 2
    assert ">= 0" in capsys.readouterr().err


def test_empty_precisions_exits_2_not_0(tmp_path, monkeypatch, capsys):
    """`--precisions ""` produced no candidates, so PASS with nothing compared."""
    def factory(*a, **kw):
        return lambda precision: _Fake(precision, correct=(precision == "fp32"))

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--precisions", ""])
    assert rc == 2
    assert "no candidate precisions" in capsys.readouterr().err


def test_malformed_eval_json_exits_2(tmp_path, monkeypatch):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")

    def factory(*a, **kw):
        return lambda precision: _Fake(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", str(p), "--model", "fake-model"])
    assert rc == 2


def test_missing_eval_file_exits_2(tmp_path, monkeypatch):
    def factory(*a, **kw):
        return lambda precision: _Fake(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", str(tmp_path / "nope.jsonl"), "--model", "fake-model"])
    assert rc == 2


# --------------------------------------------------------------------------
# --min-accuracy: the gate's exit code for a model that is simply wrong.
# This is a QUALITY failure, so exit 1 -- the tool measured correctly and the
# answer is "no". Exit 2 stays reserved for "could not measure".
# --------------------------------------------------------------------------


class _AlwaysWrong(_Fake):
    def answer(self, prompt: str) -> str:
        return "no"


def test_baseline_floor_breach_exits_1_not_0(tmp_path, monkeypatch):
    """0/N at both precisions is a 0.00-point drop; with a floor it must FAIL."""
    def factory(*a, **kw):
        return lambda precision: _AlwaysWrong(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--min-accuracy", "0.5"])
    assert rc == 1, "a model below the floor must fail the build"


def test_baseline_floor_breach_exits_1_even_with_generous_drop_budget(
    tmp_path, monkeypatch
):
    """The floor must be independent of max_drop_points."""
    def factory(*a, **kw):
        return lambda precision: _AlwaysWrong(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--max-drop-points", "99", "--min-accuracy", "0.5"])
    assert rc == 1


def test_floor_breach_is_explained_on_stderr_or_stdout(tmp_path, monkeypatch, capsys):
    """A red build must say WHY, in words a reader can act on."""
    def factory(*a, **kw):
        return lambda precision: _AlwaysWrong(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    main(["--eval", _eval(tmp_path), "--model", "fake-model",
          "--min-accuracy", "0.5"])
    out = capsys.readouterr()
    combined = out.out + out.err
    assert "min-accuracy" in combined
    assert "below" in combined.lower()


def test_no_floor_flag_keeps_the_existing_pass_behaviour(tmp_path, monkeypatch):
    """Opt-in: without --min-accuracy nothing changes, so no user is broken."""
    def factory(*a, **kw):
        return lambda precision: _AlwaysWrong(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model"])
    assert rc == 0, "default behaviour must remain delta-only"


def test_invalid_min_accuracy_exits_2(tmp_path, monkeypatch, capsys):
    """A percentage where a fraction belongs is a usage error, not a FAIL."""
    def factory(*a, **kw):
        return lambda precision: _Fake(precision, correct=(precision == "fp32"))

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--min-accuracy", "50"])
    assert rc == 2
    assert "FRACTION" in capsys.readouterr().err


def test_floor_breach_is_recorded_in_the_json_report(tmp_path, monkeypatch):
    out = tmp_path / "report.json"

    def factory(*a, **kw):
        return lambda precision: _AlwaysWrong(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    main(["--eval", _eval(tmp_path), "--model", "fake-model",
          "--min-accuracy", "0.5", "--report", str(out)])
    data = json.loads(out.read_text())
    assert data["min_accuracy"] == 0.5
    assert data["baseline_below_floor"] is True
    assert data["verdict"] == "fail"


# --------------------------------------------------------------------------
# Phase 2 #7: a broken model must exit 2, not traceback.
#
# A model whose tokenizer returns junk, or whose generate() raises, currently
# escapes main() as a raw traceback. On CI that surfaces as exit 1 -- the code
# that means "accuracy regressed" -- so a broken model impersonates a real
# regression and teaches people to ignore the one exit code that matters.
# --------------------------------------------------------------------------


class _BrokenTokenizer:
    def __call__(self, text, return_tensors=None):
        raise TypeError("tokenizer exploded")


class _BrokenGenerate:
    """Shape 1 (generate + tokenizer), where generate() raises."""

    tokenizer = _BrokenTokenizer()

    def eval(self):
        return self

    def generate(self, **kwargs):
        raise AttributeError("no .logits on this thing")


def test_broken_model_exits_2_not_a_traceback(tmp_path, monkeypatch, capsys):
    """A model that cannot produce a prediction is a config error, not a FAIL.

    Without the wrapper this test fails with TypeError escaping main().
    """
    def factory(*a, **kw):
        return lambda precision: _BrokenGenerate()

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--precisions", "int8"])
    assert rc == 2, "a broken model must exit 2, never 0 or 1"
    err = capsys.readouterr().err
    assert "error" in err.lower(), f"expected an actionable message, got {err!r}"


class _BrokenLogits:
    """Shape 3 (logits): returns a tensor without .logits at all."""

    def __init__(self, precision):
        self.precision = precision
        self.tokenizer = _BrokenTokenizer()

    def eval(self):
        return self

    def __call__(self, **kw):
        raise AttributeError("model has no .logits")


def test_broken_logits_model_exits_2(tmp_path, monkeypatch, capsys):
    def factory(*a, **kw):
        return lambda precision: _BrokenLogits(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--precisions", "int8"])
    assert rc == 2
    assert "error" in capsys.readouterr().err.lower()


class _ModelWithoutAnswer:
    """The old hard failure: no .answer(), no generate, no id2label."""

    def __init__(self, precision):
        self.precision = precision

    def eval(self):
        return self


def test_model_that_cannot_produce_text_exits_2_not_a_crash(tmp_path, monkeypatch, capsys):
    """Already raised RuntimeError; now it must also exit 2 cleanly."""
    def factory(*a, **kw):
        return lambda precision: _ModelWithoutAnswer(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "fake-model",
               "--precisions", "int8"])
    assert rc == 2
    assert "error" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------
# Phase 3 -- CLI surface. The library already accepts scorer, system_prompt
# and labels; the CLI dropped them, so the capabilities were unreachable
# from a workflow file. Progress output goes to stderr (stdout stays
# parseable), and misclassified ids must reach the report so a red build can
# be triaged.
# --------------------------------------------------------------------------


class _ContainsOnly(_Fake):
    """Right answer only under a substring scorer.

    Note the eval must contain at least one case this model gets right at
    BOTH precisions, or both arms score 0% and the 0.00-point drop passes for
    the wrong reason -- the same trap the --min-accuracy floor exists to close.
    """

    def answer(self, prompt: str) -> str:
        return "yes" if prompt.startswith("q0") else "the answer is YES, clearly"


def test_scorer_flag_selects_contains_matching(tmp_path, monkeypatch):
    """With --scorer contains, 'the answer is YES' must match 'yes'.

    Asserts on the MEASURED accuracy rather than the exit code: both arms of
    this fake answer identically, so the drop is 0.00 points and the gate
    correctly PASSes at either scorer -- the exit code cannot distinguish
    them. Only the measured number can.
    """
    import contextlib
    import io

    def factory(*a, **kw):
        return lambda precision: _ContainsOnly(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)

    def accuracy_for(scorer_args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["--eval", _eval(tmp_path), "--model", "m", *scorer_args])
        out = buf.getvalue()
        # First data row is the baseline.
        import re
        m = re.search(r"\|\s*(\d+\.\d)%", out)
        assert m, f"no accuracy in output: {out!r}"
        return float(m.group(1))

    exact_acc = accuracy_for([])
    contains_acc = accuracy_for(["--scorer", "contains"])
    assert exact_acc < contains_acc, (
        f"the scorer flag did not change any measurement: "
        f"exact={exact_acc}% contains={contains_acc}%"
    )
    assert contains_acc == 100.0, (
        f"the contains scorer should match every case, got {contains_acc}%"
    )



def test_unknown_scorer_exits_2(tmp_path, monkeypatch):
    """argparse rejects an unknown scorer before anything expensive runs.

    SystemExit(2) is argparse's usage error, which maps to the same exit code
    the CLI returns for bad configuration -- so the contract holds either way.
    This pins that a typo'd --scorer cannot silently fall back to a default.
    """
    def factory(*a, **kw):
        return lambda precision: _Fake(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    with pytest.raises(SystemExit) as exc:
        main(["--eval", _eval(tmp_path), "--model", "m", "--scorer", "regex"])
    assert exc.value.code == 2


class _T:
    """Minimal tensor stand-in: len, slicing, .to(), .shape, numel()."""

    def __init__(self, ids):
        self._ids = list(ids)

    def __getitem__(self, k):
        return _T(self._ids[k]) if isinstance(k, slice) else self._ids[k]

    def __len__(self):
        return len(self._ids)

    def to(self, *a, **kw):
        return self

    @property
    def shape(self):
        return (1, len(self._ids))

    def numel(self):
        return len(self._ids)


def test_labels_flag_reaches_the_logits_path(tmp_path, monkeypatch):
    """--labels must be forwarded to the harness, not silently dropped."""
    class _Tok:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": _T([0] * 4)}

    class _Cfg:
        id2label = None

    class _L:
        logits = [[0.9, 0.1]]

    class _LModel:
        def __init__(self, precision):
            self.precision = precision
            self.tokenizer = _Tok()
            self.config = _Cfg()

        def eval(self):
            return self

        def __call__(self, **kw):
            return _L()

    captured = {}

    real_init = QuantHarness.__init__

    def spy_init(self, *a, **kw):
        real_init(self, *a, **kw)
        captured["labels"] = list(self.labels)

    monkeypatch.setattr("quant_regress.cli.QuantHarness.__init__", spy_init)
    monkeypatch.setattr(
        "quant_regress.cli.build_model_factory",
        lambda *a, **kw: (lambda precision: _LModel(precision)),
    )
    main(["--eval", _eval(tmp_path), "--model", "m", "--labels", "no, yes"])
    assert captured.get("labels") == ["no", "yes"], captured


def test_progress_output_goes_to_stderr_not_stdout(tmp_path, monkeypatch, capsys):
    """Stdout must stay parseable; progress is a human affordance."""
    def factory(*a, **kw):
        return lambda precision: _Fake(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    main(["--eval", _eval(tmp_path, n=30), "--model", "m"])
    captured = capsys.readouterr()
    assert captured.out.count("precision") <= 1, (
        "progress output polluted stdout -- a workflow parsing the table "
        "would break"
    )
    assert "30" in captured.err or "case" in captured.err.lower(), (
        f"no progress on stderr: {captured.err!r}"
    )


def test_report_includes_misclassified_ids(tmp_path, monkeypatch):
    """A red build must name the cases that failed, or it cannot be triaged."""
    out = tmp_path / "r.json"

    def factory(*a, **kw):
        return lambda precision: _Fake(precision, correct=(precision == "fp32"))

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    main(["--eval", _eval(tmp_path, n=4), "--model", "m", "--report", str(out)])
    data = json.loads(out.read_text())
    cand = data["candidates"][0]
    assert "misclassified" in cand, f"report has no misclassified ids: {cand.keys()}"
    assert sorted(cand["misclassified"]) == ["0", "1", "2", "3"]


def test_output_format_junit_writes_valid_xml(tmp_path, monkeypatch):
    out = tmp_path / "junit.xml"

    def factory(*a, **kw):
        return lambda precision: _Fake(precision, correct=(precision == "fp32"))

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    rc = main(["--eval", _eval(tmp_path), "--model", "m",
               "--output-format", "junit", "--junit-path", str(out)])
    assert rc == 1
    assert out.exists(), "no junit file was written"
    import xml.etree.ElementTree as ET
    root = ET.parse(out).getroot()
    assert root.tag == "testsuite"
    cases = root.findall("testcase")
    assert len(cases) >= 2, f"expected one testcase per precision, got {len(cases)}"
    # The failing arm must carry a failure element.
    assert any(tc.find("failure") is not None for tc in cases), (
        "the regressing precision produced no <failure> element"
    )


def test_unknown_output_format_exits_2(tmp_path, monkeypatch):
    """Same contract as --scorer: argparse rejects it, exit code 2."""
    def factory(*a, **kw):
        return lambda precision: _Fake(precision)

    monkeypatch.setattr("quant_regress.cli.build_model_factory", factory)
    with pytest.raises(SystemExit) as exc:
        main(["--eval", _eval(tmp_path), "--model", "m",
              "--output-format", "yaml"])
    assert exc.value.code == 2, "an unknown format must not silently succeed"
