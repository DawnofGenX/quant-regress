"""The CLI must exit non-zero on regression so CI actually fails."""
from __future__ import annotations

import json

import pytest

from quant_regress.cli import main


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
