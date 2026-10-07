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
