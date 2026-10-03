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
