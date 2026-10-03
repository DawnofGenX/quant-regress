"""The gate must fail a build on a regression — and must not fail without one.

An earlier version of the self-test pointed the Action at a real tiny LM with a
fixture expecting "yes", then asserted the run FAILS. That assertion was
unsound: an untuned tiny LM answers essentially nothing with "yes", so it scored
0/N at BOTH precisions, produced no drop, and the gate correctly reported PASS.
CI would have failed on a correct gate.

So the fixture is model-specific, not a property of the Action. What IS a
property of the Action is its exit-code contract, and that is what these tests
pin — through the real ``cli.main`` entry point, with only the model factory
replaced (no production backdoor).
"""
from __future__ import annotations

import json

import pytest

from quant_regress.cli import main


class _Stable:
    def __init__(self, precision: str):
        self.precision = precision

    def eval(self):
        return self

    def answer(self, prompt: str) -> str:
        return "yes"


class _Regressing:
    def __init__(self, precision: str):
        self.precision = precision

    def eval(self):
        return self

    def answer(self, prompt: str) -> str:
        return "yes" if self.precision == "fp32" else "no"


def _fixture(tmp_path, name, n=10):
    rows = [
        {"id": f"{name}-{i:03d}", "prompt": f"Case {i}: is the claim true?",
         "expected": "yes"}
        for i in range(1, n + 1)
    ]
    p = tmp_path / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _factory_for(cls):
    return lambda *a, **kw: (lambda precision: cls(precision))


def test_gate_fails_on_a_real_regression(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("quant_regress.cli.build_model_factory", _factory_for(_Regressing))
    report = tmp_path / "report.json"
    rc = main([
        "--eval", _fixture(tmp_path, "regress"),
        "--model", "selftest/fake-regressing",
        "--max-drop-points", "2",
        "--report", str(report),
    ])
    assert rc == 1, "a 100-point regression must exit 1"
    data = json.loads(report.read_text())
    assert data["verdict"] == "fail"
    assert data["worst_drop_points"] > 2
    assert data["candidates"][0]["correct"] == 0
    assert data["baseline"]["correct"] == 10


def test_gate_passes_when_accuracy_holds(tmp_path, monkeypatch):
    """A gate that always fails is as useless as one that never does."""
    monkeypatch.setattr("quant_regress.cli.build_model_factory", _factory_for(_Stable))
    rc = main([
        "--eval", _fixture(tmp_path, "stable"),
        "--model", "selftest/fake-stable",
        "--max-drop-points", "2",
    ])
    assert rc == 0


def test_tiny_drop_within_tolerance_still_passes(tmp_path, monkeypatch):
    """Tolerance must actually tolerate: 0 points of drop exits 0."""
    monkeypatch.setattr("quant_regress.cli.build_model_factory", _factory_for(_Stable))
    rc = main([
        "--eval", _fixture(tmp_path, "tol"),
        "--model", "selftest/fake-stable",
        "--max-drop-points", "0",
    ])
    assert rc == 0


def test_missing_model_is_a_usage_error_not_a_pass(tmp_path, monkeypatch):
    """The most dangerous failure mode: a misconfigured run reporting success."""
    monkeypatch.setattr("quant_regress.cli.build_model_factory", _factory_for(_Stable))
    rc = main(["--eval", _fixture(tmp_path, "cfg")])
    assert rc == 2, "missing --model must exit 2, never 0"


def test_unreadable_eval_set_is_a_usage_error(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_regress.cli.build_model_factory", _factory_for(_Stable))
    rc = main(["--eval", str(tmp_path / "nope.jsonl"), "--model", "x"])
    assert rc == 2


def test_report_is_written_even_on_the_failing_path(tmp_path, monkeypatch):
    """A regression must still leave evidence behind, or there is nothing to triage."""
    monkeypatch.setattr("quant_regress.cli.build_model_factory", _factory_for(_Regressing))
    report = tmp_path / "r.json"
    main([
        "--eval", _fixture(tmp_path, "ev"),
        "--model", "selftest/fake-regressing",
        "--max-drop-points", "2",
        "--report", str(report),
    ])
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["verdict"] == "fail"
    # Per-case evidence must be present so a CI failure is actionable.
    assert "candidates" in data and data["candidates"][0]["total"] == 10