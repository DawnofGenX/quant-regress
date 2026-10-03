"""The eval set must load deterministically and reject malformed input."""
from __future__ import annotations

import json

import pytest

from quant_regress.evalset import EvalSet, EvalSetError


def _write(tmp_path, rows):
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_loads_rows_in_file_order(tmp_path):
    rows = [
        {"id": "a", "prompt": "x?", "expected": "yes"},
        {"id": "b", "prompt": "y?", "expected": "no"},
    ]
    es = EvalSet.load(_write(tmp_path, rows))
    assert [c.id for c in es.cases] == ["a", "b"]
    assert len(es) == 2


def test_normalises_expected_for_exact_comparison(tmp_path):
    rows = [{"id": "a", "prompt": "x?", "expected": "  YES  "}]
    es = EvalSet.load(_write(tmp_path, rows))
    assert es.cases[0].expected == "yes"


def test_rejects_row_missing_prompt(tmp_path):
    rows = [{"id": "a", "expected": "yes"}]
    with pytest.raises(EvalSetError, match="prompt"):
        EvalSet.load(_write(tmp_path, rows))


def test_rejects_duplicate_ids(tmp_path):
    rows = [
        {"id": "a", "prompt": "x?", "expected": "yes"},
        {"id": "a", "prompt": "y?", "expected": "no"},
    ]
    with pytest.raises(EvalSetError, match="duplicate"):
        EvalSet.load(_write(tmp_path, rows))


def test_rejects_empty_file(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(EvalSetError, match="empty"):
        EvalSet.load(p)
