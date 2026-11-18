"""Shape checks for the arith-tiered eval set."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from quant_regress.evalset import EvalSet

ROOT = Path(__file__).resolve().parent.parent
SET = ROOT / "evals" / "sets" / "arith-tiered.jsonl"

ID_RE = re.compile(r"^arith-[emh]\d{2}$")
INT_RE = re.compile(r"-?\d+")


def test_arith_tiered_shape():
    es = EvalSet.load(SET)
    assert len(es) == 60

    ids = [c.id for c in es.cases]
    assert len(set(ids)) == 60
    assert all(ID_RE.match(i) for i in ids)

    rows = [
        json.loads(line)
        for line in SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 60

    difficulty_counter = Counter(r["difficulty"] for r in rows)
    assert difficulty_counter == {"easy": 20, "medium": 20, "hard": 20}

    category_counter = Counter(r["category"] for r in rows)
    assert set(category_counter) == {
        "arith-add",
        "arith-sub",
        "arith-mul",
        "arith-div",
    }
    assert all(v >= 12 for v in category_counter.values())

    expected_values = [c.expected for c in es.cases]
    assert len(set(expected_values)) >= 15
    assert max(Counter(expected_values).values()) <= 6
    assert all(e.isdigit() for e in expected_values)

    assert all(c.prompt.endswith("Answer with the number only.") for c in es.cases)
    for case in es.cases:
        assert len(INT_RE.findall(case.prompt)) >= 2, case.id
