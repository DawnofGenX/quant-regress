"""Eval set: binary sentiment classification (non-arithmetic task family)."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from quant_regress.evalset import EvalSet

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "evals" / "sets" / "classify.jsonl"


def test_classify_eval_set_shape_balance_and_aliases():
    lines = [ln for ln in PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 60, f"expected 60 rows, got {len(lines)}"

    rows = [json.loads(ln) for ln in lines]

    ids = [r["id"] for r in rows]
    assert len(set(ids)) == 60, "ids must be unique"
    assert all(re.fullmatch(r"cls-\d{3}", i) for i in ids), "ids must match cls-NNN"

    assert Counter(r["expected"] for r in rows) == {"positive": 30, "negative": 30}

    assert all(r["category"] == "sentiment" for r in rows)

    diff_counts = Counter(r["difficulty"] for r in rows)
    for d in ("easy", "medium", "hard"):
        assert diff_counts.get(d, 0) >= 10, f"need >=10 {d}, got {diff_counts.get(d, 0)}"

    for r in rows:
        assert isinstance(r["accept"], list) and r["accept"], "accept must be non-empty"
        first = "pos" if r["expected"] == "positive" else "neg"
        assert r["accept"][0] == first, f"bad first alias for {r['id']}"

    assert sum(1 for r in rows if len(r["accept"]) >= 2) >= 20

    prompts = [r["prompt"] for r in rows]
    assert len(set(prompts)) == 60, "prompts must be unique"
    for r in rows:
        assert "positive" in r["prompt"]
        assert "negative" in r["prompt"]
        assert r["prompt"].endswith("Answer with one word.")

    es = EvalSet.load(PATH)
    assert len(es) == 60
    assert Counter(c.expected for c in es.cases) == {"positive": 30, "negative": 30}
