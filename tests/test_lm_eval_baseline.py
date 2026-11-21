"""Guard the independent lm-eval perplexity baseline.

This is a kill gate, not a decoration: if INT8 perplexity rises by at least
2.0 points, the repo is no longer allowed to claim that token-level proxies
report a quantized model as fine.  The tests check that both committed result
JSONs are complete and that the gate reports whichever outcome was measured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant_regress.lm_eval_baseline import (  # noqa: E402
    DEFAULT_THRESHOLD,
    RESULT_SCHEMA,
    evaluate_delta,
    load_result,
    perplexity_from,
)

FP32 = ROOT / "results" / "lm_eval_fp32.json"
INT8 = ROOT / "results" / "lm_eval_int8.json"


@pytest.mark.parametrize("path", [FP32, INT8])
def test_result_file_is_complete(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert not [key for key in RESULT_SCHEMA if key not in data], (
        f"{path.name} is missing a required field"
    )
    assert data["task"] == "wikitext"
    assert data["precision"] == ("fp32" if path == FP32 else "int8")
    assert data["perplexity"] > 0
    assert data["num_samples"] > 0
    assert data["lm_eval_version"].startswith("0.4.")


def test_each_arm_records_the_dtype_it_actually_measured():
    assert load_result(FP32)["measured_dtype"] == "torch.float32"
    assert load_result(INT8)["measured_dtype"] == "torch.qint8"


def test_delta_is_computed_from_the_committed_artifacts():
    delta = perplexity_from(INT8) - perplexity_from(FP32)
    verdict = evaluate_delta(delta, DEFAULT_THRESHOLD)
    assert verdict["pass"] is (delta < DEFAULT_THRESHOLD)
    assert verdict["delta"] == pytest.approx(delta)
    if verdict["pass"]:
        assert "misses" in verdict["action"]
    else:
        assert "downgrade" in verdict["action"]


def test_gate_boundary_is_strict():
    """Exactly 2.0 points is a detectable collapse, not a pass."""
    assert evaluate_delta(1.99)["pass"] is True
    assert evaluate_delta(2.0)["pass"] is False
    assert evaluate_delta(5.0)["pass"] is False
