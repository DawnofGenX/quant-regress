"""Shared contract for the independent lm-eval perplexity baseline.

The baseline exists to answer one question honestly: does a token-level metric
(perplexity, measured by lm-eval on a real corpus) notice the same INT8
collapse that the task-level accuracy curves catch?  A result JSON is only a
claim if it carries the precision, the measured dtype, the sample count and
the lm-eval version, so this module defines that schema and the single gate
used by both the regeneration script and its tests.

Nothing here imports lm-eval or torch: the pure functions stay cheap to test.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Every field a committed result must carry.  A missing field is a broken
#: claim, not a cosmetic omission, because it removes the ability to reproduce.
RESULT_SCHEMA = (
    "task",
    "precision",
    "model",
    "perplexity",
    "num_samples",
    "measured_dtype",
    "lm_eval_version",
    "timestamp",
)

#: A perplexity increase of at least this many points is a visible collapse in
#: the token-level metric.  The gate is deliberately strict and public.
DEFAULT_THRESHOLD = 2.0


def load_result(path: str | Path) -> dict:
    """Read and validate one result JSON; fail loudly on a broken artifact."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in RESULT_SCHEMA if key not in data]
    if missing:
        raise ValueError(f"{path}: missing required fields {missing}")
    return data


def perplexity_from(path: str | Path) -> float:
    """Return the perplexity recorded in one result JSON."""
    return float(load_result(path)["perplexity"])


def evaluate_delta(delta: float, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Apply the kill rule and say what the repo is allowed to claim.

    If INT8 perplexity rises by at least ``threshold`` points the token-level
    proxy DID detect the collapse, so the repo must drop or downgrade any claim
    that "perplexity/KL says the model is fine".  If the delta is smaller, the
    proxy missed the collapse and the existing claim stands -- with the new
    evidence stated explicitly.  Both outcomes are publishable; only a hidden
    one is dishonest.
    """
    passed = bool(delta < threshold)
    if passed:
        action = (
            "perplexity also misses this collapse: task-level accuracy is the "
            "necessary signal"
        )
    else:
        action = (
            "downgrade the claim that token-level proxies report the model fine: "
            "lm-eval perplexity detects this collapse"
        )
    return {
        "delta": float(delta),
        "threshold": float(threshold),
        "pass": passed,
        "action": action,
    }
