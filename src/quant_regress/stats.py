"""Paired bootstrap confidence interval for the accuracy drop.

Why paired: both arms answer the *same* cases in the same order, so the
per-case outcomes are dependent. Resampling the two arms independently
(unpaired) would treat baseline and candidate as if they came from
different eval sets and would overstate the interval. We draw one
bootstrap sample of case indices per replicate and score both arms on it.

Why bootstrap: the metric is a difference of two proportions on a small n,
and the sampling distribution is not normal near 0% or 100% (where the
variance of a proportion collapses). A normal / Wald interval would put
mass outside [0, 100] and misstate coverage; the percentile bootstrap
makes no distributional assumption beyond the observed cases.
"""
from __future__ import annotations

import random
from typing import Sequence


def paired_bootstrap_drop_ci(
    baseline_hits: Sequence[int],
    candidate_hits: Sequence[int],
    *,
    resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired percentile bootstrap CI for the drop in accuracy (points).

    ``hits`` are per-case 0/1 correctness vectors aligned by case index.
    Each replicate recomputes ``(mean(baseline) - mean(candidate)) * 100``,
    i.e. the drop in points (positive means the candidate got worse). The
    returned interval is the percentile interval at ``alpha/2`` and
    ``1 - alpha/2`` of those replicate drops.
    """
    if len(baseline_hits) != len(candidate_hits):
        raise ValueError(
            "baseline_hits and candidate_hits must have the same length "
            f"(got {len(baseline_hits)} and {len(candidate_hits)})"
        )
    if len(baseline_hits) == 0:
        raise ValueError("baseline_hits and candidate_hits must be non-empty")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    for hits in (baseline_hits, candidate_hits):
        for h in hits:
            if h not in (0, 1):
                raise ValueError(
                    f"hits must be 0 or 1, got {h!r}"
                )

    n = len(baseline_hits)
    base = [int(x) for x in baseline_hits]
    cand = [int(x) for x in candidate_hits]

    rng = random.Random(seed)
    drops: list[float] = []
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        mb = sum(base[i] for i in indices) / n
        mc = sum(cand[i] for i in indices) / n
        drops.append((mb - mc) * 100.0)

    drops.sort()

    def percentile(p: float) -> float:
        if resamples == 1:
            return drops[0]
        h = (resamples - 1) * p
        lo = int(h // 1)
        if lo >= resamples - 1:
            lo = resamples - 2
        hi = lo + 1
        frac = h - lo
        return drops[lo] * (1.0 - frac) + drops[hi] * frac

    low = percentile(alpha / 2.0)
    high = percentile(1.0 - alpha / 2.0)
    return (low, high)
