"""Unit tests for the paired bootstrap confidence interval on the accuracy drop."""
from __future__ import annotations

import pytest

from quant_regress.stats import paired_bootstrap_drop_ci


def test_identical_arms_give_exact_zero_interval():
    base = [1, 0, 1, 0, 1, 0]
    cand = [1, 0, 1, 0, 1, 0]
    assert paired_bootstrap_drop_ci(base, cand) == (0.0, 0.0)


def test_total_collapse_gives_positive_low_bound():
    base = [1] * 10
    cand = [0] * 10
    lo, hi = paired_bootstrap_drop_ci(base, cand)
    assert lo > 0
    assert hi >= lo


def test_determinism_same_seed_equal_different_seeds_may_differ():
    base = [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    cand = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    a = paired_bootstrap_drop_ci(base, cand, seed=0)
    b = paired_bootstrap_drop_ci(base, cand, seed=0)
    assert a == b
    c = paired_bootstrap_drop_ci(base, cand, seed=1)
    d = paired_bootstrap_drop_ci(base, cand, seed=2)
    # Different seeds are permitted to differ; at minimum they must be valid intervals.
    assert isinstance(c, tuple) and isinstance(d, tuple)


def test_n1_width_zero_and_point_estimate_reported():
    assert paired_bootstrap_drop_ci([1], [0]) == (100.0, 100.0)
    assert paired_bootstrap_drop_ci([1], [1]) == (0.0, 0.0)
    assert paired_bootstrap_drop_ci([0], [1]) == (-100.0, -100.0)


def test_value_errors():
    with pytest.raises(ValueError):
        paired_bootstrap_drop_ci([1, 0], [1])
    with pytest.raises(ValueError):
        paired_bootstrap_drop_ci([], [])
    with pytest.raises(ValueError):
        paired_bootstrap_drop_ci([1, 0], [1, 0], resamples=0)
    with pytest.raises(ValueError):
        paired_bootstrap_drop_ci([2, 0], [1, 0])
    with pytest.raises(ValueError):
        paired_bootstrap_drop_ci([1, 0], [1, 0], alpha=0.0)
    with pytest.raises(ValueError):
        paired_bootstrap_drop_ci([1, 0], [1, 0], alpha=1.0)


def test_alpha_half_narrower_than_alpha_05():
    base = [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    cand = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]
    lo5, hi5 = paired_bootstrap_drop_ci(base, cand, alpha=0.05)
    loh, hih = paired_bootstrap_drop_ci(base, cand, alpha=0.5)
    assert (hih - loh) < (hi5 - lo5)
