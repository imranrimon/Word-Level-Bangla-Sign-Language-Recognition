"""Smoke tests for tools/paired_bootstrap.py.

These exercise the bootstrap mechanics on synthetic per-seed Top-1 data so
the audit-fix-#6 significance machinery doesn't silently regress.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_paired_bootstrap_detects_strong_signal():
    from tools.paired_bootstrap import paired_bootstrap

    # A is consistently 5 pp better than B across 5 seeds.
    a = {0: 0.80, 1: 0.81, 2: 0.79, 3: 0.82, 4: 0.78}
    b = {0: 0.75, 1: 0.76, 2: 0.74, 3: 0.77, 4: 0.73}
    res = paired_bootstrap(a, b, n_resamples=2000, seed=0)
    assert abs(res["delta"] - 0.050) < 1e-9
    assert res["n_paired_seeds"] == 5
    # Very strong signal — p should be near zero.
    assert res["p_value"] < 0.05


def test_paired_bootstrap_no_signal_when_tied():
    from tools.paired_bootstrap import paired_bootstrap

    # A and B with overlapping per-seed scores; no significant difference.
    a = {0: 0.80, 1: 0.78, 2: 0.82, 3: 0.79, 4: 0.81}
    b = {0: 0.79, 1: 0.81, 2: 0.80, 3: 0.82, 4: 0.78}
    res = paired_bootstrap(a, b, n_resamples=2000, seed=0)
    # |Δ| should be small.
    assert abs(res["delta"]) < 0.02
    # p-value should be substantial (not significant at α=0.05).
    assert res["p_value"] > 0.05


def test_paired_bootstrap_handles_partial_seed_overlap():
    from tools.paired_bootstrap import paired_bootstrap

    # A has seeds {0, 1, 2, 3}; B has seeds {1, 2, 3, 4}. Overlap = {1, 2, 3}.
    a = {0: 0.70, 1: 0.80, 2: 0.81, 3: 0.79}
    b = {1: 0.75, 2: 0.76, 3: 0.74, 4: 0.90}
    res = paired_bootstrap(a, b, n_resamples=500, seed=0)
    assert res["n_paired_seeds"] == 3


def test_paired_bootstrap_too_few_seeds_returns_nan():
    from tools.paired_bootstrap import paired_bootstrap
    import math

    res = paired_bootstrap({0: 0.8}, {0: 0.7}, n_resamples=100)
    assert math.isnan(res["delta"])
    assert res["n_paired_seeds"] == 1
