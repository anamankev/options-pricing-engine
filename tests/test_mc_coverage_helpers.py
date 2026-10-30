"""Helpers of experiments/mc_coverage.py: slope fit and pilot sizing (definitions registered in
docs/notes/phase_2_mc_coverage.md, section 3)."""

import math

import pytest
from experiments.mc_coverage import fit_slope, pilot_size

EPS = 2.220446049250313e-16
# An exact power law fitted over 4 points: a few roundings of logs and dot products.
FIT_TOL = 64 * EPS


def test_fit_slope_recovers_an_exact_power_law() -> None:
    n = [100, 1000, 10000, 100000]
    slope, slope_se = fit_slope(n, [3.0 * k**-0.5 for k in n])
    assert slope == pytest.approx(-0.5, abs=FIT_TOL)
    assert slope_se <= FIT_TOL


def test_fit_slope_standard_error_is_positive_with_scatter() -> None:
    n = [100, 1000, 10000, 100000]
    slope, slope_se = fit_slope(
        n, [k**-0.5 * math.exp(e) for k, e in zip(n, (0.1, -0.1, 0.1, -0.1), strict=True)]
    )
    assert -0.6 < slope < -0.4
    assert slope_se > 0.0


def test_pilot_size_rule() -> None:
    config = {"pilot_minimum": 100, "pilot_divisor": 10}
    assert pilot_size(config, 100) == 100  # floor
    assert pilot_size(config, 100_000) == 10_000  # n // 10
