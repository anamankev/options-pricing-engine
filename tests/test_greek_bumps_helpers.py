"""Helpers of experiments/greek_bumps.py (definitions registered in
docs/notes/phase_3_bump_study.md, section 3)."""

import math

import pytest
from experiments.greek_bumps import eta_grid, finite_difference, relative_error, slope

from pricing.contracts import OptionType

EPS = 2.220446049250313e-16
# A least-squares slope over exact powers: a few roundings of logs and dot products.
FIT_TOL = 64 * EPS
# One subtraction, abs and division: within 8 ulps.
METRIC_RTOL = 8 * EPS


def test_eta_grid_is_half_decades_from_1e_minus_1_to_1e_minus_12() -> None:
    grid = eta_grid({"eta_exponents": {"start": -1.0, "stop": -12.0, "step": -0.5}})
    assert len(grid) == 23
    assert grid[0] == pytest.approx(1e-1, rel=FIT_TOL)
    assert grid[-1] == pytest.approx(1e-12, rel=FIT_TOL)


def test_slope_recovers_an_exact_power_law() -> None:
    xs = [-3.0, -2.5, -2.0]
    assert slope(xs, [2 * x + 1 for x in xs]) == pytest.approx(2.0, abs=FIT_TOL)


def test_relative_error_definition() -> None:
    assert relative_error(1.1, 1.0) == pytest.approx(0.1, rel=METRIC_RTOL)


def test_theta_finite_difference_has_the_spec_sign() -> None:
    # SPEC 7: theta = -(V(T + h) - V(T - h)) / (2h); a long call decays, so theta < 0.
    market = {"S": 100.0, "K": 100.0, "T": 0.5, "r": 0.05, "q": 0.02, "sigma": 0.25}
    assert finite_difference("theta", OptionType.CALL, market, 1e-5, 1.0) < 0
    assert math.isfinite(finite_difference("gamma", OptionType.CALL, market, 1e-4, 1.0))
