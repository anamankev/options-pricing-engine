"""Metric definitions used by experiments/quantlib_reconciliation.py (no QuantLib needed)."""

import math

import pytest
from experiments.quantlib_reconciliation import natural_scale, relative_error

EPS = 2.220446049250313e-16
# One subtraction, abs and division, or one product and sqrt: within 8 ulps.
METRIC_RTOL = 8 * EPS


def test_relative_error_definition() -> None:
    # Definitions registered in docs/notes/phase_1_quantlib_reconciliation.md section 3.
    assert relative_error(1.1, 1.0) == pytest.approx(0.1, rel=METRIC_RTOL)
    assert relative_error(0.0, 0.0) == 0.0  # both exactly zero: agreement
    assert relative_error(1e-300, 0.0) == math.inf  # only the reference is zero
    assert relative_error(-2.0, -1.0) == 1.0


def test_natural_scales() -> None:
    case = {"S": 100.0, "K": 90.0, "days": 365}  # T = 1
    expected = {"price": 100.0, "delta": 1.0, "gamma": 0.01, "vega": 100.0, "rho": 90.0}
    for name, scale in expected.items():
        assert natural_scale(name, case) == pytest.approx(scale, rel=METRIC_RTOL)
