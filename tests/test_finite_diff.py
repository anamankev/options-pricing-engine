"""Finite-difference helpers (SPEC 7): truncation error matches the Taylor model exactly on
polynomials, steps are the ones actually taken, and invalid steps raise."""

import math

import numpy as np
import pytest

from pricing.finite_diff import central_first, central_second, relative_step

EPS = float(np.finfo(np.float64).eps)

# Rounding bound for these tests: about eps |f| / h for f' and 4 eps |f| / h^2 for f''. With
# |f| <= 16 and h = 1e-2 those are ~4e-13 and ~1.4e-10; the tolerances leave ~50x margin.
FIRST_ROUNDING_TOL = 2e-11
SECOND_ROUNDING_TOL = 1e-8


def test_relative_step_scales_with_x_and_floors_at_x_scale() -> None:
    assert relative_step(100.0, eta=1e-5, x_scale=1.0) == 1e-5 * 100.0
    assert relative_step(-100.0, eta=1e-5, x_scale=1.0) == 1e-5 * 100.0
    assert relative_step(0.0, eta=1e-5, x_scale=1.0) == 1e-5


@pytest.mark.parametrize(("eta", "x_scale"), [(0.0, 1.0), (-1e-5, 1.0), (1e-5, 0.0)])
def test_relative_step_rejects_non_positive_settings(eta: float, x_scale: float) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        relative_step(1.0, eta=eta, x_scale=x_scale)


def test_first_difference_truncation_error_is_h_squared_on_a_cubic() -> None:
    # f = x^3: ((x+h)^3 - (x-h)^3) / (2h) = 3x^2 + h^2 exactly, i.e. error h^2 f'''/6 with f''' = 6.
    x, h = 2.0, 1e-2
    estimate = central_first(lambda t: t**3, x, h)
    assert abs(estimate - (3 * x**2 + h**2)) <= FIRST_ROUNDING_TOL


def test_second_difference_truncation_error_is_two_h_squared_on_a_quartic() -> None:
    # f = x^4: ((x+h)^4 - 2x^4 + (x-h)^4) / h^2 = 12x^2 + 2h^2, i.e. error h^2 f''''/12, f'''' = 24.
    x, h = 2.0, 1e-2
    estimate = central_second(lambda t: t**4, x, h)
    assert abs(estimate - (12 * x**2 + 2 * h**2)) <= SECOND_ROUNDING_TOL


def test_near_optimal_step_on_exp_reaches_predicted_accuracy() -> None:
    # SPEC 7: optimal eta ~ eps^(1/3) for f'. Total error ~ h^2 |f'''| / 6 + eps |f| / h, which at
    # x = 0.3 and h = eps^(1/3) is about 1e-10 relative; 1e-9 allows a factor of 10.
    x = 0.3
    h = relative_step(x, eta=EPS ** (1 / 3), x_scale=1.0)
    assert abs(central_first(math.exp, x, h) / math.exp(x) - 1.0) <= 1e-9


def test_divides_by_the_step_actually_taken() -> None:
    # 0.1 + 1e-7 is rounded, so (x + h) - (x - h) != 2h here; dividing by 2h would give a slope
    # of 1.0000000000287557 for f(x) = x. With the step actually taken it is exactly 1.
    x, h = 0.1, 1e-7
    assert (x + h) - (x - h) != 2 * h  # precondition: this case really is inexact
    assert central_first(lambda t: t, x, h) == 1.0


@pytest.mark.parametrize("h", [0.0, -1e-3, math.nan])
def test_non_positive_or_nan_step_raises(h: float) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        central_first(math.exp, 1.0, h)


def test_step_below_resolution_of_x_raises() -> None:
    # spacing(1e8) is about 1.5e-8, so 1e8 + 1e-9 == 1e8: no step was taken at all.
    with pytest.raises(ValueError, match="below the floating-point resolution"):
        central_second(math.exp, 1e8, 1e-9)
