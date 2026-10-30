"""Input validation, option types and unit helpers (SPEC 2.2, 2.3, 2.4)."""

import math
from typing import Any

import numpy as np
import pytest

from pricing.contracts import (
    MAX_SIGMA_WITHOUT_OVERRIDE,
    OptionType,
    option_sign,
    per_calendar_day,
    per_percentage_point,
    per_vol_point,
    validate_market_inputs,
)
from pricing.errors import PricingInputError

# Hull, Options, Futures, and Other Derivatives, Example 15.6 (SPEC 9.1 R1): a valid input set.
VALID: dict[str, Any] = {"S": 42.0, "K": 40.0, "T": 0.5, "r": 0.10, "q": 0.0, "sigma": 0.20}


def _validate(**overrides: Any) -> Any:
    kwargs = {**VALID, **overrides}
    allow = kwargs.pop("allow_extreme_vol", False)
    return validate_market_inputs(**kwargs, allow_extreme_vol=allow)


def test_option_sign_is_theta_cp() -> None:
    assert option_sign(OptionType.CALL) == 1
    assert option_sign(OptionType.PUT) == -1


@pytest.mark.parametrize("bad", ["call", 1, None])
def test_option_sign_rejects_non_option_types(bad: object) -> None:
    with pytest.raises(PricingInputError, match="option_type"):
        option_sign(bad)


def test_scalar_inputs_are_flagged_scalar() -> None:
    m = _validate()
    assert m.all_scalar
    assert m.S.shape == ()
    assert m.S.dtype == np.float64


def test_array_input_broadcasts_and_is_not_scalar() -> None:
    m = _validate(S=np.array([40.0, 42.0, 44.0]), sigma=np.array([[0.1], [0.2]]))
    assert not m.all_scalar
    assert m.S.shape == m.sigma.shape == m.T.shape == (2, 3)


def test_integer_inputs_are_accepted_as_float64() -> None:
    m = _validate(S=42, K=40)
    assert m.S.dtype == np.float64
    assert float(m.S) == 42.0


@pytest.mark.parametrize("name", ["S", "K", "T", "r", "q", "sigma"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_inputs_raise_naming_the_parameter(name: str, bad: float) -> None:
    with pytest.raises(PricingInputError, match=rf"^{name} must be finite"):
        _validate(**{name: bad})


@pytest.mark.parametrize(
    ("name", "bad"),
    [("S", 0.0), ("S", -1.0), ("K", 0.0), ("K", -1.0), ("T", -1e-12), ("sigma", -1e-12)],
)
def test_out_of_domain_inputs_raise_naming_the_parameter(name: str, bad: float) -> None:
    with pytest.raises(PricingInputError, match=rf"^{name} must be"):
        _validate(**{name: bad})


def test_boundary_values_t_zero_and_sigma_zero_are_valid() -> None:
    # SPEC 3.4 defines prices at T = 0 and sigma = 0, so validation must let them through.
    _validate(T=0.0)
    _validate(sigma=0.0)


@pytest.mark.parametrize(("r", "q"), [(-0.01, 0.0), (0.0, -0.02), (-0.5, -0.5)])
def test_negative_rates_and_dividend_yields_are_valid(r: float, q: float) -> None:
    _validate(r=r, q=q)  # SPEC 2.2: any finite r and q


def test_sigma_above_cap_needs_explicit_override() -> None:
    _validate(sigma=MAX_SIGMA_WITHOUT_OVERRIDE)  # the cap itself is allowed
    with pytest.raises(PricingInputError, match="allow_extreme_vol"):
        _validate(sigma=20.0)  # the classic mistake: 20 for 20%
    assert float(_validate(sigma=20.0, allow_extreme_vol=True).sigma) == 20.0


def test_array_error_reports_value_and_index() -> None:
    with pytest.raises(PricingInputError, match=r"got -1\.0 at index \(2,\)"):
        _validate(K=np.array([40.0, 41.0, -1.0]))


@pytest.mark.parametrize("bad", ["42", 42 + 0j, True, np.array(["a"]), None])
def test_non_real_inputs_raise(bad: object) -> None:
    with pytest.raises(PricingInputError, match="real number"):
        _validate(S=bad)


def test_incompatible_shapes_raise_numpy_value_error_untouched() -> None:
    with pytest.raises(ValueError, match="broadcast") as info:
        _validate(S=np.ones(3), K=np.ones(4))
    assert not isinstance(info.value, PricingInputError)  # SPEC 2.3: NumPy's error, untouched


def test_unit_helpers_follow_spec_definitions() -> None:
    # SPEC 2.2: per_vol_point = vega / 100, per_calendar_day = theta / 365,
    # per_percentage_point = rho / 100.
    assert per_vol_point(8.81) == 8.81 / 100
    assert per_calendar_day(-4.559) == -4.559 / 365
    assert per_percentage_point(12.0) == 0.12
    np.testing.assert_array_equal(per_vol_point(np.array([1.0, 2.0])), np.array([0.01, 0.02]))
