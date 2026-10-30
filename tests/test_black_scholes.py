"""BSM prices (SPEC 3.1 to 3.5, 2.3): return types, spot form, a 50-digit reference, the boundary
policy, small-volatility continuity and extreme moneyness."""

import itertools
import math
from typing import Any

import numpy as np
import pytest
from scipy.special import ndtr

from mp_reference import mp_price_and_error_scale
from pricing import black_scholes as bs
from pricing.contracts import OptionType
from pricing.errors import PricingInputError

CALL, PUT = OptionType.CALL, OptionType.PUT
EPS = float(np.finfo(np.float64).eps)

# Hull, Options, Futures, and Other Derivatives, Example 15.6 (SPEC 9.1 R1).
R1: dict[str, Any] = {"S": 42.0, "K": 40.0, "T": 0.5, "r": 0.10, "q": 0.0, "sigma": 0.20}

# Rounding budget of the closed form: |error| <= ROUNDING_FACTOR * eps * scale, with
#     scale = D F N(theta d1) + D K N(theta d2) + D F phi(d1) (1 + |d1| + |d2|).
# The first two terms are what the formula subtracts, so their rounding survives the
# subtraction. The last covers rounding in d1 and d2: an absolute error of order eps (1 + |d|)
# that the two terms do not share moves the price by dV/ds = D F phi(d1) times that error. In a
# far tail phi(d) ~ |d| N(theta d) (Mills ratio), so this term grows like d^2 times the others.
# A first version of this budget omitted it and failed at d1 = -11.4, where the spot form's
# error was 63 eps times the terms and ours 1.8 (diagnosis in docs/notes/phase_1_bsm.md).
# There are about 10 roundings on the path; 64 gives margin (SPEC 9.3 starting factor).
ROUNDING_FACTOR = 64

# NumPy may evaluate exp and log with SIMD code for arrays and scalar code for single values;
# the two can differ by a few units in the last place.
VECTOR_SCALAR_RTOL = 8 * EPS

# Boundary formulas are a few roundings of exp, a product and a max: 4 ulps covers them.
BOUNDARY_RTOL = 4 * EPS

# A grid spanning moneyness, maturity, volatility and signed r and q (SPEC 9.2), with total
# volatility s >= 0.03 so every price is well away from underflow.
GRID_S = (70.0, 95.0, 100.0, 130.0)
GRID_T = (0.1, 1.0, 5.0)
GRID_R = (-0.01, 0.05)
GRID_Q = (-0.02, 0.03)
GRID_SIGMA = (0.1, 0.4, 1.0)
GRID_K = 100.0


def _grid_cases() -> list[dict[str, float]]:
    return [
        {"S": S, "K": GRID_K, "T": T, "r": r, "q": q, "sigma": sigma}
        for S, T, r, q, sigma in itertools.product(GRID_S, GRID_T, GRID_R, GRID_Q, GRID_SIGMA)
    ]


# --- SPEC 2.3: types and broadcasting ---------------------------------------------------------


def test_scalar_inputs_return_python_float() -> None:
    assert type(bs.price(CALL, **R1)) is float


def test_array_input_returns_float64_array_of_broadcast_shape() -> None:
    out = bs.price(CALL, **{**R1, "S": np.array([40.0, 42.0]), "sigma": np.array([[0.1], [0.2]])})
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == (2, 2)


def test_zero_dimensional_array_input_returns_array() -> None:
    out = bs.price(CALL, **{**R1, "S": np.array(42.0)})
    assert isinstance(out, np.ndarray)
    assert out.shape == ()


@pytest.mark.parametrize("option_type", [CALL, PUT])
def test_vectorized_matches_elementwise_scalar_evaluation(option_type: OptionType) -> None:
    # Mixes regular entries with T = 0 and sigma = 0 so the boundary masks are exercised in arrays.
    S = np.array([35.0, 40.0, 42.0, 50.0])[:, None]
    T = np.array([0.0, 0.25, 0.5, 2.0])[None, :]
    sigma = np.array([0.2, 0.0, 0.3, 0.6])[None, :]
    vector = bs.price(option_type, S=S, K=40.0, T=T, r=0.05, q=0.01, sigma=sigma)
    scalar = np.array(
        [
            [
                bs.price(option_type, S=s_, K=40.0, T=t_, r=0.05, q=0.01, sigma=v_)
                for t_, v_ in zip(T[0], sigma[0], strict=True)
            ]
            for s_ in S[:, 0]
        ]
    )
    np.testing.assert_allclose(vector, scalar, rtol=VECTOR_SCALAR_RTOL, atol=0.0)


def test_invalid_inputs_and_option_types_raise() -> None:
    with pytest.raises(PricingInputError, match=r"^sigma"):
        bs.price(CALL, **{**R1, "sigma": -0.2})
    with pytest.raises(PricingInputError, match="option_type"):
        bs.price("call", **R1)  # type: ignore[arg-type]


# --- SPEC 3.1, 3.2: closed form ---------------------------------------------------------------


@pytest.mark.parametrize("option_type", [CALL, PUT])
def test_forward_form_matches_spot_form(option_type: OptionType) -> None:
    # SPEC 3.2: C = S e^{-qT} N(d1) - K e^{-rT} N(d2), P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1),
    # with d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt(T)), evaluated independently here.
    th = 1 if option_type is CALL else -1
    S, T, r, q, sigma = np.meshgrid(GRID_S, GRID_T, GRID_R, GRID_Q, GRID_SIGMA, indexing="ij")
    K = GRID_K
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Sq, Kr = S * np.exp(-q * T), K * np.exp(-r * T)
    spot_term, strike_term = Sq * ndtr(th * d1), Kr * ndtr(th * d2)
    spot_form = th * (spot_term - strike_term)
    ours = bs.price(option_type, S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    vega_s = Sq * np.exp(-(d1**2) / 2) / math.sqrt(2 * math.pi)  # = D F phi(d1)
    scale = spot_term + strike_term + vega_s * (1 + np.abs(d1) + np.abs(d2))
    budget = ROUNDING_FACTOR * EPS * scale
    assert np.all(np.abs(ours - spot_form) <= budget)


@pytest.mark.parametrize("option_type", [CALL, PUT])
def test_matches_50_digit_reference_within_rounding_budget(option_type: OptionType) -> None:
    # Expected values: computed by mpmath at 50 significant digits (tests/mp_reference.py).
    th = 1 if option_type is CALL else -1
    failures = []
    for case in _grid_cases():
        exact, scale = mp_price_and_error_scale(th, **case)
        error = abs(bs.price(option_type, **case) - exact)
        if error > ROUNDING_FACTOR * EPS * scale:
            failures.append((case, error, error / (EPS * scale)))
    assert not failures, failures


def test_deep_out_of_the_money_put_keeps_relative_precision() -> None:
    # SPEC 3.4: tails use N(-x), never 1 - N(x). Here d2 = 9.24, and 1 - N(d2) evaluates to
    # exactly 0 in float64, so a 1 - N(x) put would return 0: a 100% error.
    case = {"S": 100.0, "K": 40.0, "T": 0.25, "r": 0.05, "q": 0.0, "sigma": 0.20}
    k = bs._kernel(PUT, **case, allow_extreme_vol=False)
    assert 1.0 - ndtr(float(k.d2)) == 0.0  # the failure the rule prevents
    # Expected value: computed by mpmath at 50 significant digits.
    exact, scale = mp_price_and_error_scale(-1, **case)
    assert 0.0 < exact < 1e-15  # precondition: genuinely deep out of the money
    ours = bs.price(PUT, **case)
    assert abs(ours - exact) <= ROUNDING_FACTOR * EPS * scale


def test_far_out_of_the_money_call_underflows_to_exact_zero() -> None:
    # d2 = -69: N(d2) is below the smallest subnormal float, so 0 is the correctly rounded price.
    assert bs.price(CALL, S=1.0, K=1000.0, T=0.25, r=0.05, q=0.0, sigma=0.20) == 0.0


# --- SPEC 3.4: boundary policy ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("option_type", "S", "intrinsic"),
    [(CALL, 42.0, 2.0), (CALL, 38.0, 0.0), (CALL, 40.0, 0.0), (PUT, 38.0, 2.0), (PUT, 42.0, 0.0)],
)
def test_expired_price_is_undiscounted_intrinsic(
    option_type: OptionType, S: float, intrinsic: float
) -> None:
    # SPEC 3.4: at T = 0 the price is max(theta (S - K), 0) with K = 40, whatever r, q, sigma.
    assert bs.price(option_type, S=S, K=40.0, T=0.0, r=0.10, q=0.03, sigma=0.2) == intrinsic


@pytest.mark.parametrize("option_type", [CALL, PUT])
@pytest.mark.parametrize("S", [90.0, 100.0, 110.0])
def test_zero_vol_price_is_discounted_payoff_on_certain_forward(
    option_type: OptionType, S: float
) -> None:
    # SPEC 3.4: sigma = 0, T > 0 gives D max(theta (F - K), 0).
    K, T, r, q = 100.0, 0.75, 0.04, 0.01
    th = 1 if option_type is CALL else -1
    expected = math.exp(-r * T) * max(th * (S * math.exp((r - q) * T) - K), 0.0)
    ours = bs.price(option_type, S=S, K=K, T=T, r=r, q=q, sigma=0.0)
    assert ours == pytest.approx(expected, rel=BOUNDARY_RTOL, abs=0.0)


@pytest.mark.parametrize("option_type", [CALL, PUT])
@pytest.mark.parametrize("S", [95.0, 100.0, 105.0])
def test_price_approaches_zero_vol_limit_as_total_vol_shrinks(
    option_type: OptionType, S: float
) -> None:
    # SPEC 3.4: continuity for s = 1e-4 ... 1e-12, with no NaN or sign error. With T = 1, s = sigma,
    # and r = q makes F = S, so S = 100 sits exactly at the forward. Bounds: V(s) >= V(0) by
    # Jensen's inequality, and time value peaks at the forward where it equals
    # D F (2 N(s/2) - 1) <= D F s / sqrt(2 pi), so V(s) - V(0) <= D F s / sqrt(2 pi).
    kw = {"S": S, "K": 100.0, "T": 1.0, "r": 0.02, "q": 0.02}
    th = 1 if option_type is CALL else -1
    limit = bs.price(option_type, **kw, sigma=0.0)
    D, F = math.exp(-0.02), S
    for s in 10.0 ** -np.arange(4, 13):
        value = bs.price(option_type, **kw, sigma=float(s))
        _, scale = mp_price_and_error_scale(th, **kw, sigma=float(s))
        rounding = ROUNDING_FACTOR * EPS * scale
        assert math.isfinite(value)
        assert value >= 0.0
        assert limit - rounding <= value <= limit + D * F * s / math.sqrt(2 * math.pi) + rounding
