"""No-arbitrage identities, bounds, monotonicity and Greek signs (SPEC 9.2).

Hypothesis draws markets from the valid domain; a deterministic grid adds the parity identity over
signed r and q. Tolerances come from the rounding model in test_black_scholes.py: an absolute
error of at most ROUNDING_FACTOR * eps * scale, where scale = Sq + Kr + Sq phi(d1) (1 + |d1| +
|d2|) bounds the terms the closed form subtracts plus the effect of rounding in d1 and d2.
"""

import itertools
import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from pricing import black_scholes as bs
from pricing.contracts import OptionType

CALL, PUT = OptionType.CALL, OptionType.PUT
EPS = float(np.finfo(np.float64).eps)
ROUNDING_FACTOR = 64  # as in test_black_scholes.py


@st.composite
def markets(draw: st.DrawFn, *, r_min: float = -0.05, q_max: float = 0.15) -> dict[str, float]:
    """|ln(S/K)| <= 2, T from 1 day to 10 years, sigma in [0.01, 2], r and q in [-5%, 15%]."""
    K = draw(st.floats(1.0, 1000.0))
    S = K * math.exp(draw(st.floats(-2.0, 2.0)))
    return {
        "S": S,
        "K": K,
        "T": draw(st.floats(1 / 365, 10.0)),
        "r": draw(st.floats(r_min, 0.15)),
        "q": draw(st.floats(-0.05, q_max)) if q_max > 0 else 0.0,
        "sigma": draw(st.floats(0.01, 2.0)),
    }


def _rounding(case: dict[str, float]) -> float:
    S, K, T, r, q, sigma = (case[k] for k in ("S", "K", "T", "r", "q", "sigma"))
    Sq, Kr = S * math.exp(-q * T), K * math.exp(-r * T)
    s = sigma * math.sqrt(T)
    d1 = math.log(Sq / Kr) / s + s / 2
    d2 = d1 - s
    vega_s = Sq * math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
    return ROUNDING_FACTOR * EPS * (Sq + Kr + vega_s * (1 + abs(d1) + abs(d2)))


def _discounted(case: dict[str, float]) -> tuple[float, float]:
    return (
        case["S"] * math.exp(-case["q"] * case["T"]),
        case["K"] * math.exp(-case["r"] * case["T"]),
    )


# --- Identities ------------------------------------------------------------------------------


def _parity_tolerance(Sq: float | np.ndarray, Kr: float | np.ndarray) -> float | np.ndarray:
    # C - P = D (F (N(d1) + N(-d1)) - K (N(d2) + N(-d2))): rounding in d1 and d2 cancels exactly
    # because N(d) + N(-d) = 1 for any d, leaving rounding of four terms summing to Sq + Kr.
    return ROUNDING_FACTOR * EPS * (Sq + Kr)


def test_put_call_parity_on_signed_rate_grid() -> None:
    # SPEC 9.2: parity C - P = S e^{-qT} - K e^{-rT} over r, q (including negative), T, sigma and
    # moneyness. 720 cases, evaluated as one broadcast call.
    axes = itertools.product(
        (50.0, 90.0, 100.0, 110.0, 200.0),
        (1 / 365, 0.1, 1.0, 10.0),
        (-0.03, 0.0, 0.08),
        (-0.03, 0.0, 0.08),
        (0.01, 0.2, 1.0, 3.0),
    )
    S, T, r, q, sigma = (np.array(a) for a in zip(*axes, strict=True))
    kw = {"S": S, "K": 100.0, "T": T, "r": r, "q": q, "sigma": sigma}
    Sq, Kr = S * np.exp(-q * T), 100.0 * np.exp(-r * T)
    residual = bs.price(CALL, **kw) - bs.price(PUT, **kw) - (Sq - Kr)
    assert np.all(np.abs(residual) <= _parity_tolerance(Sq, Kr))


@given(markets())
def test_put_call_parity(case: dict[str, float]) -> None:
    Sq, Kr = _discounted(case)
    residual = bs.price(CALL, **case) - bs.price(PUT, **case) - (Sq - Kr)
    assert abs(residual) <= _parity_tolerance(Sq, Kr)


# --- Bounds ----------------------------------------------------------------------------------


@given(markets())
def test_no_arbitrage_bounds(case: dict[str, float]) -> None:
    # SPEC 9.2: max(Sq - Kr, 0) <= C <= Sq and max(Kr - Sq, 0) <= P <= Kr, and never negative.
    Sq, Kr = _discounted(case)
    tol = _rounding(case)
    call, put = bs.price(CALL, **case), bs.price(PUT, **case)
    assert isinstance(call, float)
    assert isinstance(put, float)
    assert call >= 0.0
    assert put >= 0.0
    assert max(Sq - Kr, 0.0) - tol <= call <= Sq + tol
    assert max(Kr - Sq, 0.0) - tol <= put <= Kr + tol


# --- Monotonicity ----------------------------------------------------------------------------

bumps = st.floats(1e-6, 0.5)


@given(markets(), bumps)
def test_monotone_in_spot(case: dict[str, float], f: float) -> None:
    up = {**case, "S": case["S"] * (1 + f)}
    tol = _rounding(case) + _rounding(up)
    assert bs.price(CALL, **up) >= bs.price(CALL, **case) - tol
    assert bs.price(PUT, **up) <= bs.price(PUT, **case) + tol


@given(markets(), bumps)
def test_monotone_in_strike(case: dict[str, float], f: float) -> None:
    up = {**case, "K": case["K"] * (1 + f)}
    tol = _rounding(case) + _rounding(up)
    assert bs.price(CALL, **up) <= bs.price(CALL, **case) + tol
    assert bs.price(PUT, **up) >= bs.price(PUT, **case) - tol


@given(markets(), bumps)
def test_monotone_in_volatility(case: dict[str, float], f: float) -> None:
    up = {**case, "sigma": case["sigma"] * (1 + f)}
    tol = _rounding(case) + _rounding(up)
    for option_type in (CALL, PUT):
        assert bs.price(option_type, **up) >= bs.price(option_type, **case) - tol


@given(markets(r_min=0.0, q_max=0.0), bumps)
def test_call_monotone_in_maturity_without_dividends(case: dict[str, float], f: float) -> None:
    # Holds only under stated assumptions (SPEC 9.2): with q = 0 and r >= 0, SPEC 3.3 gives
    # theta_call = -S phi(d1) sigma / (2 sqrt(T)) - r K e^{-rT} N(d2) <= 0, so dC/dT >= 0.
    longer = {**case, "T": case["T"] * (1 + f)}
    tol = _rounding(case) + _rounding(longer)
    assert bs.price(CALL, **longer) >= bs.price(CALL, **case) - tol


# --- Greek signs -----------------------------------------------------------------------------


@given(markets())
def test_greek_signs(case: dict[str, float]) -> None:
    # Each Greek in SPEC 3.3 is a product of factors with known signs, so these hold exactly in
    # floating point: rounding a product never changes its sign or pushes it past a factor <= 1.
    disc_q = math.exp(-case["q"] * case["T"])
    assert 0.0 <= bs.delta(CALL, **case) <= disc_q
    assert -disc_q <= bs.delta(PUT, **case) <= 0.0
    for option_type in (CALL, PUT):
        assert bs.gamma(option_type, **case) >= 0.0
        assert bs.vega(option_type, **case) >= 0.0
    assert bs.rho(CALL, **case) >= 0.0
    assert bs.rho(PUT, **case) <= 0.0
    assert bs.phi(CALL, **case) <= 0.0
    assert bs.phi(PUT, **case) >= 0.0


@given(markets())
def test_theta_is_non_positive_without_carry(case: dict[str, float]) -> None:
    # With r = q = 0 only the -Sq phi(d1) sigma / (2 sqrt(T)) term of SPEC 3.3 remains.
    no_carry = {**case, "r": 0.0, "q": 0.0}
    for option_type in (CALL, PUT):
        assert bs.theta(option_type, **no_carry) <= 0.0
