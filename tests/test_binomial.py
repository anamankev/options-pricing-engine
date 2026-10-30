"""Binomial trees (SPEC 4): parameters, validity, convergence to BSM, American exercise."""

import dataclasses
import math
from typing import Any

import mpmath
import numpy as np
import pytest

from pricing import binomial as bt
from pricing import black_scholes as bs
from pricing.contracts import ExerciseStyle, OptionType
from pricing.errors import PricingInputError, TreeParameterError

CALL, PUT = OptionType.CALL, OptionType.PUT
EUROPEAN, AMERICAN = ExerciseStyle.EUROPEAN, ExerciseStyle.AMERICAN
EPS = float(np.finfo(np.float64).eps)

# Hull, Options, Futures, and Other Derivatives, Example 15.6 (SPEC 9.1 R1).
R1: dict[str, Any] = {"S": 42.0, "K": 40.0, "T": 0.5, "r": 0.10, "q": 0.0, "sigma": 0.20}
# Ju and Zhong (1999), Exhibit 3, (K, sigma, T) = (40, 0.2, 0.5833) (SPEC 9.1 R4).
R4_PUT: dict[str, Any] = {"S": 40.0, "K": 40.0, "T": 0.5833, "r": 0.0488, "q": 0.0, "sigma": 0.2}

# CRR's European error is O(1/N) (Cox, Ross and Rubinstein 1979; Leisen and Reimer 1996).
# Observed N |error| <= 0.4 on R1 for N <= 1001; 1.0 leaves 2.5x margin.
CRR_ERROR_CONSTANT = 1.0
# The textbook probability (g - d) / (u - d) loses about eps / (u - d) relatively; at
# u - d = 0.028 that is ~1e-14, so 1e-12 compares the two forms with margin.
NAIVE_P_RTOL = 1e-12
# Tree parameters are one or two roundings of exp, sqrt and a product: 4 ulps.
PARAM_RTOL = 4 * EPS
# p u + (1 - p) d = g: three products and a sum on top of the parameters: 8 ulps.
MARTINGALE_RTOL = 8 * EPS
# expm1 / sinh form against the 50-digit value: ~6 roundings; 16 ulps.
STABLE_P_RTOL = 16 * EPS


def _naive_crr_p(T: float, r: float, q: float, sigma: float, n: int) -> float:
    dt = T / n
    u, d, g = (
        math.exp(sigma * math.sqrt(dt)),
        math.exp(-sigma * math.sqrt(dt)),
        math.exp((r - q) * dt),
    )
    return (g - d) / (u - d)


# --- SPEC 4.1: parameters and validity ---------------------------------------------------------


def test_crr_parameters_follow_spec() -> None:
    res = bt.crr(CALL, **R1, n_steps=50)
    assert res.up == pytest.approx(math.exp(0.20 * math.sqrt(0.5 / 50)), rel=PARAM_RTOL)
    assert res.up * res.down == pytest.approx(1.0, rel=PARAM_RTOL)  # d = 1 / u
    assert res.p_up == pytest.approx(_naive_crr_p(0.5, 0.10, 0.0, 0.20, 50), rel=NAIVE_P_RTOL)
    # Martingale condition p u + (1 - p) d = g, which defines p.
    g = math.exp(0.10 * res.dt)
    assert res.p_up * res.up + (1 - res.p_up) * res.down == pytest.approx(g, rel=MARTINGALE_RTOL)


def test_crr_probability_keeps_precision_at_tiny_steps() -> None:
    # dt = 1e-12: g - d and u - d are ~2e-7, so the textbook form loses ~1e-9 relatively. The
    # expm1/sinh form keeps full precision. Expected value: mpmath at 50 significant digits.
    T, r, q, sigma, n = 1.0, 0.05, 0.01, 0.2, 10**12
    lattice = bt._crr_lattice(CALL, 100.0, 100.0, T, r, q, sigma, n, False)
    with mpmath.workdps(50):
        dt = mpmath.mpf(T) / n
        u, d = mpmath.exp(sigma * mpmath.sqrt(dt)), mpmath.exp(-sigma * mpmath.sqrt(dt))
        exact = float((mpmath.exp((r - q) * dt) - d) / (u - d))
    assert lattice.p == pytest.approx(exact, rel=STABLE_P_RTOL)
    assert abs(_naive_crr_p(T, r, q, sigma, n) / exact - 1) > 1e-11  # the failure avoided


def test_crr_invalid_probability_raises_with_minimum_steps() -> None:
    # |r - q| sqrt(dt) < sigma fails for r - q = 0.5, sigma = 0.1, T = 1, N = 10; the bound is
    # N > (r - q)^2 T / sigma^2 = 25 (SPEC 4.1). p is reported, never clipped.
    case = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.5, "q": 0.0, "sigma": 0.1}
    with pytest.raises(TreeParameterError, match=r"n_steps > 25 \(got 10\)"):
        bt.crr(CALL, **case, n_steps=10)
    assert 0.0 < bt.crr(CALL, **case, n_steps=26).p_up < 1.0


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, None])
def test_n_steps_must_be_positive_int(bad: object) -> None:
    with pytest.raises(TreeParameterError, match="n_steps"):
        bt.crr(CALL, **R1, n_steps=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("override", [{"T": 0.0}, {"sigma": 0.0}])
def test_tree_needs_positive_maturity_and_volatility(override: dict[str, float]) -> None:
    with pytest.raises(TreeParameterError, match="T > 0 and sigma > 0"):
        bt.crr(CALL, **{**R1, **override}, n_steps=10)


def test_tree_rejects_array_inputs() -> None:
    with pytest.raises(PricingInputError, match="scalar inputs"):
        bt.crr(CALL, **{**R1, "S": np.array([42.0, 43.0])}, n_steps=10)  # type: ignore[arg-type]


def test_tree_result_is_frozen() -> None:
    res = bt.crr(CALL, **R1, n_steps=10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.value = 0.0  # type: ignore[misc]


# --- Convergence to BSM (SPEC 9.2 cross-method) -------------------------------------------


@pytest.mark.parametrize("option_type", [CALL, PUT])
@pytest.mark.parametrize("n_steps", [50, 100, 201, 400, 801])
def test_crr_european_error_is_order_one_over_n(option_type: OptionType, n_steps: int) -> None:
    exact = bs.price(option_type, **R1)
    error = bt.crr(option_type, **R1, n_steps=n_steps).value - exact
    assert abs(error) <= CRR_ERROR_CONSTANT / n_steps


def test_crr_error_oscillates_between_odd_and_even_steps() -> None:
    # The error is not monotone in N, so Richardson extrapolation of raw CRR is unreliable
    # (SPEC 4.5). On R1 it changes sign between N = 1000 and 1001.
    exact = bs.price(CALL, **R1)
    even = bt.crr(CALL, **R1, n_steps=1000).value - exact
    odd = bt.crr(CALL, **R1, n_steps=1001).value - exact
    assert even > 0 > odd


# --- SPEC 4.4: American exercise ----------------------------------------------------------


@pytest.mark.parametrize("option_type", [CALL, PUT])
@pytest.mark.parametrize("q", [0.0, 0.06])
def test_american_is_worth_at_least_european(option_type: OptionType, q: float) -> None:
    # Exercise rights only add value: max(continuation, exercise) >= continuation, node by node.
    case = {**R1, "q": q}
    eu = bt.crr(option_type, **case, n_steps=300).value
    am = bt.crr(option_type, **case, n_steps=300, exercise=AMERICAN).value
    assert am >= eu


@pytest.mark.parametrize(("r", "q"), [(0.05, 0.0), (0.02, -0.01), (0.0, -0.02)])
def test_american_call_equals_european_when_early_exercise_is_never_optimal(
    r: float, q: float
) -> None:
    # Healy (2021), arXiv:2109.15157, Proposition 1: with q <= 0 and q <= r the American call is
    # never exercised early (SPEC 4.4). On the tree, continuation exceeds exercise at every node by
    # at least S (e^{-q dt} - 1) + K (1 - e^{-r dt}) >= 0, here far above rounding, so the two
    # inductions are identical. (r = q = 0 is excluded: the margin is then exactly 0.)
    case = {**R1, "r": r, "q": q}
    eu = bt.crr(CALL, **case, n_steps=300).value
    am = bt.crr(CALL, **case, n_steps=300, exercise=AMERICAN)
    assert am.value == eu
    assert all(
        b is None
        for b in bt.crr(
            CALL, **case, n_steps=50, exercise=AMERICAN, record_boundary=True
        ).exercise_boundary
        or ()
    )


def test_american_call_can_be_exercised_early_at_negative_rates() -> None:
    # Healy (2021): with r < 0 holding the strike costs money, so a deep in-the-money call is
    # exercised early even without dividends. Tests must not assert equality here (SPEC 4.4).
    case = {"S": 150.0, "K": 100.0, "T": 2.0, "r": -0.03, "q": 0.0, "sigma": 0.2}
    eu = bt.crr(CALL, **case, n_steps=400).value
    am = bt.crr(CALL, **case, n_steps=400, exercise=AMERICAN).value
    assert am > eu


def test_american_put_exercise_boundary() -> None:
    res = bt.crr(PUT, **R4_PUT, n_steps=200, exercise=AMERICAN, record_boundary=True)
    boundary = res.exercise_boundary
    assert boundary is not None
    assert len(boundary) == 200  # one entry per layer 0 .. N-1
    critical = [b for b in boundary if b is not None]
    assert critical
    assert all(b < R4_PUT["K"] for b in critical)  # a put is only exercised in the money
    # One step before expiry the boundary is close to K (exercise loses only one step of time
    # value); 0.9 K is a loose bracket for sigma sqrt(dt) = 0.011.
    last = boundary[-1]
    assert last is not None
    assert 0.9 * R4_PUT["K"] < last < R4_PUT["K"]


def test_boundary_is_only_recorded_for_american_when_requested() -> None:
    assert bt.crr(PUT, **R4_PUT, n_steps=20, record_boundary=True).exercise_boundary is None
    assert bt.crr(PUT, **R4_PUT, n_steps=20, exercise=AMERICAN).exercise_boundary is None
