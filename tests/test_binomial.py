"""Binomial trees (SPEC 4): parameters, validity, convergence to BSM, American exercise."""

import dataclasses
import itertools
import math
from typing import Any

import mpmath
import numpy as np
import pytest

from pricing import binomial as bt
from pricing import black_scholes as bs
from pricing.contracts import ExerciseStyle, OptionType, option_sign
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


# --- SPEC 4.2: Leisen-Reimer --------------------------------------------------------------

# R2: Haug (2007), 2nd ed., p. 4 (SPEC 9.1); a case with a dividend yield.
R2: dict[str, Any] = {"S": 100.0, "K": 95.0, "T": 0.5, "r": 0.10, "q": 0.05, "sigma": 0.20}
# LR's European error is O(1/N^2) for odd N (SPEC 4.2). Observed N^2 |error| = 0.036 (R1) and
# 0.151 (R2) for N = 101 ... 801; 0.5 leaves more than 3x margin.
LR_ERROR_CONSTANT = 0.5
# BBS's error is O(1/N) without oscillation (Broadie and Detemple 1996). Observed N |error| <= 0.35
# on R1 and R2 for N = 64 ... 512; 1.0 leaves ~3x margin.
BBS_ERROR_CONSTANT = 1.0
# h(z) + (1 - h(z)) = 1 up to one rounding of the sum.
PP_SUM_TOL = 2 * EPS
# Richardson on an exact c N^-2 sequence leaves only rounding of a 4-term combination.
RICHARDSON_TOL = 16 * EPS


def test_lr_needs_odd_steps() -> None:
    with pytest.raises(TreeParameterError, match="odd"):
        bt.leisen_reimer(CALL, **R1, n_steps=100)


def test_peizer_pratt_inversion_is_symmetric_and_centred() -> None:
    assert bt._peizer_pratt_2(0.0, 101) == (0.5, 0.5)
    previous = 0.0
    for z in (-5.0, -0.3, 0.7, 4.0):
        h, one_minus_h = bt._peizer_pratt_2(z, 101)
        assert bt._peizer_pratt_2(-z, 101) == (one_minus_h, h)  # h(-z) = 1 - h(z)
        assert abs(h + one_minus_h - 1.0) <= PP_SUM_TOL
        assert h > previous  # increasing in z
        previous = h


def test_peizer_pratt_tail_keeps_relative_precision() -> None:
    # z = 30, n = 3: exp(-x) ~ 1e-110, so 1 - h computed as 1 - (1/2 + sqrt(...)/2) is exactly 0.
    # The rewritten form keeps it. Expected value: mpmath at 50 significant digits.
    z, n = 30.0, 3
    h, one_minus_h = bt._peizer_pratt_2(z, n)
    assert 1.0 - h == 0.0  # what the naive subtraction would return
    with mpmath.workdps(50):
        t = mpmath.mpf(z) / (n + mpmath.mpf(1) / 3 + mpmath.mpf("0.1") / (n + 1))
        exact = (1 - mpmath.sqrt(1 - mpmath.exp(-t * t * (n + mpmath.mpf(1) / 6)))) / 2
    assert one_minus_h == pytest.approx(float(exact), rel=STABLE_P_RTOL)


def test_lr_parameters_satisfy_the_martingale_condition() -> None:
    res = bt.leisen_reimer(CALL, **R2, n_steps=101)
    g = math.exp((0.10 - 0.05) * res.dt)
    assert res.p_up * res.up + (1 - res.p_up) * res.down == pytest.approx(g, rel=MARTINGALE_RTOL)
    assert res.down < g < res.up


@pytest.mark.parametrize(("case", "option_type"), [(R1, CALL), (R1, PUT), (R2, CALL), (R2, PUT)])
@pytest.mark.parametrize("n_steps", [101, 201, 401, 801])
def test_lr_european_error_is_order_one_over_n_squared(
    case: dict[str, Any], option_type: OptionType, n_steps: int
) -> None:
    error = bt.leisen_reimer(option_type, **case, n_steps=n_steps).value - bs.price(
        option_type, **case
    )
    assert abs(error) <= LR_ERROR_CONSTANT / n_steps**2


def test_lr_error_is_monotone_in_n() -> None:
    # A monotone error is what makes Richardson extrapolation legitimate for LR (SPEC 4.5).
    exact = bs.price(CALL, **R1)
    errors = [bt.leisen_reimer(CALL, **R1, n_steps=n).value - exact for n in (101, 201, 401, 801)]
    assert all(e < 0 for e in errors)
    assert all(abs(a) > abs(b) for a, b in itertools.pairwise(errors))


# --- SPEC 4.3: BBS and BBSR ---------------------------------------------------------------


@pytest.mark.parametrize("option_type", [CALL, PUT])
def test_bbs_with_one_step_is_black_scholes(option_type: OptionType) -> None:
    # With N = 1 the only layer before expiry is t = 0, which BBS fills with the BSM value.
    assert bt.bbs(option_type, **R2, n_steps=1).value == bs.price(option_type, **R2)
    intrinsic = max(option_sign(option_type) * (R2["S"] - R2["K"]), 0.0)
    american = bt.bbs(option_type, **R2, n_steps=1, exercise=AMERICAN).value
    assert american == max(intrinsic, bs.price(option_type, **R2))


@pytest.mark.parametrize(("case", "option_type"), [(R1, CALL), (R2, PUT)])
def test_bbs_error_is_order_one_over_n_without_sign_changes(
    case: dict[str, Any], option_type: OptionType
) -> None:
    exact = bs.price(option_type, **case)
    errors = {n: bt.bbs(option_type, **case, n_steps=n).value - exact for n in (64, 128, 256, 512)}
    assert all(abs(e) <= BBS_ERROR_CONSTANT / n for n, e in errors.items())
    assert len({e > 0 for e in errors.values()}) == 1  # one sign, unlike CRR


@pytest.mark.parametrize(("case", "option_type"), [(R1, CALL), (R2, PUT)])
@pytest.mark.parametrize("n_steps", [64, 128, 256, 512])
def test_bbsr_beats_bbs_at_the_same_n(
    case: dict[str, Any], option_type: OptionType, n_steps: int
) -> None:
    exact = bs.price(option_type, **case)
    bbsr = bt.bbsr(option_type, **case, n_steps=n_steps)
    assert abs(bbsr.value - exact) < abs(bt.bbs(option_type, **case, n_steps=n_steps).value - exact)
    assert bbsr.components is not None
    full, half = bbsr.components
    assert bbsr.value == 2 * full - half  # SPEC 4.3


def test_bbsr_needs_even_steps() -> None:
    with pytest.raises(TreeParameterError, match="even"):
        bt.bbsr(CALL, **R1, n_steps=101)


def test_richardson_cancels_the_leading_error_term() -> None:
    # V(N) = 3 + 0.7 N^-2 exactly: extrapolating V(10) and V(20) with order 2 recovers 3.
    def v(n: int) -> float:
        return 3.0 + 0.7 / n**2

    assert abs(bt.richardson(v(10), v(20), order=2.0) - 3.0) <= RICHARDSON_TOL * 3.0
    with pytest.raises(ValueError, match="order"):
        bt.richardson(v(10), v(20), order=0.0)


@pytest.mark.parametrize("method", [bt.leisen_reimer, bt.bbs])
def test_american_is_worth_at_least_european_for_lr_and_bbs(method: Any) -> None:
    eu = method(PUT, **R4_PUT, n_steps=301).value
    am = method(PUT, **R4_PUT, n_steps=301, exercise=AMERICAN).value
    assert am >= eu


def test_bbs_records_a_boundary_entry_per_layer() -> None:
    res = bt.bbs(PUT, **R4_PUT, n_steps=50, exercise=AMERICAN, record_boundary=True)
    assert res.exercise_boundary is not None
    assert len(res.exercise_boundary) == 50
