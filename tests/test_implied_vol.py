"""Implied-volatility solvers (SPEC 6): feasibility, SPEC 6.4 round trips, failure reporting."""

import dataclasses
import itertools
import math
from typing import Any

import mpmath
import pytest
from experiments.validation import IV_GRID, IV_Q, IV_R, IV_STRIKE, iv_market
from scipy.special import ndtr

from pricing import black_scholes as bs
from pricing import implied_vol as iv
from pricing.contracts import OptionType
from pricing.errors import ConvergenceError, InfeasiblePriceError, PricingInputError
from pricing.implied_vol import IVMethod, IVReason

CALL, PUT = OptionType.CALL, OptionType.PUT
EPS = iv.EPS
R, Q, STRIKE = IV_R, IV_Q, IV_STRIKE  # shared with experiments/validation.py

# Accuracy model. The target price carries rounding error up to ~eps x scale (Phase 1 model:
# scale = F N(td1) + K N(td2) + F phi(d1)(1 + |d1| + |d2|), observed <= 0.76 eps x scale), and the
# solver evaluates c(s) with the same error, so |s_hat - s| ~ eps x scale / (dc/ds). Observed worst
# on the SPEC 6.4 grid: 3.3 (naive Newton), 1.1 (safeguarded; 3.0 before Study A2's fixes),
# 1.0 (Brent, on top of its stopping tolerance). 32 leaves ~10x margin.
IV_ROUNDING_FACTOR = 32


_market = iv_market


def _allowed_error(option_type: OptionType, x: float, s: float, method: IVMethod) -> float:
    th = 1 if option_type is CALL else -1
    F = STRIKE * math.exp(x)
    d1 = x / s + s / 2
    d2 = d1 - s
    terms = F * float(ndtr(th * d1)) + STRIKE * float(ndtr(th * d2))
    vega = F * math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
    allowed = IV_ROUNDING_FACTOR * EPS * (terms + vega * (1 + abs(d1) + abs(d2))) / vega
    if method is IVMethod.BRENT:
        allowed += iv.BRENT_XTOL + iv.CONVERGENCE_FACTOR * EPS * s  # its stopping tolerance
    return allowed


# SPEC 6.4 grid (x in [-3, 3], T in {1/365, 7/365, 0.25, 1, 5}, sigma in {0.01, 0.2, 1, 3}),
# shared with the validation summary.
ROUND_TRIP = IV_GRID


def _check_round_trip(method: IVMethod) -> dict[IVReason, int]:
    counts: dict[IVReason, int] = {}
    for x, T, sigma, option_type in ROUND_TRIP:
        market = _market(float(x), T)
        price = float(bs.price(option_type, **market, sigma=sigma))
        res = iv.implied_volatility_result(option_type, price, **market, method=method)
        counts[res.reason] = counts.get(res.reason, 0) + 1
        case = (float(x), T, sigma, option_type)
        assert (res.sigma is None) == (not res.converged), case  # never a value unless converged
        if res.reason is IVReason.AT_LOWER_BOUND:
            # The time value is below rounding (in the money) or the price underflowed to 0.
            limit = float(bs.price(option_type, **market, sigma=0.0))
            F = STRIKE * math.exp(float(x))
            rounding = iv.BOUND_TOL_FACTOR * EPS * max(F, STRIKE) * math.exp(-R * T)
            assert abs(price - limit) <= rounding, case
            assert res.sigma == 0.0
        elif res.reason is IVReason.CONVERGED:
            assert res.total_vol is not None
            s = sigma * math.sqrt(T)
            assert abs(res.total_vol - s) <= _allowed_error(option_type, float(x), s, method), case
    return counts


@pytest.mark.parametrize("method", [IVMethod.SAFEGUARDED_NEWTON, IVMethod.BRENT])
def test_round_trip_grid_converges_or_hits_lower_bound(method: IVMethod) -> None:
    counts = _check_round_trip(method)
    assert set(counts) <= {IVReason.CONVERGED, IVReason.AT_LOWER_BOUND}, counts


def test_naive_newton_reports_failures_and_is_accurate_when_it_converges() -> None:
    # SPEC 6.3: no safeguards; it exists to be measured (Study A, A1).
    counts = _check_round_trip(IVMethod.NAIVE_NEWTON)
    failures = {
        r: n for r, n in counts.items() if r not in (IVReason.CONVERGED, IVReason.AT_LOWER_BOUND)
    }
    assert failures  # it does fail on this grid
    assert set(failures) <= {IVReason.ZERO_DERIVATIVE, IVReason.DIVERGED, IVReason.MAX_ITERATIONS}


# --- SPEC 6.2: feasibility --------------------------------------------------------------------

ITM: dict[str, Any] = _market(0.3, 0.5)  # F > K: call in the money
OTM: dict[str, Any] = _market(-0.7, 0.5)  # F < K: call out of the money


def test_price_below_intrinsic_is_infeasible() -> None:
    intrinsic = float(bs.price(CALL, **ITM, sigma=0.0))
    with pytest.raises(InfeasiblePriceError, match="below the intrinsic"):
        iv.implied_volatility(CALL, intrinsic * (1 - 1e-6), **ITM)
    with pytest.raises(InfeasiblePriceError, match="below the intrinsic"):
        iv.implied_volatility(CALL, -1e-300, **OTM)


def test_price_at_upper_bound_is_infeasible() -> None:
    # A call is worth less than the discounted forward S e^{-qT} for any finite sigma.
    upper = ITM["S"] * math.exp(-Q * ITM["T"])
    with pytest.raises(InfeasiblePriceError, match="upper bound"):
        iv.implied_volatility(CALL, upper, **ITM)


def test_in_the_money_price_at_or_just_below_intrinsic_is_at_lower_bound() -> None:
    # SPEC 6.2: c at iota, or below it by no more than the rounding tolerance, gives s = 0.
    F, K = 100.0, 90.0
    for c in (F - K, (F - K) * (1 - 4 * EPS)):
        res = iv.solve_black(CALL, c, F=F, K=K, T=1.0)
        assert res.reason is IVReason.AT_LOWER_BOUND
        assert res.converged
        assert res.sigma == 0.0
        assert res.iterations == 0


def test_in_the_money_tiny_positive_time_value_is_solved() -> None:
    # Study A2 follow-up (SPEC 6.2): any positive time value is solved, and the answer reproduces
    # the price's time value to the solver's stopping rule (4 eps relative residual).
    F, K = 100.0, 90.0
    c = (F - K) + 1e-13  # time value ~ 0.45 eps max(F, K), below one ulp of c
    res = iv.solve_black(CALL, c, F=F, K=K, T=1.0)
    assert res.reason is IVReason.CONVERGED
    assert res.total_vol is not None
    put = iv.solve_black(PUT, c - (F - K), F=F, K=K, T=1.0)  # the same problem, out of the money
    assert put.total_vol == res.total_vol


def test_out_of_the_money_zero_price_is_at_lower_bound() -> None:
    assert iv.implied_volatility_result(CALL, 0.0, **OTM).reason is IVReason.AT_LOWER_BOUND


def test_tiny_out_of_the_money_price_is_solved_not_rounded_to_zero() -> None:
    # Phase 3 amendment of SPEC 6.2: out of the money the intrinsic value is exactly 0, so no
    # rounding tolerance applies. This price is far below SPEC's original tolerance
    # 8 eps max(F, K), which would have returned sigma = 0, yet its volatility is recoverable.
    sigma = 0.05
    price = float(bs.price(CALL, **OTM, sigma=sigma))
    F = OTM["S"] * math.exp((R - Q) * OTM["T"])
    assert 0.0 < price < iv.BOUND_TOL_FACTOR * EPS * max(F, STRIKE)  # precondition
    for method in (IVMethod.SAFEGUARDED_NEWTON, IVMethod.BRENT):
        res = iv.implied_volatility_result(CALL, price, **OTM, method=method)
        s = sigma * math.sqrt(OTM["T"])
        assert res.total_vol is not None
        assert abs(res.total_vol - s) <= _allowed_error(CALL, -0.7, s, method)


# --- Study A2 fixes: no cancellation near the money, in the money via out of the money ---------

# black_price against mpmath on exact inputs. The double path is taken only when the two terms
# cancel by at most a factor CANCELLATION_LIMIT = 4, each term carrying ~2 eps, so its error is
# at most ~4 x 2 x 2 = 16 eps; the extra-digit path is correctly rounded. Observed worst on this
# grid: 1.94 eps (the old direct formula: 1.25e10 eps, at s = 1e-10).
BLACK_PRICE_RTOL = 16 * EPS
NEAR_MONEY_X = (
    0.0,
    1e-12,
    -1e-12,
    1e-8,
    -1e-8,
    1e-6,
    -1e-6,
    1e-4,
    -1e-4,
    1e-3,
    -1e-3,
    1e-2,
    -1e-2,
    0.05,
    -0.05,
    0.1,
    -0.1,
    0.3,
    -0.3,
    0.5,
    -0.5,
)
SMALL_S = (1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)


def _mp_black(theta: int, F: float, K: float, s: float) -> float:
    # Expected values: computed by mpmath at 60 significant digits on the exact double inputs.
    with mpmath.workdps(60):
        F_, K_, s_ = (mpmath.mpf(v) for v in (F, K, s))
        d1 = mpmath.log(F_ / K_) / s_ + s_ / 2
        return float(theta * (F_ * mpmath.ncdf(theta * d1) - K_ * mpmath.ncdf(theta * (d1 - s_))))


@pytest.mark.parametrize("theta", [1, -1])
def test_black_price_keeps_relative_accuracy_near_the_money_at_small_vol(theta: int) -> None:
    failures = []
    for x, s in itertools.product(NEAR_MONEY_X, SMALL_S):
        F = 100.0
        K = F * math.exp(-x)
        problem = iv._Problem(theta, F, K, math.log(F / K), 1.0, 0.0, max(theta * (F - K), 0.0))
        exact = _mp_black(theta, F, K, s)
        if exact == 0.0:
            continue
        got = iv.black_price(problem, s)
        if abs(got - exact) > BLACK_PRICE_RTOL * exact:
            failures.append((x, s, got, exact))
    assert not failures, failures[:5]


# A round trip near the money at small vol is well conditioned: c ~ 0.4 F s, so s inherits the
# price's relative error. Target rounding (1 eps) + stopping rule (4 eps relative residual) +
# evaluation (2 eps) is ~7 eps; 16 eps leaves 2x margin. Brent also stops at its absolute
# xtol = 1e-15 (SPEC 6.3), which dominates at small s. Moneyness is set relative to s,
# x = h s with |h| <= 2, so every price is informative (|h| >> 1 underflows to 0).
NEAR_MONEY_S_RTOL = 16 * EPS


@pytest.mark.parametrize("method", [IVMethod.SAFEGUARDED_NEWTON, IVMethod.BRENT])
def test_near_the_money_small_vol_round_trip_is_accurate(method: IVMethod) -> None:
    failures = []
    for h, s, theta in itertools.product(
        (0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0), (1e-6, 1e-4, 1e-3, 1e-2), (1, -1)
    ):
        F, T = 100.0, 1.0
        K = F * math.exp(-h * s)
        option_type = CALL if theta == 1 else PUT
        c = _mp_black(theta, F, K, s)
        res = iv.solve_black(option_type, c, F=F, K=K, T=T, method=method)
        allowed = NEAR_MONEY_S_RTOL * s
        if method is IVMethod.BRENT:
            allowed += iv.BRENT_XTOL
        if res.total_vol is None or abs(res.total_vol - s) > allowed:
            failures.append((h, s, theta, res.reason, res.total_vol))
    assert not failures, failures[:5]


def test_in_the_money_price_with_informative_time_value_is_solved() -> None:
    # Deep in the money, small vol: the time value is a few eps max(F, K), below Study A's 8-eps
    # lower-bound rule, yet the price still pins sigma because the time value is so sensitive to s
    # (elasticity d ln c_otm / d ln s ~ x^2 / s^2). Study A found 30 such prices returned as s = 0.
    F, K, T, s = 100.0, 100.0 * math.exp(-0.5), 1.0, 0.0685  # put ~ 1e-13 ~ 4.5 eps F
    put = _mp_black(-1, F, K, s)  # the out-of-the-money counterpart
    call = put + (F - K)  # rounded once, as a quoted price would be
    time_value = call - (F - K)
    assert 2 * EPS * F < time_value < iv.BOUND_TOL_FACTOR * EPS * F  # precondition
    # Allowed error: the time value's rounding (~eps F) relative to it, over the elasticity.
    elasticity = (0.5 / s) ** 2
    allowed = (EPS * F / time_value) / elasticity * s
    for method in (IVMethod.SAFEGUARDED_NEWTON, IVMethod.BRENT):
        res = iv.solve_black(CALL, call, F=F, K=K, T=T, method=method)
        assert res.reason is IVReason.CONVERGED, method
        assert res.total_vol is not None
        assert abs(res.total_vol - s) <= 4 * allowed, (method, res.total_vol)


@pytest.mark.parametrize(
    ("override", "match"),
    [({"price": math.nan}, "price"), ({"price": True}, "price"), ({"T": 0.0}, "T must be")],
)
def test_invalid_inputs_raise(override: dict[str, Any], match: str) -> None:
    kwargs = {"price": 1.0, **ITM, **override}
    price = kwargs.pop("price")
    with pytest.raises(PricingInputError, match=match):
        iv.implied_volatility_result(CALL, price, **kwargs)


def test_invalid_solver_settings_raise() -> None:
    with pytest.raises(PricingInputError, match="IVMethod"):
        iv.implied_volatility_result(CALL, 5.0, **ITM, method="brent")  # type: ignore[arg-type]
    with pytest.raises(PricingInputError, match="max_iterations"):
        iv.implied_volatility_result(CALL, 5.0, **ITM, max_iterations=0)


# --- SPEC 6.3: solver behaviour ---------------------------------------------------------------

DEEP_OTM = _market(-3.0, 1.0)  # call: x = -3, so at sigma = 0.2, d1 = -14.9


def test_deep_out_of_the_money_low_vega_case_is_recovered() -> None:
    # SPEC 6.4: deep OTM, low vega. Price ~1e-50: tiny but well conditioned.
    price = float(bs.price(CALL, **DEEP_OTM, sigma=0.2))
    assert 0.0 < price < 1e-40
    for method in (IVMethod.SAFEGUARDED_NEWTON, IVMethod.BRENT):
        res = iv.implied_volatility_result(CALL, price, **DEEP_OTM, method=method)
        assert res.total_vol is not None
        assert abs(res.total_vol - 0.2) <= _allowed_error(CALL, -3.0, 0.2, method)


def test_safeguarded_newton_converges_where_unguarded_steps_crawl() -> None:
    # x = -3, T = 1 day, sigma = 3: from the Manaster-Koehler start sqrt(6) Newton alone crawls
    # (c / c' ~ s^3 / x^2); the progress test switches to bisection. Before that test existed,
    # this case hit the 100-iteration cap.
    market = _market(-3.0, 1 / 365)
    price = float(bs.price(CALL, **market, sigma=3.0))
    res = iv.implied_volatility_result(CALL, price, **market, record_path=True)
    assert res.reason is IVReason.CONVERGED
    assert res.path is not None
    assert res.path[0] == math.sqrt(6.0)  # Manaster and Koehler (1982): sqrt(2 |x|)
    assert all(s > 0 for s in res.path)


def test_start_falls_back_to_default_at_the_money() -> None:
    market = _market(0.0, 0.25)
    price = float(bs.price(CALL, **market, sigma=0.3))
    res = iv.implied_volatility_result(CALL, price, **market, record_path=True)
    assert res.path is not None
    assert res.path[0] == iv.DEFAULT_TOTAL_VOL_FACTOR * math.sqrt(0.25)  # |x| < 1e-8


def test_iteration_cap_returns_no_value() -> None:
    price = float(bs.price(CALL, **DEEP_OTM, sigma=0.2))
    res = iv.implied_volatility_result(CALL, price, **DEEP_OTM, max_iterations=3)
    assert res.reason is IVReason.MAX_ITERATIONS
    assert not res.converged
    assert res.sigma is None
    assert res.total_vol is None
    assert res.iterations == 3


def test_wrapper_raises_convergence_error_instead_of_returning_a_value() -> None:
    # Naive Newton starts at s0 = 0.2 sqrt(T) = 0.0105, where d1 = -286 and dc/ds underflows to 0.
    market = _market(-3.0, 1 / 365)
    price = float(bs.price(CALL, **market, sigma=3.0))
    naive = iv.implied_volatility_result(CALL, price, **market, method=IVMethod.NAIVE_NEWTON)
    assert naive.reason is IVReason.ZERO_DERIVATIVE  # precondition: naive Newton fails here
    with pytest.raises(ConvergenceError, match="naive_newton did not converge: zero_derivative"):
        iv.implied_volatility(CALL, price, **market, method=IVMethod.NAIVE_NEWTON)


def test_brent_reports_bracket_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # With valid inputs c(20) is within ~1e-20 of the upper bound, so the cap is unreachable;
    # lowering it exercises the branch. True s here is 1.5.
    monkeypatch.setattr(iv, "S_MAX", 1.0)
    market = _market(0.0, 1.0)
    price = float(bs.price(CALL, **market, sigma=1.5))
    res = iv.implied_volatility_result(CALL, price, **market, method=IVMethod.BRENT)
    assert res.reason is IVReason.BRACKET_FAILURE
    assert res.sigma is None


def test_black_form_entry_point() -> None:
    # Undiscounted Black price on a forward, as Study A will call it.
    F, K, s = 100.0, 120.0, 0.4
    d1 = math.log(F / K) / s + s / 2
    c = F * float(ndtr(d1)) - K * float(ndtr(d1 - s))
    res = iv.solve_black(CALL, c, F=F, K=K, T=4.0)
    assert res.total_vol is not None
    assert res.sigma is not None
    assert abs(res.total_vol - s) <= _allowed_error(
        CALL, math.log(F / K), s, IVMethod.SAFEGUARDED_NEWTON
    )
    assert res.sigma == res.total_vol / 2.0  # sigma = s / sqrt(T)


def test_result_is_frozen() -> None:
    res = iv.implied_volatility_result(CALL, float(bs.price(CALL, **ITM, sigma=0.2)), **ITM)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.sigma = 0.0  # type: ignore[misc]
