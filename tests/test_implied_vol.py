"""Implied-volatility solvers (SPEC 6): feasibility, SPEC 6.4 round trips, failure reporting."""

import dataclasses
import math
from typing import Any

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
# on the SPEC 6.4 grid: 3.3 (naive Newton), 3.0 (safeguarded), 1.0 (Brent, on top of its stopping
# tolerance). 32 leaves ~10x margin.
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


def test_in_the_money_time_value_below_rounding_is_at_lower_bound() -> None:
    intrinsic = float(bs.price(CALL, **ITM, sigma=0.0))
    res = iv.implied_volatility_result(CALL, intrinsic + 1e-15, **ITM)
    assert res.reason is IVReason.AT_LOWER_BOUND
    assert res.converged
    assert res.sigma == 0.0
    assert res.iterations == 0


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
