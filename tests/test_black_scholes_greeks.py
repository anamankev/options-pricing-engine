"""Analytic BSM Greeks (SPEC 3.3, 3.4): agreement with finite differences, which also pins the
units of SPEC 2.2, call-put identities, sanity anchors, and the boundary policy."""

import math
from typing import Any

import numpy as np
import pytest
from experiments.validation import GREEK_GRID, greek_finite_difference

from pricing import black_scholes as bs
from pricing.contracts import OptionType
from pricing.errors import UndefinedGreekError

CALL, PUT = OptionType.CALL, OptionType.PUT
EPS = float(np.finfo(np.float64).eps)
GREEKS = {
    "delta": bs.delta,
    "gamma": bs.gamma,
    "vega": bs.vega,
    "theta": bs.theta,
    "rho": bs.rho,
    "phi": bs.phi,
}

# Hull, Options, Futures, and Other Derivatives, Example 15.6 (SPEC 9.1 R1).
R1: dict[str, Any] = {"S": 42.0, "K": 40.0, "T": 0.5, "r": 0.10, "q": 0.0, "sigma": 0.20}

# Finite-difference check (SPEC 7 as a test tool), with steps near the predicted optimum
# eta ~ eps^(1/3) (FD_ETA = 1e-5 in experiments/validation.py). On this grid the predicted
# error, truncation h^2 V'''/6 plus rounding eps V / h, is ~1e-8 relative (worst observed:
# vega, 5e-8), so 1e-6 leaves 20x margin while any formula error (wrong sign, factor or
# exponent) shows up at 1e-3 or more. GREEK_FLOOR keeps the
# tolerance absolute (1e-9) for Greeks smaller than 1e-3.
#
# Gamma is checked as the S-derivative of analytic delta (itself checked against prices), not as
# a second difference of prices. That second difference has error h^2 gamma''/12 + 4 eps V / h^2;
# at S = 120, T = 0.25 the price (21) dwarfs gamma (0.0014), and even the best step reaches only
# ~6e-7 relative. A first version used it and failed there (docs/notes/phase_1_bsm.md).
FD_RTOL = 1e-6
GREEK_FLOOR = 1e-3

HALF_UNIT_2DP = 0.005  # anchor quoted to 2 decimals
HALF_UNIT_3DP = 0.0005  # anchor quoted to 3 decimals
# Identities: each side is a few roundings of terms no larger than `scale`.
IDENTITY_FACTOR = 64
# vega = S^2 sigma T gamma: two products of the same factors in a different order, ~8 roundings.
VEGA_GAMMA_RTOL = 16 * EPS
# Boundary Greeks: a few roundings of exp and products; 8 ulps covers them.
BOUNDARY_RTOL = 8 * EPS

# Interior grid and finite-difference routine: shared with the validation summary.
INTERIOR = GREEK_GRID


@pytest.mark.parametrize("option_type", [CALL, PUT])
@pytest.mark.parametrize("greek", list(GREEKS))
def test_analytic_greek_matches_finite_difference(option_type: OptionType, greek: str) -> None:
    # The bump is 1.00-unit based (S in currency, sigma, r, q as decimals, T in years), so
    # agreement also confirms the SPEC 2.2 units: vega and rho per 1.00, theta per year.
    failures = []
    for case in INTERIOR:
        analytic = float(GREEKS[greek](option_type, **case))
        fd = greek_finite_difference(greek, option_type, case)
        if abs(analytic - fd) > FD_RTOL * max(abs(analytic), GREEK_FLOOR):
            failures.append((case, analytic, fd))
    assert not failures, failures


def test_hull_sanity_anchors_theta_and_vega() -> None:
    # SPEC 3.3 sanity anchor for the R1 case: call theta about -4.559 per year, vega about 8.81.
    # Computed during spec review, not published values; tolerances are half a unit of the digits
    # quoted. QuantLib confirms them independently in the oracle tests.
    assert bs.theta(CALL, **R1) == pytest.approx(-4.559, abs=HALF_UNIT_3DP)
    assert bs.vega(CALL, **R1) == pytest.approx(8.81, abs=HALF_UNIT_2DP)


def test_call_put_greek_identities() -> None:
    # Differentiating put-call parity C - P = Sq - Kr (SPEC 9.2) term by term:
    #   delta_C - delta_P = exp(-qT); gamma and vega equal; theta_C - theta_P = q Sq - r Kr;
    #   rho_C - rho_P = T Kr; phi_C - phi_P = -T Sq.
    for case in INTERIOR:
        S, K, T, r, q = case["S"], case["K"], case["T"], case["r"], case["q"]
        Sq, Kr = S * math.exp(-q * T), K * math.exp(-r * T)
        expected = {
            "delta": math.exp(-q * T),
            "gamma": 0.0,
            "vega": 0.0,
            "theta": q * Sq - r * Kr,
            "rho": T * Kr,
            "phi": -T * Sq,
        }
        for greek, fn in GREEKS.items():
            call, put = float(fn(CALL, **case)), float(fn(PUT, **case))
            scale = max(abs(call), abs(put), abs(expected[greek]), 1.0)
            assert abs((call - put) - expected[greek]) <= IDENTITY_FACTOR * EPS * scale, (
                greek,
                case,
            )


def test_vega_gamma_relation() -> None:
    # vega = S^2 sigma T gamma, from the SPEC 3.3 rows: both contain exp(-qT) phi(d1).
    for case in INTERIOR:
        g, v = float(bs.gamma(CALL, **case)), float(bs.vega(CALL, **case))
        expected = case["S"] ** 2 * case["sigma"] * case["T"] * g
        assert v == pytest.approx(expected, rel=VEGA_GAMMA_RTOL)


# --- SPEC 3.4: boundary policy ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("option_type", "S", "expected"),
    [(CALL, 42.0, 1.0), (CALL, 38.0, 0.0), (PUT, 38.0, -1.0), (PUT, 42.0, 0.0)],
)
def test_expired_delta_is_signed_indicator(
    option_type: OptionType, S: float, expected: float
) -> None:
    # SPEC 3.4: delta at T = 0 is theta 1{theta (S - K) > 0}, K = 40.
    assert bs.delta(option_type, S=S, K=40.0, T=0.0, r=0.1, q=0.03, sigma=0.2) == expected


@pytest.mark.parametrize("option_type", [CALL, PUT])
def test_expired_at_the_money_delta_raises(option_type: OptionType) -> None:
    with pytest.raises(UndefinedGreekError, match="delta is undefined"):
        bs.delta(option_type, S=40.0, K=40.0, T=0.0, r=0.1, q=0.0, sigma=0.2)


@pytest.mark.parametrize("greek", ["gamma", "vega", "theta"])
@pytest.mark.parametrize("S", [38.0, 40.0, 42.0])
def test_gamma_vega_theta_raise_at_expiry_for_every_spot(greek: str, S: float) -> None:
    with pytest.raises(UndefinedGreekError, match=rf"{greek} is undefined at T = 0"):
        GREEKS[greek](CALL, S=S, K=40.0, T=0.0, r=0.1, q=0.0, sigma=0.2)


@pytest.mark.parametrize("greek", ["rho", "phi"])
@pytest.mark.parametrize("S", [38.0, 40.0, 42.0])
def test_rho_and_phi_are_zero_at_expiry(greek: str, S: float) -> None:
    # The expiry payoff max(theta (S - K), 0) does not depend on r or q.
    assert GREEKS[greek](PUT, S=S, K=40.0, T=0.0, r=0.1, q=0.03, sigma=0.2) == 0.0


@pytest.mark.parametrize("option_type", [CALL, PUT])
@pytest.mark.parametrize("S", [90.0, 110.0])
def test_zero_vol_greeks_are_the_deterministic_limit(option_type: OptionType, S: float) -> None:
    # SPEC 3.4: derivatives of D max(theta (F - K), 0) = max(theta (Sq - Kr), 0) away from F = K.
    K, T, r, q = 100.0, 0.75, 0.04, 0.01
    th = 1 if option_type is CALL else -1
    Sq, Kr = S * math.exp(-q * T), K * math.exp(-r * T)
    itm = float(th * (S * math.exp((r - q) * T) - K) > 0)
    expected = {
        "delta": th * math.exp(-q * T) * itm,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": th * (q * Sq - r * Kr) * itm,
        "rho": th * T * Kr * itm,
        "phi": -th * T * Sq * itm,
    }
    for greek, fn in GREEKS.items():
        value = fn(option_type, S=S, K=K, T=T, r=r, q=q, sigma=0.0)
        assert value == pytest.approx(expected[greek], rel=BOUNDARY_RTOL, abs=0.0), greek


@pytest.mark.parametrize("greek", list(GREEKS))
def test_zero_vol_greeks_raise_at_the_forward(greek: str) -> None:
    # r = q makes F = S = K exactly: the kink of the certain-forward payoff.
    with pytest.raises(UndefinedGreekError, match=rf"{greek} is undefined where"):
        GREEKS[greek](CALL, S=100.0, K=100.0, T=1.0, r=0.03, q=0.03, sigma=0.0)


@pytest.mark.parametrize("greek", list(GREEKS))
def test_small_vol_greeks_approach_zero_vol_limit_away_from_forward(greek: str) -> None:
    # Continuity (SPEC 3.4) for F != K: at s = 1e-6, d1 and d2 are about -1e5, so N and phi have
    # reached their limits exactly and the closed form equals the deterministic limit.
    case = {"S": 90.0, "K": 100.0, "T": 1.0, "r": 0.02, "q": 0.0}
    limit = GREEKS[greek](CALL, **case, sigma=0.0)
    assert GREEKS[greek](CALL, **case, sigma=1e-6) == pytest.approx(
        limit, rel=BOUNDARY_RTOL, abs=0.0
    )


def test_array_with_one_undefined_entry_raises_and_names_it() -> None:
    with pytest.raises(UndefinedGreekError, match=r"1 inputs; first at index \(2,\)"):
        bs.delta(CALL, S=np.array([38.0, 42.0, 40.0]), K=40.0, T=0.0, r=0.1, q=0.0, sigma=0.2)


def test_greeks_return_types_follow_spec_2_3() -> None:
    assert type(bs.gamma(CALL, **R1)) is float
    out = bs.gamma(CALL, **{**R1, "S": np.array([40.0, 42.0])})
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)
