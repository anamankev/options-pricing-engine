"""Reconciliation against QuantLib (SPEC 8.1, 9.2). Needs the `oracle` extra.

QuantLib is imported when a test runs (pricing.oracles is imported lazily by the experiment
code), not when this file is collected, so the fast CI job, which does not install the extra, can
still collect it. In the full suite a missing QuantLib fails loudly with ImportError.
"""

import dataclasses
import itertools
import math
from pathlib import Path

import mpmath
import pytest
from experiments.quantlib_reconciliation import (
    QUANTITIES,
    compare_case,
    exact_value,
    iter_cases,
    load_config,
)

from pricing import binomial as bt
from pricing import black_scholes as bs
from pricing.contracts import ExerciseStyle, OptionType

pytestmark = pytest.mark.oracle

CONFIG = Path(__file__).resolve().parents[1] / "experiments/configs/quantlib_reconciliation.toml"

# Justified in docs/notes/phase_1_quantlib_reconciliation.md (1800-case run at commit 8a00ed2).
# The two closed forms agree to relative 1e-11 (median observed 2e-16 to 1e-15) except in far
# tails (|d1| >= 3.2), where QuantLib loses relative precision: scored against mpmath it is up to
# 100% off there, while ours stays within 2.5e-10. In those cases the absolute difference is at
# most 0.65 eps times the quantity's natural scale, so a case passes on either test; the absolute
# limit of 4 leaves a 6x margin. Our own tail accuracy is held to relative precision by the
# mpmath tests in test_black_scholes.py.
ORACLE_RTOL = 1e-11
ORACLE_ABS_EPS_SCALE = 4.0

EPS = 2.220446049250313e-16
# LBR on a clean R1 price: Jaeckel (2015) reports full double-precision accuracy; the price itself
# carries a few eps of rounding, so 64 eps relative is a generous check of the convention.
LBR_R1_RTOL = 64 * EPS
# Phase 1 price rounding model factor (tests/test_black_scholes.py).
VOLLIB_ROUNDING_FACTOR = 64

HALF_UNIT_2DP = 0.005  # anchor quoted to 2 decimals
HALF_UNIT_3DP = 0.0005  # anchor quoted to 3 decimals
# Far-tail relative accuracy of ours: d^2 amplification at |d1| = 9.2 is ~85 eps; 1e-12 is
# ~50x that, and QuantLib's error at the same point is ~1 (reconciliation note, section 4).
TAIL_RTOL = 1e-12


def test_reconciliation_grid_agrees_with_quantlib() -> None:
    cases = list(iter_cases(load_config(CONFIG)))
    assert len(cases) >= 500  # SPEC 8.1
    failures = [
        (case, name, row[f"{name}_ours"], row[f"{name}_quantlib"])
        for case in cases
        for row in [compare_case(case)]
        for name in QUANTITIES
        if row[f"{name}_rel_err"] > ORACLE_RTOL
        and row[f"{name}_abs_err_eps_scale"] > ORACLE_ABS_EPS_SCALE
    ]
    assert not failures, failures[:10]


def test_quantlib_greek_units_match_spec_on_r1() -> None:
    # H2. SPEC 3.3 sanity anchor: R1 call vega about 8.81 per 1.00 of sigma and theta about
    # -4.559 per year. Per-point vega would read 0.0881 and per-day theta -0.0125.
    from pricing.oracles import quantlib_european

    ql = quantlib_european(
        OptionType.CALL, S=42.0, K=40.0, r=0.10, q=0.0, sigma=0.20, days=180, year_days=360
    )
    assert ql.T == 0.5  # Actual360, 180 days: exactly the R1 maturity
    assert ql.vega == pytest.approx(8.81, abs=HALF_UNIT_2DP)
    assert ql.theta == pytest.approx(-4.559, abs=HALF_UNIT_3DP)


@pytest.mark.parametrize(
    ("option_type", "case", "days", "digits", "expected"),
    [
        # R1, Hull Example 15.6: unrounded values from spec review, confirmed here (SPEC 9.1).
        (OptionType.CALL, {"S": 42.0, "K": 40.0, "r": 0.10, "q": 0.0}, 180, 6, 4.759422),
        (OptionType.PUT, {"S": 42.0, "K": 40.0, "r": 0.10, "q": 0.0}, 180, 6, 0.808599),
        # R2, Haug (2007) p. 4 via the py_vollib doctest; R3, Haug (2007) pp. 2 to 3.
        (OptionType.PUT, {"S": 100.0, "K": 95.0, "r": 0.10, "q": 0.05}, 180, 4, 2.4648),
        (OptionType.CALL, {"S": 60.0, "K": 65.0, "r": 0.08, "q": 0.0}, 90, 4, 2.1334),
    ],
)
def test_reference_values_reproduced_by_quantlib(
    option_type: OptionType, case: dict[str, float], days: int, digits: int, expected: float
) -> None:
    # H4. Actual360 makes T = 0.5 (180 days) and 0.25 (90 days) exact. R1 and R2 use sigma = 0.20,
    # R3 uses 0.30.
    from pricing.oracles import quantlib_european

    sigma = 0.30 if days == 90 else 0.20
    ql = quantlib_european(option_type, **case, sigma=sigma, days=days, year_days=360)
    ours = float(bs.price(option_type, **case, sigma=sigma, T=days / 360))
    assert round(ql.price, digits) == expected
    assert ours == pytest.approx(ql.price, rel=ORACLE_RTOL)


def test_far_tail_put_delta_stays_accurate_where_quantlib_does_not() -> None:
    # Documents the oracle's limitation found in the reconciliation: a deep out-of-the-money put
    # (|d1| = 9.2). Expected value: computed by mpmath at 50 significant digits.
    case = {"option_type": "PUT", "S": 130.0, "K": 100.0, "days": 30, "r": 0.0, "q": 0.0}
    case["sigma"] = 0.10
    row = compare_case(case)
    exact = exact_value("delta", case)
    with mpmath.workdps(50):
        ours_rel = float(abs(mpmath.mpf(row["delta_ours"]) - exact) / abs(exact))
    assert ours_rel <= TAIL_RTOL
    # Precondition: if a QuantLib release fixes its tail precision, this fails, and
    # ORACLE_ABS_EPS_SCALE above should be revisited.
    assert row["delta_rel_err"] > ORACLE_RTOL


def test_quantlib_american_engine_is_available() -> None:
    # SPEC 8.1: Study B's oracle; the first oracle test must fail loudly if it is absent.
    import QuantLib as ql

    assert hasattr(ql, "QdFpAmericanEngine")


# --- Trees against QuantLib's BinomialVanillaEngine (SPEC 8.1) ----------------------------

# 192 cases: S in {90, 100, 110}, K = 100, 30 or 365 days, (r, q) in {(0.05, 0.02), (-0.01, 0.03)},
# sigma in {0.2, 0.4}, N in {101, 401}, call and put, European and American.
TREE_GRID = list(
    itertools.product(
        (90.0, 100.0, 110.0),
        (30, 365),
        ((0.05, 0.02), (-0.01, 0.03)),
        (0.2, 0.4),
        (101, 401),
        (OptionType.CALL, OptionType.PUT),
        (ExerciseStyle.EUROPEAN, ExerciseStyle.AMERICAN),
    )
)
# Same algorithm, two implementations. Observed worst relative difference on TREE_GRID: 8.4e-12,
# the same at N = 101 and N = 401, so it comes from how each library evaluates the step
# parameters, not from rounding accumulated over steps. 1e-10 leaves 12x margin; any algorithmic
# difference (probability convention, an off-by-one step) shows up at 1e-9 or more.
TREE_ORACLE_RTOL = 1e-10


def _quantlib_crr_probability(T: float, r: float, q: float, sigma: float, n: int) -> float:
    # QuantLib's CoxRossRubinstein tree: pu = 1/2 + (1/2) (r - q - sigma^2/2) dt / (sigma sqrt(dt)).
    return 0.5 + 0.5 * (r - q - 0.5 * sigma**2) * math.sqrt(T / n) / sigma


def test_leisen_reimer_matches_quantlib_lr() -> None:
    from pricing.oracles import quantlib_tree

    failures = []
    for S, days, (r, q), sigma, n, option_type, exercise in TREE_GRID:
        market = {"S": S, "K": 100.0, "r": r, "q": q, "sigma": sigma}
        theirs = quantlib_tree(
            option_type, **market, days=days, n_steps=n, tree="lr", exercise=exercise
        )
        ours = bt.leisen_reimer(option_type, **market, T=days / 365, n_steps=n, exercise=exercise)
        if abs(ours.value / theirs - 1) > TREE_ORACLE_RTOL:
            failures.append((S, days, r, q, sigma, n, option_type, exercise, ours.value, theirs))
    assert not failures, failures[:5]


def test_tree_engine_reproduces_quantlib_crr_given_its_probability() -> None:
    # SPEC 8.1 expected our CRR to match QuantLib's "near machine precision". It does not, because
    # QuantLib's CRR uses a log-drift approximation for the up-probability where SPEC 4.1 uses the
    # martingale probability (g - d) / (u - d). Feeding QuantLib's probability into our backward
    # induction reproduces QuantLib, which isolates the convention as the only difference.
    from pricing.oracles import quantlib_tree

    failures, smallest_gap = [], math.inf
    for S, days, (r, q), sigma, n, option_type, exercise in TREE_GRID:
        market = {"S": S, "K": 100.0, "r": r, "q": q, "sigma": sigma}
        T = days / 365
        theirs = quantlib_tree(
            option_type, **market, days=days, n_steps=n, tree="crr", exercise=exercise
        )
        lattice = bt._crr_lattice(option_type, S, 100.0, T, r, q, sigma, n, False)
        lattice = dataclasses.replace(lattice, p=_quantlib_crr_probability(T, r, q, sigma, n))
        ours, _ = bt._roll_back(
            lattice, bt._payoff(lattice, bt._spots(lattice, n)), n, exercise, False
        )
        if abs(ours / theirs - 1) > TREE_ORACLE_RTOL:
            failures.append((S, days, r, q, sigma, n, option_type, exercise, ours, theirs))
        own_crr = bt.crr(option_type, **market, T=T, n_steps=n, exercise=exercise).value
        smallest_gap = min(smallest_gap, abs(own_crr / theirs - 1))
    assert not failures, failures[:5]
    assert smallest_gap > 10 * TREE_ORACLE_RTOL  # with SPEC's probability the prices do differ


# --- American reference for Study B: QdFp (SPEC 8.1, 11, 17) -------------------------------

# Ju and Zhong (1999) cases with the nearest representable maturities: Actual360 with 30, 120 and
# 210 days gives T = 1/12, 1/3, 7/12 for the printed 0.0833, 0.3333, 0.5833, which moves these
# prices by at most 1.6e-4 (measured with LR, N = 2001). Tolerance as in SPEC 9.1.
JUZHONG_ABS_TOL = 2e-3
QDFP_CASES = [
    # (type, S, K, r, q, sigma, days, year_days, published)
    (OptionType.PUT, 40.0, 35.0, 0.0488, 0.0, 0.2, 30, 360, 0.006),
    (OptionType.PUT, 40.0, 40.0, 0.0488, 0.0, 0.2, 120, 360, 1.580),
    (OptionType.PUT, 40.0, 40.0, 0.0488, 0.0, 0.2, 210, 360, 1.990),
    (OptionType.PUT, 40.0, 45.0, 0.0488, 0.0, 0.2, 210, 360, 5.267),
    (OptionType.PUT, 40.0, 45.0, 0.0488, 0.0, 0.3, 120, 360, 5.706),
    (OptionType.PUT, 40.0, 40.0, 0.0488, 0.0, 0.4, 210, 360, 4.353),
    (OptionType.PUT, 40.0, 45.0, 0.0488, 0.0, 0.4, 210, 360, 7.383),
    (OptionType.PUT, 40.0, 40.0, 0.0488, 0.0, 0.2, 30, 360, 0.852),
    (OptionType.CALL, 100.0, 100.0, 0.03, 0.07, 0.2, 180, 360, 4.783),
    (OptionType.CALL, 110.0, 100.0, 0.03, 0.07, 0.2, 180, 360, 11.098),
    (OptionType.CALL, 100.0, 100.0, 0.03, 0.07, 0.4, 180, 360, 10.239),
    (OptionType.CALL, 100.0, 100.0, 0.00, 0.07, 0.3, 180, 360, 7.035),
    (OptionType.CALL, 100.0, 100.0, 0.07, 0.03, 0.3, 180, 360, 9.250),
    (OptionType.PUT, 80.0, 100.0, 0.08, 0.12, 0.2, 1095, 365, 25.658),
    (OptionType.PUT, 90.0, 100.0, 0.08, 0.12, 0.2, 1095, 365, 20.083),
    (OptionType.PUT, 100.0, 100.0, 0.08, 0.12, 0.2, 1095, 365, 15.498),
    (OptionType.PUT, 110.0, 100.0, 0.08, 0.12, 0.2, 1095, 365, 11.803),
    (OptionType.PUT, 120.0, 100.0, 0.08, 0.12, 0.2, 1095, 365, 8.886),
]
# BBSR at N = 16384 against QdFp: observed at most 1.5e-6 on seven of these cases; BBSR's error at
# that N is O(1e-6) (Phase 2 measurement), so 1e-5 checks agreement with margin.
QDFP_BBSR_ABS_TOL = 1e-5


def test_qdfp_reproduces_ju_zhong() -> None:
    # SPEC 11: confirm the Study B reference agrees with Ju-Zhong within 2e-3 before using it.
    from pricing.oracles import quantlib_american

    for option_type, S, K, r, q, sigma, days, year_days, published in QDFP_CASES:
        value = quantlib_american(
            option_type, S=S, K=K, r=r, q=q, sigma=sigma, days=days, year_days=year_days
        )
        assert value == pytest.approx(published, abs=JUZHONG_ABS_TOL), (S, K, sigma, days)


@pytest.mark.slow
def test_qdfp_agrees_with_bbsr_at_large_n() -> None:
    from pricing.oracles import quantlib_american

    for option_type, S, K, r, q, sigma, days, year_days, _ in QDFP_CASES[::3]:
        market = {"S": S, "K": K, "r": r, "q": q, "sigma": sigma}
        reference = quantlib_american(option_type, **market, days=days, year_days=year_days)
        tree = bt.bbsr(
            option_type,
            **market,
            T=days / year_days,
            n_steps=16384,
            exercise=ExerciseStyle.AMERICAN,
        )
        assert tree.value == pytest.approx(reference, abs=QDFP_BBSR_ABS_TOL), (S, K, sigma, days)


# --- Let's Be Rational (SPEC 6.3, 8.2) ------------------------------------------------------


def test_lbr_signature_and_price_convention() -> None:
    # SPEC 6.3: confirm the signature and conventions before relying on the oracle.
    import inspect

    from lets_be_rational import implied_volatility_from_a_transformed_rational_guess as lbr

    from pricing.oracles import lbr_implied_volatility

    assert list(inspect.signature(lbr).parameters) == ["price", "F", "K", "T", "q"]
    # R1 (Hull, Example 15.6): sigma = 0.20. LBR wants the undiscounted price on the forward.
    S, K, T, r = 42.0, 40.0, 0.5, 0.10
    discounted = float(bs.price(OptionType.CALL, S=S, K=K, T=T, r=r, q=0.0, sigma=0.20))
    F, D = S * math.exp(r * T), math.exp(-r * T)
    sigma = lbr_implied_volatility(OptionType.CALL, discounted / D, F=F, K=K, T=T)
    assert sigma == pytest.approx(0.20, rel=LBR_R1_RTOL)
    # Passing the discounted price instead gives a different volatility: the convention matters.
    wrong = lbr_implied_volatility(OptionType.CALL, discounted, F=F, K=K, T=T)
    assert abs(wrong - 0.20) > 0.01
    # q = -1 for puts.
    put = float(bs.price(OptionType.PUT, S=S, K=K, T=T, r=r, q=0.0, sigma=0.20))
    assert lbr_implied_volatility(OptionType.PUT, put / D, F=F, K=K, T=T) == pytest.approx(
        0.20, rel=LBR_R1_RTOL
    )


def test_lbr_rejects_infeasible_prices_with_our_exception() -> None:
    from pricing.errors import InfeasiblePriceError
    from pricing.oracles import lbr_implied_volatility

    with pytest.raises(InfeasiblePriceError, match="Let's Be Rational rejected"):
        lbr_implied_volatility(OptionType.CALL, 15.0, F=100.0, K=80.0, T=1.0)  # below intrinsic


def test_lbr_and_safeguarded_newton_recover_the_round_trip_grid() -> None:
    # Both solvers against the true total volatility on SPEC 6.4's grid, with the accuracy model of
    # tests/test_implied_vol.py; points at the lower bound are not well posed and are skipped.
    from pricing import implied_vol as iv
    from pricing.oracles import lbr_implied_volatility
    from test_implied_vol import ROUND_TRIP, _allowed_error, _market

    checked = 0
    for x, T, sigma, option_type in ROUND_TRIP:
        market = _market(float(x), T)
        price = float(bs.price(option_type, **market, sigma=sigma))
        ours = iv.implied_volatility_result(option_type, price, **market)
        if ours.reason is not iv.IVReason.CONVERGED:
            continue
        F = market["S"] * math.exp((market["r"] - market["q"]) * T)
        c = price / math.exp(-market["r"] * T)
        s_lbr = lbr_implied_volatility(option_type, c, F=F, K=market["K"], T=T) * math.sqrt(T)
        s = sigma * math.sqrt(T)
        allowed = _allowed_error(option_type, float(x), s, iv.IVMethod.SAFEGUARDED_NEWTON)
        assert abs(s_lbr - s) <= allowed, (x, T, sigma, option_type)
        checked += 1
    assert checked > 250  # 306 of 520 are well posed


# --- vollib as a second BSM implementation (SPEC 8.2) -------------------------------------


def test_vollib_prices_match_ours_within_the_rounding_budget() -> None:
    # Both implementations carry rounding error up to ROUNDING_FACTOR eps x scale (Phase 1 model),
    # so they may differ by twice that. Observed worst relative difference: 5e-12, at a 1e-29 price.
    from pricing.oracles import vollib_price

    for S, T, r, q, sigma, option_type in itertools.product(
        (70.0, 90.0, 100.0, 110.0, 130.0),
        (0.1, 1.0, 5.0),
        (-0.01, 0.05),
        (0.0, 0.03),
        (0.1, 0.4, 1.0),
        (OptionType.CALL, OptionType.PUT),
    ):
        market = {"S": S, "K": 100.0, "T": T, "r": r, "q": q, "sigma": sigma}
        ours = float(bs.price(option_type, **market))
        theirs = vollib_price(option_type, **market)
        Sq, Kr = S * math.exp(-q * T), 100.0 * math.exp(-r * T)
        s = sigma * math.sqrt(T)
        d1 = math.log(Sq / Kr) / s + s / 2
        scale = Sq + Kr + Sq * math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi) * (2 + 2 * abs(d1))
        assert abs(ours - theirs) <= 2 * VOLLIB_ROUNDING_FACTOR * EPS * scale, market
