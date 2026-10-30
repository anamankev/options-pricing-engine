"""Reconciliation against QuantLib (SPEC 8.1, 9.2). Needs the `oracle` extra.

QuantLib is imported when a test runs (pricing.oracles is imported lazily by the experiment
code), not when this file is collected, so the fast CI job, which does not install the extra, can
still collect it. In the full suite a missing QuantLib fails loudly with ImportError.
"""

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

from pricing import black_scholes as bs
from pricing.contracts import OptionType

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
