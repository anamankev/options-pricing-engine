"""Published reference values (SPEC 9.1, R1 to R6), checked to the precision they were printed.

A value printed to n decimals pins the true value only within half a unit of its last digit, so
each tolerance is that half unit. Tighter claims come from the QuantLib and mpmath oracles.
"""

from typing import Any

import pytest

from pricing import binomial as bt
from pricing import black_scholes as bs
from pricing.contracts import ExerciseStyle, OptionType

HALF_UNIT_2DP = 0.005  # value printed to 2 decimals
HALF_UNIT_4DP = 0.00005  # value printed to 4 decimals

# R1: Hull, Options, Futures, and Other Derivatives, Example 15.6. Printed values verified (SPEC
# 9.1); edition and page still to record (SPEC 17).
R1: dict[str, Any] = {"S": 42.0, "K": 40.0, "T": 0.5, "r": 0.10, "q": 0.0, "sigma": 0.20}


def test_r1_hull_example_15_6_prices() -> None:
    assert bs.price(OptionType.CALL, **R1) == pytest.approx(4.76, abs=HALF_UNIT_2DP)
    assert bs.price(OptionType.PUT, **R1) == pytest.approx(0.81, abs=HALF_UNIT_2DP)


def test_r1_hull_example_15_6_d1_d2() -> None:
    k = bs._kernel(OptionType.CALL, **R1, allow_extreme_vol=False)
    assert float(k.d1) == pytest.approx(0.7693, abs=HALF_UNIT_4DP)
    assert float(k.d2) == pytest.approx(0.6278, abs=HALF_UNIT_4DP)


def test_r2_haug_put() -> None:
    # R2: Haug (2007), The Complete Guide to Option Pricing Formulas, 2nd ed., p. 4, as quoted in
    # the py_vollib doctest. Secondary source (SPEC 9.1, 17).
    value = bs.price(OptionType.PUT, S=100.0, K=95.0, T=0.5, r=0.10, q=0.05, sigma=0.20)
    assert value == pytest.approx(2.4648, abs=HALF_UNIT_4DP)


def test_r3_haug_call() -> None:
    # R3: Haug (2007), 2nd ed., pp. 2 to 3. Printed value not yet checked in the book (SPEC 17).
    value = bs.price(OptionType.CALL, S=60.0, K=65.0, T=0.25, r=0.08, q=0.0, sigma=0.30)
    assert value == pytest.approx(2.1334, abs=HALF_UNIT_4DP)


# --- American options: Ju and Zhong (1999), Journal of Derivatives 7(2) (SPEC 9.1 R4 to R6) ---
# Their "TRUE" column is a binomial tree with N = 10,000, printed to 3 decimals, so SPEC 9.1 sets
# the tolerance at 2e-3. Maturities are used exactly as printed (0.0833, not 1/12): the difference
# moves these prices by under 2e-4. Each value is reproduced by two independent methods.
JUZHONG_ABS_TOL = 2e-3


def _r4(K: float, T: float, sigma: float, value: float, tag: str = "R4") -> tuple[Any, ...]:
    # Exhibit 3: American puts, S = 40, r = 0.0488, q = 0.
    case = {"S": 40.0, "K": K, "T": T, "r": 0.0488, "q": 0.0, "sigma": sigma}
    return (tag, OptionType.PUT, case, value)


def _r5(S: float, sigma: float, r: float, q: float, value: float) -> tuple[Any, ...]:
    # Exhibit 4: American calls, K = 100, T = 0.5.
    case = {"S": S, "K": 100.0, "T": 0.5, "r": r, "q": q, "sigma": sigma}
    return ("R5", OptionType.CALL, case, value)


def _r6(S: float, value: float) -> tuple[Any, ...]:
    # Exhibit 5: American puts, K = 100, T = 3, sigma = 0.2, r = 0.08, q = 0.12.
    case = {"S": S, "K": 100.0, "T": 3.0, "r": 0.08, "q": 0.12, "sigma": 0.2}
    return ("R6", OptionType.PUT, case, value)


JUZHONG_CASES = [
    # R4, R5, R6: verified from the primary PDF (SPEC 9.1).
    _r4(35.0, 0.0833, 0.2, 0.006),
    _r4(40.0, 0.3333, 0.2, 1.580),
    _r4(40.0, 0.5833, 0.2, 1.990),
    _r4(45.0, 0.5833, 0.2, 5.267),
    _r4(45.0, 0.3333, 0.3, 5.706),
    _r4(40.0, 0.5833, 0.4, 4.353),
    _r4(45.0, 0.5833, 0.4, 7.383),
    # R4b: printed value not yet checked in the paper (SPEC 17).
    _r4(40.0, 0.0833, 0.2, 0.852, tag="R4b"),
    _r5(100.0, 0.2, 0.03, 0.07, 4.783),
    _r5(110.0, 0.2, 0.03, 0.07, 11.098),
    _r5(100.0, 0.4, 0.03, 0.07, 10.239),
    _r5(100.0, 0.3, 0.00, 0.07, 7.035),
    _r5(100.0, 0.3, 0.07, 0.03, 9.250),
    _r6(80.0, 25.658),
    _r6(90.0, 20.083),
    _r6(100.0, 15.498),
    _r6(110.0, 11.803),
    _r6(120.0, 8.886),
]


@pytest.mark.parametrize(
    ("tag", "option_type", "case", "published"),
    JUZHONG_CASES,
    ids=[f"{tag}-{ot.name}-S{c['S']:g}-K{c['K']:g}" for tag, ot, c, _ in JUZHONG_CASES],
)
def test_ju_zhong_american_values(
    tag: str, option_type: OptionType, case: dict[str, float], published: float
) -> None:
    lr = bt.leisen_reimer(option_type, **case, n_steps=2001, exercise=ExerciseStyle.AMERICAN)
    bbsr = bt.bbsr(option_type, **case, n_steps=2048, exercise=ExerciseStyle.AMERICAN)
    assert lr.value == pytest.approx(published, abs=JUZHONG_ABS_TOL)
    assert bbsr.value == pytest.approx(published, abs=JUZHONG_ABS_TOL)
