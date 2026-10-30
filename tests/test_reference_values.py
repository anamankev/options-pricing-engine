"""Published reference values (SPEC 9.1, R1 to R3), checked to the precision they were printed.

A value printed to n decimals pins the true value only within half a unit of its last digit, so
each tolerance is that half unit. Tighter claims come from the QuantLib and mpmath oracles.
"""

from typing import Any

import pytest

from pricing import black_scholes as bs
from pricing.contracts import OptionType

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
