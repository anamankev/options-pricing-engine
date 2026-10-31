"""The playground (app/): its comparisons, and the page itself rendered headless."""

import math
from pathlib import Path

import pytest
from app import compare
from app.compare import Contract

from pricing import binomial as bt
from pricing import black_scholes as bs
from pricing.contracts import ExerciseStyle, OptionType
from pricing.monte_carlo import MCMethod

CALL, PUT = OptionType.CALL, OptionType.PUT
# Hull, Options, Futures, and Other Derivatives, Example 15.6 (SPEC 9.1 R1): 180 days on a
# 360-day year is T = 0.5 exactly, the setting the QuantLib R1 test uses.
R1 = Contract(CALL, 42.0, 40.0, 180, 0.10, 0.0, 0.20, 360)
# Ju and Zhong (1999), Exhibit 3, R4c: American put, published 1.990 (SPEC 9.1). T = 0.5833 years
# is 213 days on a 365-day year to within 0.0002 years.
R4C = Contract(PUT, 40.0, 40.0, 213, 0.0488, 0.0, 0.20, 365)
JUZHONG_ABS_TOL = 2e-3  # SPEC 9.1: the published values are binomial approximations to 3 dp
# A volatility solved back from a clean price: rounding of the price over vega, a few eps at R1.
ROUND_TRIP_ABS = 1e-12
PLAYGROUND = Path(__file__).resolve().parents[1] / "app" / "playground.py"


def test_r1_price_is_hulls_example() -> None:
    price = next(row for row in compare.price_and_greeks(R1) if row.quantity == "price")
    # Expected: 4.759422392871536, the R1 value of tests/test_reference_values.py at T = 0.5.
    assert price.ours == bs.price(CALL, S=42.0, K=40.0, T=0.5, r=0.10, q=0.0, sigma=0.20)
    assert round(price.ours, 2) == 4.76  # Hull prints 4.76


@pytest.mark.oracle
def test_r1_price_and_greeks_agree_with_quantlib() -> None:
    rows = compare.price_and_greeks(R1)
    assert [row.quantity for row in rows] == list(compare.QUANTITIES)
    assert all(row.agrees for row in rows), rows


def test_without_quantlib_the_comparison_is_empty_not_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compare, "quantlib_installed", lambda: False)
    rows = compare.price_and_greeks(R1)
    assert all(row.quantlib is None and row.agrees is None for row in rows)
    reference, source = compare.tree_reference(R4C, ExerciseStyle.AMERICAN)
    assert reference is None
    assert "not installed" in source


def test_tree_prices_are_the_library_functions_with_valid_step_counts() -> None:
    m = R4C.market()
    american = ExerciseStyle.AMERICAN
    values = compare.tree_convergence(R4C, american, [100])
    assert values["CRR"] == [bt.crr(PUT, **m, n_steps=100, exercise=american).value]
    assert values["Leisen-Reimer"] == [
        bt.leisen_reimer(PUT, **m, n_steps=101, exercise=american).value
    ]
    assert values["BBS"] == [bt.bbs(PUT, **m, n_steps=100, exercise=american).value]
    assert values["BBSR"] == [bt.bbsr(PUT, **m, n_steps=100, exercise=american).value]
    # Odd N is rounded up to an even count for BBSR, which needs N / 2 steps.
    assert compare.tree_price("BBSR", R4C, 101, american) == compare.tree_price(
        "BBSR", R4C, 102, american
    )


def test_european_tree_reference_is_the_exact_price() -> None:
    reference, _ = compare.tree_reference(R1, ExerciseStyle.EUROPEAN)
    assert reference == bs.price(CALL, **R1.market())


@pytest.mark.oracle
def test_american_reference_reproduces_the_published_value() -> None:
    reference, _ = compare.tree_reference(R4C, ExerciseStyle.AMERICAN)
    assert reference is not None
    assert abs(reference - 1.990) <= JUZHONG_ABS_TOL


def test_monte_carlo_check_reports_its_interval_consistently() -> None:
    mc = compare.monte_carlo_check(R1, 10_000, 1, MCMethod.ANTITHETIC)
    assert mc.exact == bs.price(CALL, **R1.market())
    assert mc.ci_low < mc.estimate < mc.ci_high
    assert mc.covered == (mc.ci_low <= mc.exact <= mc.ci_high)


def test_implied_vol_round_trip_returns_the_input_volatility() -> None:
    price = float(bs.price(CALL, **R1.market()))
    rows = {row.solver: row for row in compare.implied_vol_check(R1, price, 0.20)}
    for solver in ("safeguarded_newton", "brent"):
        assert rows[solver].sigma is not None
        assert abs(rows[solver].sigma - 0.20) <= ROUND_TRIP_ABS


def test_a_price_below_intrinsic_is_rejected_with_the_reason() -> None:
    intrinsic = R1.S - R1.K * math.exp(-R1.r * R1.T)
    rows = compare.implied_vol_check(R1, intrinsic - 0.5, None)
    ours = [row for row in rows if row.solver != "lets_be_rational"]
    assert all(row.sigma is None and row.stop_reason.startswith("rejected") for row in ours)


def test_playground_page_renders_without_errors() -> None:
    pytest.importorskip("streamlit")  # the optional `app` extra
    from streamlit.testing.v1 import AppTest

    page = AppTest.from_file(str(PLAYGROUND), default_timeout=120).run()
    assert not page.exception, page.exception
    assert len(page.tabs) == 4
    page.button[0].click().run()  # Monte Carlo: coverage over many seeds
    assert not page.exception, page.exception
    assert any(m.label.startswith("Coverage over") for m in page.metric)


# 200 runs: the coverage estimate's standard error is sqrt(0.95 x 0.05 / 200) = 0.015; 4 of them.
COVERAGE_SEEDS = 200
COVERAGE_BAND = 4 * math.sqrt(0.95 * 0.05 / COVERAGE_SEEDS)


@pytest.mark.statistical
def test_monte_carlo_coverage_over_many_seeds_is_near_95_percent() -> None:
    share = compare.monte_carlo_coverage(R1, 10_000, MCMethod.PLAIN, COVERAGE_SEEDS)
    assert abs(share - 0.95) <= COVERAGE_BAND
