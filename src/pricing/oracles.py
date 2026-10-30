"""Independent reference values from QuantLib (SPEC 8.1). Needs the `oracle` extra.

QuantLib (https://www.quantlib.org) is an independent C++ implementation, used here only to
cross-check. Settings follow SPEC 8.1: a fixed evaluation date; maturity = evaluation date + n
days; Actual365Fixed, so T = n / 365 exactly; flat continuously compounded curves for r and q; a
constant Black volatility; BlackScholesMertonProcess; AnalyticEuropeanEngine. Actual360 is
available for published cases that need T = 0.5 or 0.25 exactly (180 or 90 days).

Units of the returned Greeks match SPEC 2.2 (confirmed in
docs/notes/phase_1_quantlib_reconciliation.md, H2): vega per 1.00 of sigma, theta per year,
rho and phi (QuantLib's dividendRho) per 1.00.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import QuantLib as ql

from pricing.contracts import OptionType, option_sign

EVALUATION_DATE = ql.Date(30, ql.October, 2026)  # fixed, so results do not depend on today


@dataclass(frozen=True)
class QuantLibEuropean:
    """QuantLib's analytic European price and Greeks for one contract."""

    T: float  # year fraction QuantLib used
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    phi: float


def quantlib_european(
    option_type: OptionType,
    *,
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    days: int,
    year_days: Literal[360, 365] = 365,
) -> QuantLibEuropean:
    """Price and Greeks from QuantLib's AnalyticEuropeanEngine, maturity `days` after
    EVALUATION_DATE, with T = days / year_days."""
    sign = option_sign(option_type)
    if days <= 0:
        raise ValueError(f"days must be > 0; got {days!r}")
    day_counter = ql.Actual365Fixed() if year_days == 365 else ql.Actual360()
    ql.Settings.instance().evaluationDate = EVALUATION_DATE
    maturity = EVALUATION_DATE + days

    def flat_curve(rate: float) -> ql.YieldTermStructureHandle:
        curve = ql.FlatForward(EVALUATION_DATE, rate, day_counter, ql.Continuous)
        return ql.YieldTermStructureHandle(curve)

    volatility = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(EVALUATION_DATE, ql.NullCalendar(), sigma, day_counter)
    )
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)), flat_curve(q), flat_curve(r), volatility
    )
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if sign == 1 else ql.Option.Put, K)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(maturity))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return QuantLibEuropean(
        T=float(day_counter.yearFraction(EVALUATION_DATE, maturity)),
        price=float(option.NPV()),
        delta=float(option.delta()),
        gamma=float(option.gamma()),
        vega=float(option.vega()),
        theta=float(option.theta()),
        rho=float(option.rho()),
        phi=float(option.dividendRho()),
    )
