"""Independent reference values from QuantLib (SPEC 8.1). Needs the `oracle` extra.

QuantLib (https://www.quantlib.org) is an independent C++ implementation, used here only to
cross-check. Settings follow SPEC 8.1: a fixed evaluation date; maturity = evaluation date + n
days; Actual365Fixed, so T = n / 365 exactly; flat continuously compounded curves for r and q; a
constant Black volatility; BlackScholesMertonProcess. Actual360 is available for published cases
that need T = 0.5 or 0.25 exactly (180 or 90 days).

- quantlib_european: AnalyticEuropeanEngine price and Greeks. Units match SPEC 2.2 (confirmed in
  docs/notes/phase_1_quantlib_reconciliation.md, H2): vega per 1.00 of sigma, theta per year,
  rho and phi (QuantLib's dividendRho) per 1.00.
- quantlib_tree: BinomialVanillaEngine with the "crr" or "lr" tree. QuantLib's LR uses the same
  formulas as SPEC 4.2. Its CRR does not: its up-probability is the log-drift approximation
  1/2 + (1/2) (r - q - sigma^2 / 2) sqrt(dt) / sigma, not SPEC 4.1's martingale probability
  (g - d) / (u - d), so the two CRR prices differ by O(1/N) (found in Phase 2; see
  tests/test_oracles.py).
- quantlib_american: QdFpAmericanEngine (Andersen, Lake and Offengenden 2016). By default it uses
  QuantLib's highPrecisionScheme(), which Phase 5 found accurate only to about 6e-8 on the Ju-Zhong
  cases. Passing scheme=(m, n, eps) selects QdFpTanhSinhIterationScheme(m, n, eps); Study B uses
  (20, 60, 1e-14), which agrees with (40, 100, 1e-15) within 7.5e-10 (SPEC 8.1, 11).
- lbr_implied_volatility: Jaeckel's "Let's Be Rational" (Jaeckel 2015) through lets_be_rational
  (SPEC 6.3, 8.2): undiscounted Black price, forward, strike, T, and q = +1 call, -1 put; returns
  sigma. The signature and price convention are pinned by tests/test_oracles.py.
- vollib_price: vollib's Black-Scholes-Merton price, a second implementation (SPEC 8.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import QuantLib as ql
from lets_be_rational import implied_volatility_from_a_transformed_rational_guess
from lets_be_rational.exceptions import AboveMaximumException, BelowIntrinsicException
from vollib.black_scholes_merton import black_scholes_merton

from pricing.contracts import ExerciseStyle, OptionType, option_sign
from pricing.errors import InfeasiblePriceError

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
    option, process, day_counter, maturity = _setup(
        option_type, S, K, r, q, sigma, days, year_days, ExerciseStyle.EUROPEAN
    )
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


def quantlib_tree(
    option_type: OptionType,
    *,
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    days: int,
    n_steps: int,
    tree: Literal["crr", "lr"],
    exercise: ExerciseStyle,
    year_days: Literal[360, 365] = 365,
) -> float:
    """Price from QuantLib's BinomialVanillaEngine with `n_steps` steps of the given tree."""
    option, process, _, _ = _setup(option_type, S, K, r, q, sigma, days, year_days, exercise)
    option.setPricingEngine(ql.BinomialVanillaEngine(process, tree, n_steps))
    return float(option.NPV())


def quantlib_american(
    option_type: OptionType,
    *,
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    days: int,
    year_days: Literal[360, 365] = 365,
    scheme: tuple[int, int, float] | None = None,
) -> float:
    """American price from QdFpAmericanEngine: QuantLib's high-precision scheme, or
    QdFpTanhSinhIterationScheme(m, n, eps) when scheme = (m, n, eps) is given."""
    option, process, _, _ = _setup(
        option_type, S, K, r, q, sigma, days, year_days, ExerciseStyle.AMERICAN
    )
    if scheme is None:
        qdfp_scheme = ql.QdFpAmericanEngine.highPrecisionScheme()
    else:
        qdfp_scheme = ql.QdFpTanhSinhIterationScheme(*scheme)
    option.setPricingEngine(ql.QdFpAmericanEngine(process, qdfp_scheme))
    return float(option.NPV())


def lbr_implied_volatility(
    option_type: OptionType, c: float, *, F: float, K: float, T: float
) -> float:
    """Implied volatility from Let's Be Rational for an undiscounted Black price c (SPEC 6.3)."""
    q = option_sign(option_type)
    try:
        sigma = implied_volatility_from_a_transformed_rational_guess(c, F, K, T, q)
    except (BelowIntrinsicException, AboveMaximumException) as exc:
        raise InfeasiblePriceError(f"Let's Be Rational rejected price {c!r}: {exc}") from exc
    return float(sigma)


def vollib_price(
    option_type: OptionType, *, S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """vollib's Black-Scholes-Merton price (SPEC 8.2)."""
    flag = "c" if option_sign(option_type) == 1 else "p"
    return float(black_scholes_merton(flag, S, K, T, r, sigma, q))


def _setup(
    option_type: OptionType,
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    days: int,
    year_days: Literal[360, 365],
    exercise: ExerciseStyle,
) -> tuple[Any, Any, Any, Any]:
    """Option, Black-Scholes-Merton process, day counter and maturity date (SPEC 8.1)."""
    sign = option_sign(option_type)
    if days <= 0:
        raise ValueError(f"days must be > 0; got {days!r}")
    day_counter = ql.Actual365Fixed() if year_days == 365 else ql.Actual360()
    ql.Settings.instance().evaluationDate = EVALUATION_DATE
    maturity = EVALUATION_DATE + days

    def flat_curve(rate: float) -> Any:
        curve = ql.FlatForward(EVALUATION_DATE, rate, day_counter, ql.Continuous)
        return ql.YieldTermStructureHandle(curve)

    volatility = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(EVALUATION_DATE, ql.NullCalendar(), sigma, day_counter)
    )
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)), flat_curve(q), flat_curve(r), volatility
    )
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if sign == 1 else ql.Option.Put, K)
    if exercise is ExerciseStyle.AMERICAN:
        exercise_rule = ql.AmericanExercise(EVALUATION_DATE, maturity)
    else:
        exercise_rule = ql.EuropeanExercise(maturity)
    return ql.VanillaOption(payoff, exercise_rule), process, day_counter, maturity
