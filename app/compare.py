"""What the playground shows, computed without any user interface, so it can be tested.

Every comparison uses the same tolerance as the test suite: a value agrees with QuantLib when
the relative difference is at most 1e-11, or the absolute difference is at most 4 eps times the
quantity's natural scale (tests/test_oracles.py explains both).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from typing import Any, Literal

from experiments.quantlib_reconciliation import QUANTITIES, relative_error

from pricing import binomial as bt
from pricing import black_scholes as bs
from pricing import implied_vol as iv
from pricing.contracts import ExerciseStyle, OptionType
from pricing.errors import InfeasiblePriceError, PricingInputError
from pricing.monte_carlo import MCMethod, monte_carlo

ORACLE_RTOL = 1e-11  # same as tests/test_oracles.py
ORACLE_ABS_EPS_SCALE = 4.0  # same as tests/test_oracles.py
EPS = sys.float_info.epsilon
AMERICAN_REFERENCE_SCHEME = (20, 60, 1e-14)  # Study B's validated QdFp setting (SPEC 8.1)
NOT_INSTALLED = 'QuantLib not installed: pip install -e ".[oracle]"'


@dataclass(frozen=True)
class Contract:
    option_type: OptionType
    S: float
    K: float
    days: int  # maturity in days, as QuantLib takes it; T = days / year_days
    r: float
    q: float
    sigma: float
    year_days: Literal[360, 365] = 365

    @property
    def T(self) -> float:
        return self.days / self.year_days

    def market(self) -> dict[str, float]:
        return {
            "S": self.S,
            "K": self.K,
            "T": self.T,
            "r": self.r,
            "q": self.q,
            "sigma": self.sigma,
        }


def quantlib_installed() -> bool:
    return importlib.util.find_spec("QuantLib") is not None


@dataclass(frozen=True)
class Row:
    quantity: str
    ours: float
    quantlib: float | None
    difference: float | None  # |ours - QuantLib|
    agrees: bool | None  # None when QuantLib is not installed


def natural_scale(quantity: str, c: Contract) -> float:
    """The size each quantity has when it is not small (as in the QuantLib reconciliation)."""
    scales = {
        "price": c.S,
        "delta": 1.0,
        "gamma": 1.0 / c.S,
        "vega": c.S * math.sqrt(c.T),
        "theta": c.S,
        "rho": c.K * c.T,
        "phi": c.S * c.T,
    }
    return scales[quantity]


def price_and_greeks(c: Contract) -> list[Row]:
    """European price and six Greeks: ours beside QuantLib's AnalyticEuropeanEngine."""
    ours = {q: float(getattr(bs, q)(c.option_type, **c.market())) for q in QUANTITIES}
    if not quantlib_installed():
        return [Row(q, ours[q], None, None, None) for q in QUANTITIES]
    from pricing.oracles import quantlib_european

    theirs = quantlib_european(
        c.option_type,
        S=c.S,
        K=c.K,
        r=c.r,
        q=c.q,
        sigma=c.sigma,
        days=c.days,
        year_days=c.year_days,
    )
    if theirs.T != c.T:
        raise RuntimeError(f"QuantLib used T={theirs.T!r}, expected {c.T!r}")
    rows = []
    for q in QUANTITIES:
        ref = float(getattr(theirs, q))
        difference = abs(ours[q] - ref)
        agrees = relative_error(
            ours[q], ref
        ) <= ORACLE_RTOL or difference <= ORACLE_ABS_EPS_SCALE * EPS * natural_scale(q, c)
        rows.append(Row(q, ours[q], ref, difference, agrees))
    return rows


TREE_METHODS = ("CRR", "Leisen-Reimer", "BBS", "BBSR")


def tree_price(method: str, c: Contract, n_steps: int, exercise: ExerciseStyle) -> float:
    m = c.market()
    if method == "CRR":
        return bt.crr(c.option_type, **m, n_steps=n_steps, exercise=exercise).value
    if method == "Leisen-Reimer":
        odd = n_steps if n_steps % 2 else n_steps + 1  # LR needs an odd step count
        return bt.leisen_reimer(c.option_type, **m, n_steps=odd, exercise=exercise).value
    if method == "BBS":
        return bt.bbs(c.option_type, **m, n_steps=n_steps, exercise=exercise).value
    even = n_steps if n_steps % 2 == 0 else n_steps + 1  # BBSR needs an even step count
    return bt.bbsr(c.option_type, **m, n_steps=even, exercise=exercise).value


def tree_reference(c: Contract, exercise: ExerciseStyle) -> tuple[float | None, str]:
    """The exact European price, or QuantLib's high-accuracy American price (Study B)."""
    if exercise is ExerciseStyle.EUROPEAN:
        return float(bs.price(c.option_type, **c.market())), "exact Black-Scholes price"
    if not quantlib_installed():
        return None, NOT_INSTALLED
    from pricing.oracles import quantlib_american

    value = quantlib_american(
        c.option_type,
        S=c.S,
        K=c.K,
        r=c.r,
        q=c.q,
        sigma=c.sigma,
        days=c.days,
        year_days=c.year_days,
        scheme=AMERICAN_REFERENCE_SCHEME,
    )
    return value, "QuantLib QdFp American engine, strict setting"


def tree_convergence(
    c: Contract, exercise: ExerciseStyle, steps: list[int]
) -> dict[str, list[float]]:
    return {m: [tree_price(m, c, n, exercise) for n in steps] for m in TREE_METHODS}


@dataclass(frozen=True)
class MonteCarloCheck:
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    exact: float
    covered: bool


def monte_carlo_check(c: Contract, n_paths: int, seed: int, method: MCMethod) -> MonteCarloCheck:
    pilot = max(100, n_paths // 10) if method is MCMethod.CONTROL_VARIATE else None
    res = monte_carlo(
        c.option_type, **c.market(), n_paths=n_paths, seed=seed, method=method, pilot_paths=pilot
    )
    exact = float(bs.price(c.option_type, **c.market()))
    return MonteCarloCheck(
        res.estimate,
        res.std_error,
        res.ci_low,
        res.ci_high,
        exact,
        res.ci_low <= exact <= res.ci_high,
    )


def monte_carlo_coverage(c: Contract, n_paths: int, method: MCMethod, seeds: int) -> float:
    """Share of `seeds` independent runs (seeds 0 .. seeds-1) whose 95% interval covers the exact
    price. It should be close to 0.95."""
    covered = sum(monte_carlo_check(c, n_paths, seed, method).covered for seed in range(seeds))
    return covered / seeds


@dataclass(frozen=True)
class SolverRow:
    solver: str
    sigma: float | None
    iterations: int | None
    stop_reason: str
    error: float | None  # |sigma - true sigma| when the price came from a known sigma


def implied_vol_check(c: Contract, price: float, true_sigma: float | None) -> list[SolverRow]:
    """Back out sigma from `price` with each solver (and Let's Be Rational, if installed)."""
    market = {"S": c.S, "K": c.K, "T": c.T, "r": c.r, "q": c.q}
    rows = []
    for method in (iv.IVMethod.SAFEGUARDED_NEWTON, iv.IVMethod.BRENT, iv.IVMethod.NAIVE_NEWTON):
        try:
            res = iv.implied_volatility_result(c.option_type, price, **market, method=method)
        except (InfeasiblePriceError, PricingInputError) as exc:
            rows.append(SolverRow(method.value, None, None, f"rejected: {exc}", None))
            continue
        error = abs(res.sigma - true_sigma) if res.sigma is not None and true_sigma else None
        rows.append(SolverRow(method.value, res.sigma, res.iterations, res.reason.value, error))
    if quantlib_installed():
        from pricing.oracles import lbr_implied_volatility

        F = c.S * math.exp((c.r - c.q) * c.T)
        undiscounted = price * math.exp(c.r * c.T)
        try:
            sigma = lbr_implied_volatility(c.option_type, undiscounted, F=F, K=c.K, T=c.T)
            error = abs(sigma - true_sigma) if true_sigma else None
            rows.append(SolverRow("lets_be_rational", sigma, 2, "converged", error))
        except InfeasiblePriceError as exc:
            rows.append(SolverRow("lets_be_rational", None, None, f"rejected: {exc}", None))
    return rows


def describe(value: Any, digits: int = 10) -> str:
    """A number as the playground prints it."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)
