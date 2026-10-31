"""Validation summary: measures every number the README's Validation section states.

Run from the repository root:

    python -m experiments.validation --config experiments/configs/validation.toml

The tests check these properties against tolerances; this script records the measured values
themselves (worst errors, case counts), with provenance, in
experiments/results/tables/validation_summary.json. docs/claims.toml points each README number at
a key in that file. The grids and cases are defined here once and imported by the tests, so a
README number and the test that guards it always describe the same cases.

Sections that need the `oracle` extra (QuantLib) are recorded as skipped when it is not installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import itertools
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any

import mpmath
import numpy as np
from scipy.special import ndtr

from experiments.mp_reference import DIGITS, mp_price_and_error_scale
from experiments.provenance import environment
from pricing import binomial as bt
from pricing import black_scholes as bs
from pricing import implied_vol as iv
from pricing.contracts import ExerciseStyle, OptionType
from pricing.finite_diff import central_first, relative_step

LIBRARIES = ("numpy", "scipy", "mpmath")
EPS = sys.float_info.epsilon
CALL, PUT = OptionType.CALL, OptionType.PUT
AMERICAN = ExerciseStyle.AMERICAN

# --- BSM against mpmath (SPEC 9.2): 144 markets x call and put --------------------------------
# A grid spanning moneyness, maturity, volatility and signed r and q, with total volatility
# s >= 0.03 so every price is well away from underflow.
GRID_S = (70.0, 95.0, 100.0, 130.0)
GRID_T = (0.1, 1.0, 5.0)
GRID_R = (-0.01, 0.05)
GRID_Q = (-0.02, 0.03)
GRID_SIGMA = (0.1, 0.4, 1.0)
GRID_K = 100.0
MPMATH_GRID = [
    {"S": S, "K": GRID_K, "T": T, "r": r, "q": q, "sigma": sigma}
    for S, T, r, q, sigma in itertools.product(GRID_S, GRID_T, GRID_R, GRID_Q, GRID_SIGMA)
]

# --- Greeks against finite differences: smooth interior points spanning SPEC 9.2 ---------------
GREEK_GRID = [
    {"S": S, "K": 100.0, "T": T, "r": r, "q": q, "sigma": sigma}
    for S, T, r, q, sigma in itertools.product(
        (80.0, 100.0, 120.0), (0.25, 1.0, 3.0), (-0.01, 0.05), (0.0, 0.03), (0.15, 0.4)
    )
]
GREEK_FUNCTIONS = ("delta", "gamma", "vega", "theta", "rho", "phi")
FD_ETA = 1e-5  # step near the predicted optimum eta ~ eps^(1/3) (SPEC 7)
GREEK_FLOOR = 1e-3  # relative error uses max(|greek|, floor), i.e. absolute below 1e-3

# --- Implied-volatility round trips: the SPEC 6.4 grid -----------------------------------------
IV_R, IV_Q, IV_STRIKE = 0.03, 0.01, 100.0
IV_GRID = list(
    itertools.product(
        np.arange(-3.0, 3.01, 0.5),  # log-moneyness x = ln(F / K)
        (1 / 365, 7 / 365, 0.25, 1.0, 5.0),
        (0.01, 0.2, 1.0, 3.0),
        (CALL, PUT),
    )
)


def iv_market(x: float, T: float) -> dict[str, float]:
    """Market with log-moneyness x = ln(F / K) for K = 100."""
    F = IV_STRIKE * math.exp(x)
    return {"S": F * math.exp(-(IV_R - IV_Q) * T), "K": IV_STRIKE, "T": T, "r": IV_R, "q": IV_Q}


def iv_error_unit(option_type: OptionType, x: float, s: float) -> float:
    """eps x (price rounding scale) / vega: the total-vol error that rounding alone explains."""
    th = 1 if option_type is CALL else -1
    F = IV_STRIKE * math.exp(x)
    d1 = x / s + s / 2
    d2 = d1 - s
    terms = F * float(ndtr(th * d1)) + IV_STRIKE * float(ndtr(th * d2))
    vega = F * math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
    return EPS * (terms + vega * (1 + abs(d1) + abs(d2))) / vega


# --- Trees against QuantLib's trees (oracle) ---------------------------------------------------
TREE_GRID = list(
    itertools.product(
        (90.0, 100.0, 110.0),
        (30, 365),  # days
        ((0.05, 0.02), (-0.01, 0.03)),  # (r, q)
        (0.2, 0.4),
        (101, 401),
        (CALL, PUT),
        (ExerciseStyle.EUROPEAN, AMERICAN),
    )
)


def quantlib_crr_probability(T: float, r: float, q: float, sigma: float, n: int) -> float:
    """QuantLib's CoxRossRubinstein up-probability: 1/2 + (1/2)(r - q - sigma^2/2) dt / (sigma
    sqrt(dt)), a log-drift approximation (SPEC 8.1), not SPEC 4.1's martingale probability."""
    return 0.5 + 0.5 * (r - q - 0.5 * sigma**2) * math.sqrt(T / n) / sigma


def crr_with_quantlib_probability(
    option_type: OptionType, market: dict[str, float], T: float, n: int, exercise: ExerciseStyle
) -> float:
    """Our backward induction run with QuantLib's CRR probability substituted."""
    m = market
    lattice = bt._crr_lattice(option_type, m["S"], m["K"], T, m["r"], m["q"], m["sigma"], n, False)
    lattice = dataclasses.replace(
        lattice, p=quantlib_crr_probability(T, m["r"], m["q"], m["sigma"], n)
    )
    value, _ = bt._roll_back(
        lattice, bt._payoff(lattice, bt._spots(lattice, n)), n, exercise, False
    )
    return value


# --- Ju and Zhong (1999), Journal of Derivatives 7(2), Exhibits 3 to 5 (SPEC 9.1 R4 to R6) ----
# Their "TRUE" column is a binomial tree with N = 10,000, printed to 3 decimals. Maturities are
# used exactly as printed (0.0833, not 1/12).
def _r4(K: float, T: float, sigma: float, value: float, tag: str = "R4") -> tuple[Any, ...]:
    # Exhibit 3: American puts, S = 40, r = 0.0488, q = 0.
    case = {"S": 40.0, "K": K, "T": T, "r": 0.0488, "q": 0.0, "sigma": sigma}
    return (tag, PUT, case, value)


def _r5(S: float, sigma: float, r: float, q: float, value: float) -> tuple[Any, ...]:
    # Exhibit 4: American calls, K = 100, T = 0.5.
    case = {"S": S, "K": 100.0, "T": 0.5, "r": r, "q": q, "sigma": sigma}
    return ("R5", CALL, case, value)


def _r6(S: float, value: float) -> tuple[Any, ...]:
    # Exhibit 5: American puts, K = 100, T = 3, sigma = 0.2, r = 0.08, q = 0.12.
    case = {"S": S, "K": 100.0, "T": 3.0, "r": 0.08, "q": 0.12, "sigma": 0.2}
    return ("R6", PUT, case, value)


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
JUZHONG_LR_STEPS = 2001
JUZHONG_BBSR_STEPS = 2048


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def oracle_available() -> bool:
    return importlib.util.find_spec("QuantLib") is not None


def bsm_vs_mpmath() -> dict[str, Any]:
    worst = 0.0
    for option_type, case in itertools.product((CALL, PUT), MPMATH_GRID):
        th = 1 if option_type is CALL else -1
        exact, scale = mp_price_and_error_scale(th, **case)
        worst = max(worst, abs(bs.price(option_type, **case) - exact) / (EPS * scale))
    return {"cases": 2 * len(MPMATH_GRID), "worst_error_eps_scale": worst}


def greek_finite_difference(greek: str, option_type: OptionType, case: dict[str, float]) -> float:
    """Central difference of price (or of delta, for gamma) in the Greek's variable (SPEC 7)."""

    def bumped(name: str, fn: Any = bs.price) -> float:
        h = relative_step(case[name], eta=FD_ETA, x_scale=1.0)
        return central_first(lambda x: float(fn(option_type, **{**case, name: x})), case[name], h)

    if greek == "gamma":
        return bumped("S", bs.delta)
    if greek == "theta":
        return -bumped("T")  # SPEC 7: theta = -dV/dT
    return bumped({"delta": "S", "vega": "sigma", "rho": "r", "phi": "q"}[greek])


def greeks_vs_finite_differences() -> dict[str, Any]:
    worst: dict[str, float] = {}
    for greek, option_type, case in itertools.product(GREEK_FUNCTIONS, (CALL, PUT), GREEK_GRID):
        analytic = float(getattr(bs, greek)(option_type, **case))
        fd = greek_finite_difference(greek, option_type, case)
        error = abs(analytic - fd) / max(abs(analytic), GREEK_FLOOR)
        worst[greek] = max(worst.get(greek, 0.0), error)
    return {
        "cases_per_greek": 2 * len(GREEK_GRID),
        "worst_relative_error": worst,
        "worst_relative_error_any": max(worst.values()),
    }


def iv_round_trips() -> dict[str, Any]:
    out: dict[str, Any] = {"cases": len(IV_GRID)}
    for method in (iv.IVMethod.SAFEGUARDED_NEWTON, iv.IVMethod.BRENT):
        reasons: dict[str, int] = {}
        worst, value_without_convergence = 0.0, 0
        for x, T, sigma, option_type in IV_GRID:
            market = iv_market(float(x), T)
            price = float(bs.price(option_type, **market, sigma=sigma))
            res = iv.implied_volatility_result(option_type, price, **market, method=method)
            reasons[res.reason.name] = reasons.get(res.reason.name, 0) + 1
            if (res.sigma is not None) != res.converged:
                value_without_convergence += 1
            if res.reason is iv.IVReason.CONVERGED and res.total_vol is not None:
                s = sigma * math.sqrt(T)
                error = abs(res.total_vol - s)
                if method is iv.IVMethod.BRENT:  # remove its stopping tolerance, then normalize
                    error = max(0.0, error - iv.BRENT_XTOL - iv.CONVERGENCE_FACTOR * EPS * s)
                worst = max(worst, error / iv_error_unit(option_type, float(x), s))
        out[method.value] = {
            "reasons": reasons,
            "worst_error_in_rounding_units": worst,
            "value_returned_without_convergence": value_without_convergence,
        }
    return out


def juzhong() -> dict[str, Any]:
    worst_lr = worst_bbsr = 0.0
    for _, option_type, case, published in JUZHONG_CASES:
        lr = bt.leisen_reimer(option_type, **case, n_steps=JUZHONG_LR_STEPS, exercise=AMERICAN)
        bbsr = bt.bbsr(option_type, **case, n_steps=JUZHONG_BBSR_STEPS, exercise=AMERICAN)
        worst_lr = max(worst_lr, abs(lr.value - published))
        worst_bbsr = max(worst_bbsr, abs(bbsr.value - published))
    return {
        "cases": len(JUZHONG_CASES),
        "lr_steps": JUZHONG_LR_STEPS,
        "bbsr_steps": JUZHONG_BBSR_STEPS,
        "lr_worst_abs_difference": worst_lr,
        "bbsr_worst_abs_difference": worst_bbsr,
        "worst_abs_difference": max(worst_lr, worst_bbsr),
    }


def trees_vs_quantlib() -> dict[str, Any]:
    from pricing.oracles import quantlib_tree

    worst_lr = worst_crr = 0.0
    for S, days, (r, q), sigma, n, option_type, exercise in TREE_GRID:
        market = {"S": S, "K": 100.0, "r": r, "q": q, "sigma": sigma}
        T = days / 365
        ql_lr = quantlib_tree(
            option_type, **market, days=days, n_steps=n, tree="lr", exercise=exercise
        )
        ours_lr = bt.leisen_reimer(option_type, **market, T=T, n_steps=n, exercise=exercise)
        worst_lr = max(worst_lr, abs(ours_lr.value / ql_lr - 1))
        ql_crr = quantlib_tree(
            option_type, **market, days=days, n_steps=n, tree="crr", exercise=exercise
        )
        ours_crr = crr_with_quantlib_probability(option_type, market, T, n, exercise)
        worst_crr = max(worst_crr, abs(ours_crr / ql_crr - 1))
    return {
        "cases": len(TREE_GRID),
        "lr_worst_relative_difference": worst_lr,
        "crr_with_quantlib_probability_worst_relative_difference": worst_crr,
    }


def bump_refined_optimum(bumps_config: dict[str, Any], bumps_summary: Path) -> dict[str, Any]:
    """Observed optimal FD step against the error model's own optimum (post hoc, bump study):
    h* = (3 eps |V| / |V'''|)^(1/3) for first derivatives, (48 eps |V| / |V''''|)^(1/4) for gamma,
    with V's higher derivatives from mpmath; eta* = h* / max(|x|, x_scale)."""
    m = bumps_config["market"]
    th = 1 if bumps_config["option_type"] == "CALL" else -1
    x_scale = bumps_config["x_scale"]
    variable = {"delta": "S", "gamma": "S", "vega": "sigma", "theta": "T", "rho": "r"}
    observed = json.loads(bumps_summary.read_text())["per_greek"]

    def price(**overrides: Any) -> Any:
        p = {**{k: mpmath.mpf(v) for k, v in m.items()}, **overrides}
        F = p["S"] * mpmath.exp((p["r"] - p["q"]) * p["T"])
        s = p["sigma"] * mpmath.sqrt(p["T"])
        d1 = mpmath.log(F / p["K"]) / s + s / 2
        d2 = d1 - s
        disc = mpmath.exp(-p["r"] * p["T"])
        return th * disc * (F * mpmath.ncdf(th * d1) - p["K"] * mpmath.ncdf(th * d2))

    out: dict[str, Any] = {}
    with mpmath.workdps(DIGITS):
        V = price()
        for greek, name in variable.items():
            order = 4 if greek == "gamma" else 3
            derivative = mpmath.diff(lambda v, n=name: price(**{n: v}), mpmath.mpf(m[name]), order)
            if greek == "gamma":
                h = (48 * EPS * abs(V) / abs(derivative)) ** mpmath.mpf(0.25)
            else:
                h = (3 * EPS * abs(V) / abs(derivative)) ** (mpmath.mpf(1) / 3)
            refined = float(h) / max(abs(m[name]), x_scale)
            ratio = observed[greek]["optimal_eta"] / refined
            out[greek] = {
                "refined_eta": refined,
                "observed_eta": observed[greek]["optimal_eta"],
                "observed_over_refined": ratio,
            }
        out["price"] = float(V)
    out["worst_factor"] = max(
        max(v["observed_over_refined"], 1 / v["observed_over_refined"])
        for k, v in out.items()
        if k != "price"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    env = environment(LIBRARIES)

    record: dict[str, Any] = {"config": config, "environment": env}
    record["bsm_vs_mpmath"] = bsm_vs_mpmath()
    record["greeks_vs_finite_differences"] = greeks_vs_finite_differences()
    record["iv_round_trips"] = iv_round_trips()
    record["juzhong"] = juzhong()
    record["bump_refined_optimum"] = bump_refined_optimum(
        load_config(Path(config["bump_study"]["config"])), Path(config["bump_study"]["summary"])
    )
    if oracle_available():
        record["trees_vs_quantlib"] = trees_vs_quantlib()
    else:
        record["trees_vs_quantlib"] = {"skipped": "QuantLib not installed (oracle extra)"}

    out = Path(config["output"]["summary_json"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"commit {env['commit'][:7]}{' (dirty)' if env['dirty'] else ''}")
    for section in (
        "bsm_vs_mpmath",
        "greeks_vs_finite_differences",
        "juzhong",
        "trees_vs_quantlib",
    ):
        print(section, json.dumps(record[section]))
    print("iv_round_trips", json.dumps(record["iv_round_trips"]))
    print("bump_refined_optimum worst factor", record["bump_refined_optimum"]["worst_factor"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
