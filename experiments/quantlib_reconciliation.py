"""QuantLib reconciliation of European BSM prices and Greeks (SPEC 8.1).

Pre-registered in docs/notes/phase_1_quantlib_reconciliation.md. Run from the repository root:

    python -m experiments.quantlib_reconciliation \
        --config experiments/configs/quantlib_reconciliation.toml

Writes one CSV row per case and a JSON summary that records the config, commit hash, library
versions and platform. Deterministic: no random numbers are used.

Post-hoc diagnosis, added after the first run and not pre-registered: wherever the two
implementations differ by more than the relative tolerance, both are scored against the exact
value from mpmath at 50 significant digits, and each difference is also expressed in units of
eps times the quantity's natural scale.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mpmath
import numpy as np

from experiments.provenance import environment
from pricing import black_scholes as bs
from pricing.contracts import DAYS_PER_YEAR, OptionType

QUANTITIES = ("price", "delta", "gamma", "vega", "theta", "rho", "phi")
LIBRARIES = ("numpy", "scipy", "QuantLib", "mpmath")
EPS = float(np.finfo(np.float64).eps)
MP_DIGITS = 50


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def iter_cases(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    g = config["grid"]
    for option_type, S, days, r, q, sigma in itertools.product(
        g["option_types"], g["S"], g["days"], g["r"], g["q"], g["sigma"]
    ):
        yield {
            "option_type": option_type,
            "S": S,
            "K": g["K"],
            "days": days,
            "r": r,
            "q": q,
            "sigma": sigma,
        }


def relative_error(ours: float, reference: float) -> float:
    """|ours - reference| / |reference|; 0 if both are exactly 0, inf if only reference is."""
    if reference == 0.0:
        return 0.0 if ours == 0.0 else math.inf
    return abs(ours - reference) / abs(reference)


def natural_scale(name: str, case: dict[str, Any]) -> float:
    """The size each quantity has when it is not small: S for price and theta, 1 for delta,
    1/S for gamma, S sqrt(T) for vega, K T for rho and S T for phi."""
    S, K, T = case["S"], case["K"], case["days"] / DAYS_PER_YEAR
    scales = {
        "price": S,
        "delta": 1.0,
        "gamma": 1.0 / S,
        "vega": S * math.sqrt(T),
        "theta": S,
        "rho": K * T,
        "phi": S * T,
    }
    return scales[name]


def compare_case(case: dict[str, Any]) -> dict[str, Any]:
    """Our price and Greeks against QuantLib for one case: values, errors and |d1|."""
    from pricing.oracles import quantlib_european  # needs the oracle extra; imported on use

    option_type = OptionType[case["option_type"]]
    T = case["days"] / DAYS_PER_YEAR
    market = {k: case[k] for k in ("S", "K", "r", "q", "sigma")}
    reference = quantlib_european(option_type, **market, days=case["days"])
    if reference.T != T:
        raise RuntimeError(f"QuantLib used T={reference.T!r}, expected {T!r}")
    s = case["sigma"] * math.sqrt(T)
    d1 = math.log(case["S"] * math.exp((case["r"] - case["q"]) * T) / case["K"]) / s + s / 2
    row: dict[str, Any] = {**case, "T": T, "abs_d1": abs(d1)}
    for name in QUANTITIES:
        ours = float(getattr(bs, name)(option_type, **market, T=T))
        ref = float(getattr(reference, name))
        row |= {
            f"{name}_ours": ours,
            f"{name}_quantlib": ref,
            f"{name}_abs_err": abs(ours - ref),
            f"{name}_abs_err_eps_scale": abs(ours - ref) / (EPS * natural_scale(name, case)),
            f"{name}_rel_err": relative_error(ours, ref),
        }
    return row


def exact_value(name: str, case: dict[str, Any]) -> mpmath.mpf:
    """SPEC 3.1 and 3.3 evaluated by mpmath at 50 digits on the exact float inputs."""
    th = 1 if case["option_type"] == "CALL" else -1
    with mpmath.workdps(MP_DIGITS):
        S, K, r, q, sigma = (mpmath.mpf(case[k]) for k in ("S", "K", "r", "q", "sigma"))
        T = mpmath.mpf(case["days"] / DAYS_PER_YEAR)
        Sq, Kr = S * mpmath.exp(-q * T), K * mpmath.exp(-r * T)
        s = sigma * mpmath.sqrt(T)
        d1 = mpmath.log(Sq / Kr) / s + s / 2
        d2 = d1 - s
        N, pdf = mpmath.ncdf, mpmath.npdf
        values = {
            "price": th * (Sq * N(th * d1) - Kr * N(th * d2)),
            "delta": th * mpmath.exp(-q * T) * N(th * d1),
            "gamma": mpmath.exp(-q * T) * pdf(d1) / (S * s),
            "vega": Sq * pdf(d1) * mpmath.sqrt(T),
            "theta": -Sq * pdf(d1) * sigma / (2 * mpmath.sqrt(T))
            - th * r * Kr * N(th * d2)
            + th * q * Sq * N(th * d1),
            "rho": th * T * Kr * N(th * d2),
            "phi": -th * T * Sq * N(th * d1),
        }
        return values[name]


def _rel_to_exact(value: float, exact: mpmath.mpf) -> float:
    with mpmath.workdps(MP_DIGITS):
        return float(abs(mpmath.mpf(value) - exact) / abs(exact))


def summarize(rows: list[dict[str, Any]], rtol: float, small_d1: float) -> dict[str, Any]:
    """Per quantity: error distribution, exceedances, the worst case, and (post hoc) both
    implementations scored against mpmath wherever they differ by more than rtol."""
    summary: dict[str, Any] = {}
    for name in QUANTITIES:
        rel = np.array([row[f"{name}_rel_err"] for row in rows])
        worst = rows[int(np.argmax(rel))]
        over = [row for row in rows if row[f"{name}_rel_err"] > rtol]
        vs_exact = [
            (
                _rel_to_exact(row[f"{name}_ours"], exact_value(name, row)),
                _rel_to_exact(row[f"{name}_quantlib"], exact_value(name, row)),
            )
            for row in over
        ]
        summary[name] = {
            "max_rel_err": _json_float(float(rel.max())),
            "p99_rel_err": _json_float(float(np.percentile(rel, 99))),
            "median_rel_err": _json_float(float(np.median(rel))),
            "max_abs_err": max(row[f"{name}_abs_err"] for row in rows),
            "count_over_rtol": len(over),
            "count_over_rtol_with_small_d1": sum(row["abs_d1"] < small_d1 for row in over),
            "min_abs_d1_over_rtol": min((row["abs_d1"] for row in over), default=None),
            "max_abs_err_eps_scale_over_rtol": max(
                (row[f"{name}_abs_err_eps_scale"] for row in over), default=0.0
            ),
            "over_rtol_ours_max_rel_vs_exact": max((o for o, _ in vs_exact), default=0.0),
            "over_rtol_quantlib_max_rel_vs_exact": max((ql for _, ql in vs_exact), default=0.0),
            "worst_case": {k: worst[k] for k in ("option_type", "S", "days", "r", "q", "sigma")}
            | {
                "abs_d1": worst["abs_d1"],
                "ours": worst[f"{name}_ours"],
                "quantlib": worst[f"{name}_quantlib"],
            },
        }
    return summary


def _json_float(x: float) -> float | str:
    return x if math.isfinite(x) else str(x)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    env = environment(LIBRARIES)

    rows = [compare_case(case) for case in iter_cases(config)]
    summary = summarize(rows, config["relative_tolerance"], config["small_d1"])

    cases_csv = Path(config["output"]["cases_csv"])
    cases_csv.parent.mkdir(parents=True, exist_ok=True)
    with cases_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {"config": config, "environment": env, "n_cases": len(rows), "summary": summary}
    Path(config["output"]["summary_json"]).write_text(json.dumps(record, indent=2) + "\n")

    print(f"{len(rows)} cases; relative tolerance {config['relative_tolerance']:g}; ", end="")
    print(f"commit {env['commit'][:7]}{' (dirty)' if env['dirty'] else ''}")
    header = ("quantity", "max rel", "p99 rel", "median", ">rtol", "min|d1|", "eps*scl")
    print("{:8s} {:>9s} {:>9s} {:>9s} {:>6s} {:>8s} {:>8s}".format(*header), end="")
    print(" | vs mpmath: ours, QuantLib")
    for name, s in summary.items():
        min_d1 = s["min_abs_d1_over_rtol"]
        print(
            f"{name:8s} {_fmt(s['max_rel_err'])} {_fmt(s['p99_rel_err'])} "
            f"{_fmt(s['median_rel_err'])} {s['count_over_rtol']:>6d} "
            f"{'-' if min_d1 is None else f'{min_d1:.1f}':>8s} "
            f"{s['max_abs_err_eps_scale_over_rtol']:>8.2f} | "
            f"{s['over_rtol_ours_max_rel_vs_exact']:.1e}, "
            f"{s['over_rtol_quantlib_max_rel_vs_exact']:.1e}"
        )
    return 0


def _fmt(x: float | str) -> str:
    return f"{x:>9s}" if isinstance(x, str) else f"{x:>9.1e}"


if __name__ == "__main__":
    raise SystemExit(main())
