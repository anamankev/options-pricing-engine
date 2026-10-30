"""Monte Carlo interval coverage, bias and error scaling (SPEC 5.5).

Pre-registered in docs/notes/phase_2_mc_coverage.md. Run from the repository root:

    python -m experiments.mc_coverage --config experiments/configs/mc_coverage.toml

For every (method, strike, n) cell it prices the same call R times with independent seeds, then
records coverage of the true BSM price, bias, RMSE, and how well the reported standard error
matches the spread of the estimates. It also fits log(SE) against log(n) and computes variance
reduction factors from the empirical variance of the estimates. The registered checks (note,
section 2) are evaluated here and written with the results.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from experiments.provenance import environment
from pricing import black_scholes as bs
from pricing.contracts import OptionType
from pricing.monte_carlo import MCMethod, monte_carlo

LIBRARIES = ("numpy", "scipy")
COVERAGE_LEVEL = 0.95


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def pilot_size(config: dict[str, Any], n: int) -> int:
    return max(config["pilot_minimum"], n // config["pilot_divisor"])


def run_cell(config: dict[str, Any], i: int, j: int, k: int) -> dict[str, Any]:
    """R independent replications of one (method i, strike j, path count k) cell."""
    market = dict(config["market"])
    option_type = OptionType[market.pop("option_type")]
    method = MCMethod(config["methods"][i])
    K, n, R = config["strikes"][j], config["n_paths"][k], config["replications"]
    pilot = pilot_size(config, n) if method is MCMethod.CONTROL_VARIATE else None
    truth = float(bs.price(option_type, **market, K=K))

    estimates, errors, covered, cost = np.empty(R), np.empty(R), 0, 0
    for rep in range(R):
        seed = np.random.SeedSequence(config["root_seed"], spawn_key=(i, j, k, rep))
        res = monte_carlo(
            option_type, **market, K=K, n_paths=n, seed=seed, method=method, pilot_paths=pilot
        )
        estimates[rep], errors[rep] = res.estimate, res.std_error
        covered += res.ci_low <= truth <= res.ci_high
        cost = res.n_payoff_evals
    spread = float(estimates.std(ddof=1))
    coverage = covered / R
    return {
        "method": method.value,
        "K": K,
        "n_paths": n,
        "pilot_paths": pilot or 0,
        "cost": cost,
        "replications": R,
        "truth": truth,
        "coverage": coverage,
        "coverage_se_at_95": math.sqrt(COVERAGE_LEVEL * (1 - COVERAGE_LEVEL) / R),
        "coverage_se_observed": math.sqrt(coverage * (1 - coverage) / R),
        "bias": float(estimates.mean()) - truth,
        "bias_se": spread / math.sqrt(R),
        "rmse": float(np.sqrt(np.mean((estimates - truth) ** 2))),
        "sd_estimates": spread,
        "mean_reported_se": float(errors.mean()),
        "calibration_ratio": float(errors.mean()) / spread,
    }


def fit_slope(n: list[int], se: list[float]) -> tuple[float, float]:
    """Least-squares slope of log(se) on log(n) and its standard error."""
    x, y = np.log(np.asarray(n, dtype=float)), np.log(np.asarray(se))
    xc = x - x.mean()
    slope = float(xc @ (y - y.mean()) / (xc @ xc))
    residuals = y - y.mean() - slope * xc
    dof = len(x) - 2
    slope_se = math.sqrt(float(residuals @ residuals) / dof / float(xc @ xc)) if dof > 0 else 0.0
    return slope, slope_se


def analyse(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Scaling fits, variance reduction factors and the registered checks C1 to C5."""
    checks = config["checks"]
    cell = {(r["method"], r["K"], r["n_paths"]): r for r in rows}

    scaling = {}
    for method, K in itertools.product(config["methods"], config["strikes"]):
        ns = config["n_paths"]
        slope, slope_se = fit_slope(ns, [cell[(method, K, n)]["mean_reported_se"] for n in ns])
        scaling[f"{method}/K={K:g}"] = {"slope": slope, "slope_se": slope_se}

    n_vrf = checks["vrf_n"]
    vrf: dict[str, dict[str, float]] = {}
    for K in config["strikes"]:
        plain = cell[("plain", K, n_vrf)]
        work_plain = plain["sd_estimates"] ** 2 * plain["cost"]
        vrf[f"K={K:g}"] = {
            method: work_plain
            / (cell[(method, K, n_vrf)]["sd_estimates"] ** 2 * cell[(method, K, n_vrf)]["cost"])
            for method in config["methods"]
            if method != "plain"
        }

    low, high = checks["coverage_band"]
    slope_low, slope_high = checks["slope_band"]
    strikes = config["strikes"]
    itm, atm, otm = (f"K={K:g}" for K in strikes)
    deep_small_plain = cell[("plain", max(strikes), min(config["n_paths"]))]["coverage"]
    results = {
        "C1_coverage_in_band_for_n_ge_1000": all(
            low <= r["coverage"] <= high for r in rows if r["n_paths"] >= checks["coverage_min_n"]
        ),
        "C2_plain_deep_otm_n100_undercovers": deep_small_plain < low,
        "C3_no_bias_beyond_4_sigma": all(
            abs(r["bias"]) <= checks["bias_sigmas"] * r["bias_se"] for r in rows
        ),
        "C4_slopes_in_band": all(slope_low <= s["slope"] <= slope_high for s in scaling.values()),
        "C5_antithetic_vrf_above_1": all(v["antithetic"] > 1 for v in vrf.values()),
        "C5_cv_vrf_above_1_itm_atm": vrf[itm]["control_variate"] > 1
        and vrf[atm]["control_variate"] > 1,
        "C5_cv_vrf_largest_itm": vrf[itm]["control_variate"]
        > max(vrf[atm]["control_variate"], vrf[otm]["control_variate"]),
        "C5_cv_vrf_lower_deep_otm_than_atm": vrf[otm]["control_variate"]
        < vrf[atm]["control_variate"],
    }
    return {"scaling": scaling, "vrf_at_n": n_vrf, "vrf": vrf, "checks": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    env = environment(LIBRARIES)

    shape = (len(config["methods"]), len(config["strikes"]), len(config["n_paths"]))
    rows = [run_cell(config, i, j, k) for i, j, k in itertools.product(*map(range, shape))]
    analysis = analyse(config, rows)

    cells_csv = Path(config["output"]["cells_csv"])
    cells_csv.parent.mkdir(parents=True, exist_ok=True)
    with cells_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {"config": config, "environment": env, "cells": rows, **analysis}
    Path(config["output"]["summary_json"]).write_text(json.dumps(record, indent=2) + "\n")

    print(
        f"commit {env['commit'][:7]}{' (dirty)' if env['dirty'] else ''}; R = "
        f"{config['replications']}"
    )
    print(f"{'method':16s} {'K':>5s} {'n':>7s} {'coverage':>9s} {'bias/se':>8s} {'SE/sd':>6s}")
    for r in rows:
        print(
            f"{r['method']:16s} {r['K']:>5g} {r['n_paths']:>7d} {r['coverage']:>9.4f} "
            f"{r['bias'] / r['bias_se']:>+8.2f} {r['calibration_ratio']:>6.3f}"
        )
    for key, s in analysis["scaling"].items():
        print(f"slope {key:24s} {s['slope']:+.4f} +/- {s['slope_se']:.4f}")
    for key, v in analysis["vrf"].items():
        print(
            f"VRF at n={analysis['vrf_at_n']} {key}: "
            + ", ".join(f"{m} {x:.2f}" for m, x in v.items())
        )
    for name, passed in analysis["checks"].items():
        print(f"{name}: {'SUPPORTED' if passed else 'REFUTED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
