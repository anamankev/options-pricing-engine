"""QuantLib reconciliation of European BSM prices and Greeks (SPEC 8.1).

Pre-registered in docs/notes/phase_1_quantlib_reconciliation.md. Run from the repository root:

    python -m experiments.quantlib_reconciliation \
        --config experiments/configs/quantlib_reconciliation.toml

Writes one CSV row per case and a JSON summary that records the config, commit hash, library
versions and platform. Deterministic: no random numbers are used.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import subprocess
import tomllib
from collections.abc import Iterator
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from pricing import black_scholes as bs
from pricing.contracts import DAYS_PER_YEAR, OptionType

QUANTITIES = ("price", "delta", "gamma", "vega", "theta", "rho", "phi")
LIBRARIES = ("numpy", "scipy", "QuantLib", "mpmath")


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
            f"{name}_rel_err": relative_error(ours, ref),
        }
    return row


def summarize(rows: list[dict[str, Any]], rtol: float, small_d1: float) -> dict[str, Any]:
    """Per quantity: max, p99 and median relative error, exceedances, and the worst case."""
    summary: dict[str, Any] = {}
    for name in QUANTITIES:
        rel = np.array([row[f"{name}_rel_err"] for row in rows])
        worst = rows[int(np.argmax(rel))]
        over = [row for row in rows if row[f"{name}_rel_err"] > rtol]
        summary[name] = {
            "max_rel_err": _json_float(float(rel.max())),
            "p99_rel_err": _json_float(float(np.percentile(rel, 99))),
            "median_rel_err": _json_float(float(np.median(rel))),
            "max_abs_err": max(row[f"{name}_abs_err"] for row in rows),
            "count_over_rtol": len(over),
            "count_over_rtol_with_small_d1": sum(row["abs_d1"] < small_d1 for row in over),
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


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def environment() -> dict[str, Any]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libraries": {lib: version(lib) for lib in LIBRARIES},
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)

    rows = [compare_case(case) for case in iter_cases(config)]
    summary = summarize(rows, config["relative_tolerance"], config["small_d1"])

    cases_csv = Path(config["output"]["cases_csv"])
    cases_csv.parent.mkdir(parents=True, exist_ok=True)
    with cases_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {
        "config": config,
        "environment": environment(),
        "n_cases": len(rows),
        "summary": summary,
    }
    Path(config["output"]["summary_json"]).write_text(json.dumps(record, indent=2) + "\n")

    print(f"{len(rows)} cases; relative tolerance {config['relative_tolerance']:g}")
    print(f"{'quantity':8s} {'max rel':>10s} {'p99 rel':>10s} {'median':>10s} {'> rtol':>7s}")
    for name, s in summary.items():
        print(
            f"{name:8s} {s['max_rel_err']:>10.2e} {s['p99_rel_err']:>10.2e} "
            f"{s['median_rel_err']:>10.2e} {s['count_over_rtol']:>7d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
