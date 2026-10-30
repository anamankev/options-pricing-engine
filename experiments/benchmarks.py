"""Performance benchmarks (SPEC 12).

Pre-registered in docs/notes/benchmarks.md. Run from the repository root:

    python -m experiments.benchmarks --config experiments/configs/benchmarks.toml

Measures vectorized BSM latency and throughput at batch sizes 1, 1e3 and 1e6, tree runtime against
step count, and Monte Carlo cost per path. Every measurement has a warm-up and repeated timed calls
with the garbage collector paused, and reports the median and interquartile range, never the
fastest run (SPEC 12). The output records the hardware, OS, library versions, BLAS and thread
settings, and commit. One machine's numbers are not evidence of production performance (SPEC 15).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt
import numpy as np

from experiments.provenance import environment
from pricing import binomial as bt
from pricing import black_scholes as bs
from pricing.contracts import ExerciseStyle, OptionType
from pricing.monte_carlo import MCMethod, monte_carlo

LIBRARIES = ("numpy", "scipy", "matplotlib")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
TREES: dict[str, Callable[..., bt.TreeResult]] = {
    "crr": bt.crr,
    "lr": bt.leisen_reimer,
    "bbs": bt.bbs,
    "bbsr": bt.bbsr,
}

# Reference palette of the data-viz method (light surface), categorical slots in fixed order.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def measure(fn: Callable[[], Any], repeats: int, warmup: int) -> dict[str, float]:
    """Median and interquartile range of `repeats` timed calls, in seconds, with the garbage
    collector paused during timing and restored to its previous state afterwards."""
    for _ in range(warmup):
        fn()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            fn()
            samples.append(time.perf_counter_ns() - start)
    finally:
        if was_enabled:
            gc.enable()
    seconds = np.array(samples, dtype=float) / 1e9
    return {
        "median_s": float(np.median(seconds)),
        "p25_s": float(np.percentile(seconds, 25)),
        "p75_s": float(np.percentile(seconds, 75)),
        "repeats": repeats,
    }


def loglog_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of log(y) on log(x)."""
    x, y = np.log(np.asarray(xs, dtype=float)), np.log(np.asarray(ys, dtype=float))
    xc = x - x.mean()
    return float(xc @ (y - y.mean()) / (xc @ xc))


def cpu_model() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def hardware_and_threads() -> dict[str, Any]:
    """SPEC 12: hardware, OS, BLAS and thread settings (library versions and commit come from
    experiments.provenance)."""
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    return {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "os": platform.platform(),
        "blas": {"name": blas.get("name"), "version": blas.get("version")},
        "thread_variables": {v: os.environ.get(v, "unset") for v in THREAD_VARIABLES},
    }


def bench_bsm(config: dict[str, Any]) -> list[dict[str, Any]]:
    b, warmup = config["bsm"], config["warmup"]
    rows = []
    for name in b["functions"]:
        fn = getattr(bs, name)
        for size, repeats in zip(b["batches"], b["repeats"], strict=True):
            if size == 1:
                S: Any = 100.0  # scalar call: the float path of SPEC 2.3
            else:
                S = np.linspace(*b["S_range"], size)
            m = measure(lambda fn=fn, S=S: fn(OptionType.CALL, S=S, **b["market"]), repeats, warmup)
            rows.append(
                {
                    "section": "bsm",
                    "method": name,
                    "exercise": "",
                    "size": size,
                    **m,
                    "per_unit_s": m["median_s"] / size,
                    "per_second": size / m["median_s"],
                }
            )
    return rows


def bench_trees(config: dict[str, Any]) -> list[dict[str, Any]]:
    t, warmup = config["trees"], config["warmup"]
    option_type = OptionType[t["option_type"]]
    rows = []
    for exercise in (ExerciseStyle.EUROPEAN, ExerciseStyle.AMERICAN):
        for method, tree in TREES.items():
            for n in t["n"]:
                steps = n + 1 if method == "lr" else n  # LR needs odd N
                repeats = t["repeats_large"] if n >= t["large_from"] else t["repeats_small"]
                m = measure(
                    lambda tree=tree, steps=steps, exercise=exercise: tree(
                        option_type, **t["market"], n_steps=steps, exercise=exercise
                    ),
                    repeats,
                    warmup,
                )
                rows.append(
                    {
                        "section": "trees",
                        "method": method,
                        "exercise": exercise.value,
                        "size": steps,
                        **m,
                        "per_unit_s": m["median_s"],
                        "per_second": 1.0 / m["median_s"],
                    }
                )
    return rows


def bench_mc(config: dict[str, Any]) -> list[dict[str, Any]]:
    mc, warmup = config["monte_carlo"], config["warmup"]
    rows = []
    for method in MCMethod:
        for n, repeats in zip(mc["n"], mc["repeats"], strict=True):
            pilot = n // mc["pilot_divisor"] if method is MCMethod.CONTROL_VARIATE else None

            def run(method: MCMethod = method, n: int = n, pilot: int | None = pilot) -> Any:
                return monte_carlo(
                    OptionType.CALL,
                    **mc["market"],
                    n_paths=n,
                    seed=mc["seed"],
                    method=method,
                    pilot_paths=pilot,
                )

            evals = run().n_payoff_evals
            m = measure(run, repeats, warmup)
            rows.append(
                {
                    "section": "monte_carlo",
                    "method": method.value,
                    "exercise": "",
                    "size": n,
                    **m,
                    "per_unit_s": m["median_s"] / n,
                    "per_second": n / m["median_s"],
                    "per_payoff_eval_s": m["median_s"] / evals,
                }
            )
    return rows


def analyse(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = config["checks"]
    fit_min = config["trees"]["fit_min_n"]
    slopes = {}
    for exercise in ("european", "american"):
        for method in TREES:
            pts = [
                (r["size"], r["median_s"])
                for r in rows
                if r["section"] == "trees"
                and r["method"] == method
                and r["exercise"] == exercise
                and r["size"] >= fit_min
            ]
            slopes[f"{method}/{exercise}"] = loglog_slope([p[0] for p in pts], [p[1] for p in pts])
    bsm = {(r["method"], r["size"]): r["per_unit_s"] for r in rows if r["section"] == "bsm"}
    sizes = config["bsm"]["batches"]
    p2_ratios = {f: bsm[(f, sizes[0])] / bsm[(f, sizes[-1])] for f in config["bsm"]["functions"]}
    mc = [r for r in rows if r["section"] == "monte_carlo"]
    p3_ratios = {}
    for method in MCMethod:
        per_path = [
            r["per_unit_s"]
            for r in mc
            if r["method"] == method.value and r["size"] >= config["monte_carlo"]["p3_from_n"]
        ]
        p3_ratios[method.value] = max(per_path) / min(per_path)
    largest = max(config["monte_carlo"]["n"])
    per_path_largest = {r["method"]: r["per_unit_s"] for r in mc if r["size"] == largest}
    lo, hi = checks["p1_band"]
    verdicts = {
        "P1": all(lo <= s <= hi for s in slopes.values()),
        "P2": all(ratio >= checks["p2_min_ratio"] for ratio in p2_ratios.values()),
        "P3": all(ratio < checks["p3_max_ratio"] for ratio in p3_ratios.values()),
        "P4": per_path_largest["antithetic"] <= per_path_largest["plain"],
    }
    return {
        "tree_slopes": slopes,
        "bsm_per_option_ratio": p2_ratios,
        "mc_per_path_ratio": p3_ratios,
        "mc_per_path_at_largest_n": per_path_largest,
        "verdicts": verdicts,
    }


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _series(
    ax: Any, rows: list[dict[str, Any]], key: str, labels: dict[str, str], value: str
) -> None:
    for (method, label), color, marker in zip(labels.items(), SLOTS, MARKERS, strict=False):
        pts = sorted(
            (
                r["size"],
                r[value],
                r["p25_s"] / r["median_s"] * r[value],
                r["p75_s"] / r["median_s"] * r[value],
            )
            for r in rows
            if r[key] == method
        )
        xs, ys, lo, hi = (np.array(v) for v in zip(*pts, strict=True))
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.loglog(xs, ys, color=color, marker=marker, linewidth=2, markersize=5, label=label)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)


def plot(rows: list[dict[str, Any]], env: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
    bsm = [r for r in rows if r["section"] == "bsm"]
    _series(axes[0], bsm, "method", {"price": "price", "delta": "delta"}, "per_unit_s")
    axes[0].set_xlabel("batch size (options per call)", color=INK)
    axes[0].set_ylabel("seconds per option (median; band: IQR)", color=INK)
    axes[0].set_title("Vectorized BSM: time per option", color=INK, fontsize=10, loc="left")
    trees = [r for r in rows if r["section"] == "trees" and r["exercise"] == "american"]
    _series(
        axes[1],
        trees,
        "method",
        {"crr": "CRR", "lr": "Leisen-Reimer", "bbs": "BBS", "bbsr": "BBSR"},
        "per_unit_s",
    )
    axes[1].set_xlabel("steps N", color=INK)
    axes[1].set_ylabel("seconds per American price", color=INK)
    axes[1].set_title("Trees: runtime vs N (American put)", color=INK, fontsize=10, loc="left")
    # Reference slopes anchored at CRR's smallest N: runtime ~ N (overhead) and ~ N^2 (work).
    crr = sorted((r["size"], r["median_s"]) for r in trees if r["method"] == "crr")
    n0, t0 = crr[0]
    ns = np.array([n0, crr[-1][0]], dtype=float)
    for power, style in ((1, ":"), (2, "--")):
        axes[1].loglog(ns, t0 * (ns / n0) ** power, color=INK_2, linewidth=1, linestyle=style)
        axes[1].annotate(
            f"slope {power}",
            (ns[-1], t0 * (ns[-1] / n0) ** power),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
            va="center",
        )
    mc = [r for r in rows if r["section"] == "monte_carlo"]
    _series(
        axes[2],
        mc,
        "method",
        {"plain": "plain", "antithetic": "antithetic", "control_variate": "control variate"},
        "per_unit_s",
    )
    axes[2].set_xlabel("paths n", color=INK)
    axes[2].set_ylabel("seconds per path", color=INK)
    axes[2].set_title("Monte Carlo: cost per path", color=INK, fontsize=10, loc="left")
    fig.suptitle(
        f"Benchmarks on {env['cpu']}, Python {platform.python_version()}, pure Python "
        "and NumPy (one machine; not evidence of production performance)",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    env = environment(LIBRARIES) | hardware_and_threads()

    rows = bench_bsm(config) + bench_trees(config) + bench_mc(config)
    analysis = analyse(config, rows)

    out = config["output"]
    for name in (out["table_csv"], out["figure"]):
        Path(name).parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))  # every field, first-seen order
    with Path(out["table_csv"]).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    Path(out["summary_json"]).write_text(
        json.dumps({"config": config, "environment": env, **analysis, "rows": rows}, indent=2)
        + "\n"
    )
    plot(rows, env, Path(out["figure"]))

    print(
        f"commit {env['commit'][:7]}{' (dirty)' if env['dirty'] else ''}; {env['cpu']}; "
        f"BLAS {env['blas']['name']}"
    )
    for r in rows:
        print(
            f"{r['section']:12s} {r['method']:16s} {r['exercise']:9s} {r['size']:>8d}  "
            f"median {r['median_s'] * 1e3:10.4f} ms  IQR {r['p25_s'] * 1e3:.4f}-"
            f"{r['p75_s'] * 1e3:.4f}  per unit {r['per_unit_s']:.3e} s"
        )
    print(json.dumps({k: v for k, v in analysis.items() if k != "verdicts"}, indent=1))
    for name, passed in analysis["verdicts"].items():
        print(f"{name}: {'SUPPORTED' if passed else 'REFUTED'}")
    if not math.isfinite(sum(r["median_s"] for r in rows)):
        raise SystemExit("non-finite timing recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
