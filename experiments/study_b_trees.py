"""Study B: convergence of binomial trees for European and American options (SPEC 11).

Pre-registered in docs/notes/study_b.md. Run from the repository root:

    python -m experiments.study_b_trees --config experiments/configs/study_b_trees.toml

It validates the American reference (QuantLib QdFp with a strict scheme), sweeps CRR, LR, BBS and
BBSR over SPEC 11's step counts for R1 (European) and the 18 Ju-Zhong cases (American), fits
convergence orders, times the trees on one representative case, evaluates the registered
hypotheses, and draws the SPEC 11 figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from experiments.provenance import environment
from pricing import binomial as bt
from pricing import black_scholes as bs
from pricing.contracts import ExerciseStyle, OptionType

LIBRARIES = ("numpy", "scipy", "matplotlib", "QuantLib")
METHODS = ("crr", "lr", "bbs", "bbsr")
LABEL = {"crr": "CRR", "lr": "Leisen-Reimer", "bbs": "BBS", "bbsr": "BBSR"}
TREE: dict[str, Callable[..., bt.TreeResult]] = {
    "crr": bt.crr,
    "lr": bt.leisen_reimer,
    "bbs": bt.bbs,
    "bbsr": bt.bbsr,
}

# Reference palette of the data-viz method (light surface): categorical slots 1 to 4 in fixed
# order for line charts (adjacent pairs validated); single-hue blue ramp for magnitude.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"crr": "#2a78d6", "lr": "#eb6834", "bbs": "#1baf7a", "bbsr": "#eda100"}
MARKER = {"crr": "o", "lr": "s", "bbs": "^", "bbsr": "D"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# --- Inputs ---------------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def step_counts(config: dict[str, Any]) -> dict[str, list[int]]:
    s = config["steps"]
    powers = [2**k for k in range(s["bbs_k"]["start"], s["bbs_k"]["stop"] + 1)]
    return {
        "crr": list(range(s["crr"]["start"], s["crr"]["stop"] + 1)),
        "lr": list(range(s["lr"]["start"], s["lr"]["stop"] + 1, 2)),
        "bbs": powers,
        "bbsr": powers,
    }


def american_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    keys = ("id", "type", "S", "K", "r", "q", "sigma", "days", "year_days", "published")
    cases = []
    for row in config["american"]["cases"]:
        case = dict(zip(keys, row, strict=True))
        case["T"] = case["days"] / case["year_days"]
        cases.append(case)
    return cases


def market(case: dict[str, Any]) -> dict[str, float]:
    return {k: float(case[k]) for k in ("S", "K", "T", "r", "q", "sigma")}


# --- Reference --------------------------------------------------------------------------------


def american_reference(case: dict[str, Any], scheme: tuple[int, int, float]) -> float:
    from pricing.oracles import quantlib_american  # needs the oracle extra; imported on use

    return quantlib_american(
        OptionType[case["type"]],
        S=case["S"],
        K=case["K"],
        r=case["r"],
        q=case["q"],
        sigma=case["sigma"],
        days=case["days"],
        year_days=case["year_days"],
        scheme=scheme,
    )


# --- Sweeps -----------------------------------------------------------------------------------


def sweep(
    job: tuple[str, str, dict[str, float], float, dict[str, list[int]], str],
) -> list[dict[str, Any]]:
    """All tree prices for one case: (case id, option type, market, reference, steps, exercise)."""
    case_id, type_name, mkt, reference, steps, exercise_name = job
    option_type, exercise = OptionType[type_name], ExerciseStyle(exercise_name)
    rows = []
    for method in METHODS:
        for n in steps[method]:
            value = TREE[method](option_type, **mkt, n_steps=n, exercise=exercise).value
            rows.append(
                {
                    "case": case_id,
                    "method": method,
                    "n": n,
                    "value": value,
                    "error": value - reference,
                }
            )
    return rows


def time_trees(case: dict[str, Any], config: dict[str, Any]) -> dict[str, dict[int, float]]:
    """Median runtime in ms per (method, N) for the representative American case."""
    timing = config["timing"]
    grids = {
        "crr": timing["crr_n"],
        "lr": timing["lr_n"],
        "bbs": step_counts(config)["bbs"],
        "bbsr": step_counts(config)["bbsr"],
    }
    option_type, mkt = OptionType[case["type"]], market(case)
    out: dict[str, dict[int, float]] = {}
    for method, ns in grids.items():
        out[method] = {}
        for n in ns:

            def run(method: str = method, n: int = n) -> None:
                TREE[method](option_type, **mkt, n_steps=n, exercise=ExerciseStyle.AMERICAN)

            run()  # warm-up
            samples = []
            for _ in range(timing["repeats"]):
                start = time.perf_counter_ns()
                run()
                samples.append(time.perf_counter_ns() - start)
            out[method][n] = float(np.median(samples)) / 1e6
    return out


# --- Analysis ---------------------------------------------------------------------------------


def fit_order(ns: list[int], errors: list[float], resolution: float) -> dict[str, Any]:
    """Least-squares slope of log10|e| on log10 N, with its standard error, excluding errors below
    the reference resolution (or exactly 0)."""
    pts = [(n, abs(e)) for n, e in zip(ns, errors, strict=True) if abs(e) >= resolution]
    excluded = len(ns) - len(pts)
    if len(pts) < 3:
        return {"slope": None, "slope_se": None, "n_points": len(pts), "n_excluded": excluded}
    x = np.log10([n for n, _ in pts])
    y = np.log10([e for _, e in pts])
    xc = x - x.mean()
    slope = float(xc @ (y - y.mean()) / (xc @ xc))
    resid = y - y.mean() - slope * xc
    se = math.sqrt(float(resid @ resid) / (len(pts) - 2) / float(xc @ xc))
    return {"slope": slope, "slope_se": se, "n_points": len(pts), "n_excluded": excluded}


def oscillation(ns: list[int], errors: list[float], lo: int, hi: int) -> float:
    """Median over N in [lo, hi] of |e_N - e_{N+1}| / max(|e_N|, |e_{N+1}|) (B1b)."""
    by_n = dict(zip(ns, errors, strict=True))
    ratios = [
        abs(by_n[n] - by_n[n + 1]) / max(abs(by_n[n]), abs(by_n[n + 1]))
        for n in range(lo, hi + 1)
        if n + 1 in by_n
    ]
    return float(np.median(ratios))


def richardson_share(
    ns: list[int], values: list[float], reference: float, lo: int, hi: int
) -> float:
    """Share of N in [lo, hi] where 2 V(2N) - V(N) is closer to the reference than V(2N) (B1c)."""
    by_n = dict(zip(ns, values, strict=True))
    better = [
        abs(bt.richardson(by_n[n], by_n[2 * n], order=1.0) - reference)
        < abs(by_n[2 * n] - reference)
        for n in range(lo, hi + 1)
    ]
    return sum(better) / len(better)


def envelope(
    ns: list[int], errors: list[float], start: int, block: int
) -> tuple[list[int], list[float]]:
    """Largest |e_N| in each block of `block` consecutive N from `start`, at the block's centre."""
    by_n = dict(zip(ns, errors, strict=True))
    centres, peaks = [], []
    n0 = start
    while n0 + block - 1 <= max(ns):
        peaks.append(max(abs(by_n[n]) for n in range(n0, n0 + block)))
        centres.append(n0 + block // 2)
        n0 += block
    return centres, peaks


def best_within_budget(
    times: dict[str, dict[int, float]], errors: dict[str, dict[int, float]], budget_ms: float
) -> dict[str, float | None]:
    """Each method's |error| at its largest timed N with median runtime within the budget."""
    out: dict[str, float | None] = {}
    for method, by_n in times.items():
        fitting = [n for n, ms in by_n.items() if ms <= budget_ms]
        out[method] = abs(errors[method][max(fitting)]) if fitting else None
    return out


def analyse(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    refs: dict[str, Any],
    times: dict[str, dict[int, float]],
) -> dict[str, Any]:
    fit, res = config["fit"], config["resolution"]
    min_n = {
        "crr": fit["crr_min_n"],
        "lr": fit["lr_min_n"],
        "bbs": fit["bbs_min_n"],
        "bbsr": fit["bbs_min_n"],
    }
    series: dict[tuple[str, str], tuple[list[int], list[float], list[float]]] = {}
    for r in rows:
        ns, errs, vals = series.setdefault((r["case"], r["method"]), ([], [], []))
        ns.append(r["n"])
        errs.append(r["error"])
        vals.append(r["value"])

    orders = []
    for (case, method), (ns, errs, _) in series.items():
        sel = [(n, e) for n, e in zip(ns, errs, strict=True) if n >= min_n[method]]
        variants = {"all": sel}
        if method == "crr":
            variants = {
                "odd": [p for p in sel if p[0] % 2],
                "even": [p for p in sel if not p[0] % 2],
            }
        for variant, pts in variants.items():
            f = fit_order([n for n, _ in pts], [e for _, e in pts], res)
            orders.append({"case": case, "method": method, "variant": variant, **f})
        if method == "crr":
            centres, peaks = envelope(ns, errs, fit["crr_min_n"], fit["envelope_block"])
            orders.append(
                {
                    "case": case,
                    "method": "crr",
                    "variant": "envelope",
                    **fit_order(centres, peaks, res),
                }
            )
    order = {(o["case"], o["method"], o["variant"]): o for o in orders}

    ns, errs, vals = series[("R1", "crr")]
    b1 = {
        "odd_slope": order[("R1", "crr", "odd")]["slope"],
        "even_slope": order[("R1", "crr", "even")]["slope"],
        "oscillation": oscillation(ns, errs, *fit["oscillation_window"]),
        "richardson_improves_share": richardson_share(
            ns, vals, refs["R1"], *fit["richardson_window"]
        ),
    }
    puts = [c["id"] for c in american_cases(config) if c["type"] == "PUT"]
    lr_put_slopes = [order[(c, "lr", "all")]["slope"] for c in puts]
    timing = config["timing"]
    sweep_err = {
        c: {m: dict(zip(series[(c, m)][0], series[(c, m)][1], strict=True)) for m in METHODS}
        for c in puts
    }
    b4_pairs = []
    for c in puts:
        for budget in timing["budgets_ms"]:
            best = best_within_budget(times, sweep_err[c], budget)
            winner = min(
                (m for m in METHODS if best[m] is not None), key=lambda m: best[m] or math.inf
            )
            b4_pairs.append(
                {
                    "case": c,
                    "budget_ms": budget,
                    "winner": winner,
                    **{f"{m}_error": best[m] for m in METHODS},
                }
            )
    b4_share = sum(p["winner"] == "bbsr" for p in b4_pairs) / len(b4_pairs)
    stats = {
        "B1": b1,
        "B2_lr_european_slope": order[("R1", "lr", "all")]["slope"],
        "B3_lr_put_slopes": dict(zip(puts, lr_put_slopes, strict=True)),
        "B3_median_slope": float(np.median([s for s in lr_put_slopes if s is not None])),
        "B4_bbsr_share": b4_share,
        "B4_winners": {m: sum(p["winner"] == m for p in b4_pairs) for m in METHODS},
    }
    verdicts = {
        "B1a": all(-1.2 <= b1[k] <= -0.8 for k in ("odd_slope", "even_slope")),
        "B1b": b1["oscillation"] >= 0.5,
        "B1c": b1["richardson_improves_share"] < 0.9,
        "B2": -2.2 <= stats["B2_lr_european_slope"] <= -1.8,
        "B3": -1.5 <= stats["B3_median_slope"] <= -0.5,
        "B4": b4_share >= timing["b4_min_share"],
    }
    return {"orders": orders, "stats": stats, "b4_pairs": b4_pairs, "verdicts": verdicts}


# --- Figures ----------------------------------------------------------------------------------


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _plot_case(ax: Any, case: str, series: dict[tuple[str, str], Any], resolution: float) -> None:
    for method in METHODS:
        ns, errs = series[(case, method)]
        pts = [(n, abs(e)) for n, e in zip(ns, errs, strict=True) if abs(e) > 0]
        if method == "crr":
            for parity, filled in ((1, True), (0, False)):
                sub = [(n, e) for n, e in pts if n % 2 == parity]
                ax.loglog(
                    *zip(*sub, strict=True),
                    linestyle="none",
                    marker="o",
                    markersize=1.6,
                    color=COLOR["crr"],
                    alpha=0.8 if filled else 0.35,
                )
        else:
            ax.loglog(
                *zip(*pts, strict=True),
                color=COLOR[method],
                linewidth=1.8,
                marker=MARKER[method],
                markersize=3.5,
            )
    ax.axhline(resolution, color=INK_2, linewidth=0.8, linestyle=":")
    _style(ax)


def plot_errors(
    config: dict[str, Any], rows: list[dict[str, Any]], path_eu: Path, path_am: Path
) -> None:
    series: dict[tuple[str, str], tuple[list[int], list[float]]] = {}
    for r in rows:
        ns, errs = series.setdefault((r["case"], r["method"]), ([], []))
        ns.append(r["n"])
        errs.append(r["error"])
    res = config["resolution"]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    _plot_case(ax, "R1", series, res)
    n_ref = np.array([10, 8192])
    for slope, label, y0 in ((-1, "slope -1", 0.4), (-2, "slope -2", 0.04)):
        ax.loglog(n_ref, y0 * (n_ref / 10.0) ** slope, color=INK_2, linewidth=0.8, linestyle="--")
        ax.annotate(
            label,
            (n_ref[-1], y0 * (n_ref[-1] / 10.0) ** slope),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
            va="center",
        )
    handles = [
        plt.Line2D(
            [],
            [],
            color=COLOR[m],
            marker=MARKER[m],
            linestyle="-" if m != "crr" else "none",
            label=LABEL[m] + (" (dots: odd N dark, even N light)" if m == "crr" else ""),
        )
        for m in METHODS
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    ax.set_xlabel("steps N", color=INK)
    ax.set_ylabel("|tree - BSM|", color=INK)
    ax.set_title(
        "European call (R1): tree error vs steps. Dotted: reference resolution 1e-8.",
        color=INK,
        fontsize=11,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path_eu, dpi=140, facecolor=SURFACE)
    plt.close(fig)

    puts = [c for c in american_cases(config) if c["type"] == "PUT"]
    fig, axes = plt.subplots(3, 5, figsize=(17, 10), facecolor=SURFACE, sharex=True)
    for ax, case in zip(axes.flat, puts, strict=False):
        _plot_case(ax, case["id"], series, res)
        ax.set_title(
            f"{case['id']}: S={case['S']:g} K={case['K']:g} sigma={case['sigma']:g} "
            f"T={case['T']:.3g}",
            color=INK,
            fontsize=9,
            loc="left",
        )
    for ax in list(axes.flat)[len(puts) :]:
        ax.axis("off")
    fig.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=10,
        labelcolor=INK,
        bbox_to_anchor=(0.98, 0.08),
    )
    fig.suptitle(
        "American puts (Ju-Zhong R4, R6): |tree - QdFp reference| vs steps N "
        "(log-log). Dotted: reference resolution 1e-8.",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path_am, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def plot_pareto(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    times: dict[str, dict[int, float]],
    path: Path,
) -> None:
    case = config["timing"]["case"]
    err = {(r["method"], r["n"]): abs(r["error"]) for r in rows if r["case"] == case}
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    _style(ax)
    for method in METHODS:
        pts = sorted((times[method][n], err[(method, n)]) for n in times[method])
        ax.loglog(
            *zip(*pts, strict=True),
            color=COLOR[method],
            marker=MARKER[method],
            linewidth=2,
            markersize=5,
            label=LABEL[method],
        )
    for budget in config["timing"]["budgets_ms"]:
        ax.axvline(budget, color=INK_2, linewidth=0.8, linestyle=":")
        ax.annotate(
            f"{budget:g} ms",
            (budget, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(3, 0),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=INK_2,
        )
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    ax.set_xlabel("median runtime per price (ms, pure Python)", color=INK)
    ax.set_ylabel("|tree - QdFp reference|", color=INK)
    ax.set_title(
        f"Error vs runtime, American put {case} (one point per timed N)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def crr_surface(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log10 |CRR - BSM| for the European R1 call over strike and N (reporting only)."""
    s = config["surface"]
    strikes = np.linspace(s["K"]["start"], s["K"]["stop"], s["K"]["count"])
    ns = np.arange(s["n"]["start"], s["n"]["stop"] + 1)
    mkt = dict(config["european"])
    grid = np.empty((len(ns), len(strikes)))
    for j, K in enumerate(strikes):
        ref = float(bs.price(OptionType.CALL, **{**mkt, "K": float(K)}))
        for i, n in enumerate(ns):
            v = bt.crr(OptionType.CALL, **{**mkt, "K": float(K)}, n_steps=int(n)).value
            grid[i, j] = math.log10(max(abs(v - ref), 1e-16))
    return strikes, ns, grid


def plot_surface(
    strikes: np.ndarray, ns: np.ndarray, grid: np.ndarray, path_3d: Path, path_2d: Path
) -> None:
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    X, Y = np.meshgrid(strikes, np.log10(ns))
    fig = plt.figure(figsize=(10, 6.5), facecolor=SURFACE)
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, grid, cmap=cmap, linewidth=0, antialiased=True, rstride=2, cstride=1)
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("strike K", color=INK)
    ax.set_ylabel("log10 N", color=INK)
    ax.set_zlabel("log10 |CRR - BSM|", color=INK)
    ax.view_init(elev=32, azim=-120)
    ax.set_title("CRR European call error over strike and steps (R1 market; S = 42)", color=INK)
    fig.savefig(path_3d, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    mesh = ax.pcolormesh(strikes, ns, grid, cmap=cmap, shading="nearest")
    ax.set_yscale("log")
    ax.set_xlabel("strike K", color=INK)
    ax.set_ylabel("steps N", color=INK)
    ax.set_title(
        "Same data as the 3D view: log10 |CRR - BSM| (darker = larger error)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    bar = fig.colorbar(mesh, ax=ax)
    bar.set_label("log10 |error| (exact zero shown at -16)", color=INK)
    fig.tight_layout()
    fig.savefig(path_2d, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# --- Main -------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    env = environment(LIBRARIES)
    steps = step_counts(config)
    scheme = tuple(config["reference_scheme"])
    check = tuple(config["check_scheme"])

    cases = american_cases(config)
    refs: dict[str, Any] = {"R1": float(bs.price(OptionType.CALL, **config["european"]))}
    gates = []
    for c in cases:
        refs[c["id"]] = american_reference(c, scheme)
        gates.append(
            {
                "case": c["id"],
                "reference": refs[c["id"]],
                "check_difference": abs(refs[c["id"]] - american_reference(c, check)),
                "published": c["published"],
                "published_difference": abs(refs[c["id"]] - c["published"]),
            }
        )
    gate1 = all(g["check_difference"] <= config["gate1_tolerance"] for g in gates)
    gate2 = all(g["published_difference"] <= config["gate2_tolerance"] for g in gates)
    if not (gate1 and gate2):
        raise SystemExit(
            f"reference gates failed (gate 1 {gate1}, gate 2 {gate2}); use the "
            "SPEC 8.1 fallback as registered"
        )

    jobs = [
        (
            "R1",
            "CALL",
            {k: float(v) for k, v in config["european"].items()},
            refs["R1"],
            steps,
            "european",
        )
    ]
    jobs += [(c["id"], c["type"], market(c), refs[c["id"]], steps, "american") for c in cases]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        rows = [r for part in pool.map(sweep, jobs) for r in part]
    rep = next(c for c in cases if c["id"] == config["timing"]["case"])
    times = time_trees(rep, config)  # alone, after the parallel sweeps
    analysis = analyse(config, rows, refs, times)

    largest = {m: max(steps[m]) for m in METHODS}
    juzhong = [
        g
        | {
            f"{m}_N{largest[m]}": next(
                r["value"]
                for r in rows
                if r["case"] == g["case"] and r["method"] == m and r["n"] == largest[m]
            )
            for m in METHODS
        }
        for g in gates
    ]

    out = config["output"]
    figures = Path(out["figure_dir"])
    Path(out["errors_csv"]).parent.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name, data in ((out["errors_csv"], rows), (out["orders_csv"], analysis["orders"])):
        with Path(name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    record = {
        "config": config,
        "environment": env,
        "gates": {"gate1": gate1, "gate2": gate2},
        "juzhong": juzhong,
        "timing_ms": {m: {str(n): t for n, t in v.items()} for m, v in times.items()},
        "stats": analysis["stats"],
        "b4_pairs": analysis["b4_pairs"],
        "verdicts": analysis["verdicts"],
    }
    Path(out["summary_json"]).write_text(json.dumps(record, indent=2) + "\n")
    plot_errors(
        config, rows, figures / "study_b_european.png", figures / "study_b_american_puts.png"
    )
    plot_pareto(config, rows, times, figures / "study_b_pareto.png")
    plot_surface(
        *crr_surface(config),
        figures / "study_b_crr_surface_3d.png",
        figures / "study_b_crr_surface_heatmap.png",
    )

    print(f"commit {env['commit'][:7]}{' (dirty)' if env['dirty'] else ''}; gates {gate1}, {gate2}")
    print(json.dumps(analysis["stats"], indent=1))
    for name, passed in analysis["verdicts"].items():
        print(f"{name}: {'SUPPORTED' if passed else 'REFUTED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
