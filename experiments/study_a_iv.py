"""Study A: implied-volatility solver accuracy, robustness and cost (SPEC 10).

Pre-registered in docs/notes/study_a.md. Run from the repository root:

    python -m experiments.study_a_iv --config experiments/configs/study_a_iv.toml

For every grid problem (log-moneyness x, total volatility s, call or put) it computes the true
undiscounted Black price with mpmath at 50 digits, rounds it once to float64, and asks each solver
to recover s. It records the outcome, the error in ulps, relative and normalized by the condition
number kappa = |c / (s dc/ds)|, iterations, and the median of 50 timed solves. It then evaluates
the registered hypotheses and draws the SPEC 10 figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt
import mpmath
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

from experiments.provenance import environment
from pricing.contracts import OptionType
from pricing.errors import InfeasiblePriceError
from pricing.implied_vol import IVMethod, solve_black
from pricing.oracles import lbr_implied_volatility

EPS = sys.float_info.epsilon
LIBRARIES = ("numpy", "scipy", "mpmath", "matplotlib", "lets_be_rational")
FAILURES = ("MAX_ITERATIONS", "BRACKET_FAILURE", "DIVERGED", "ZERO_DERIVATIVE")
LABEL = {
    "brent": "Brent",
    "naive_newton": "naive Newton",
    "safeguarded_newton": "safeguarded Newton",
    "lets_be_rational": "Let's Be Rational",
}

# Reference palette of the data-viz method (light surface). Categorical slots 1 to 3 are validated
# all-pairs for colour-vision deficiency; the sequential ramp is single-hue blue.
SURFACE, INK, INK_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
UNINFORMATIVE, FAILED = "#f0efec", "#8a8984"  # pale: no information; mid grey: no value
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# --- Problems and ground truth -----------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def grid_axes(config: dict[str, Any]) -> tuple[list[float], list[float]]:
    gx, gs = config["x"], config["s"]
    n_x = round((gx["stop"] - gx["start"]) / gx["step"]) + 1
    xs = [gx["start"] + i * gx["step"] for i in range(n_x)]
    ss = [float(v) for v in np.geomspace(gs["low"], gs["high"], gs["count"])]
    return xs, ss


def truth(theta: int, F: float, K: float, s: float, digits: int) -> dict[str, float]:
    """Exact-input undiscounted price, its float64 rounding, dc/ds and kappa (SPEC 8.3, 10)."""
    with mpmath.workdps(digits):
        Fm, Km, sm = mpmath.mpf(F), mpmath.mpf(K), mpmath.mpf(s)
        d1 = mpmath.log(Fm / Km) / sm + sm / 2
        c = theta * (Fm * mpmath.ncdf(theta * d1) - Km * mpmath.ncdf(theta * (d1 - sm)))
        vega = Fm * mpmath.npdf(d1)
        kappa = abs(c / (sm * vega)) if vega > 0 else mpmath.inf
        return {
            "c_target": float(c),
            "kappa": float(kappa) if kappa < 1e300 else math.inf,
            "log10_c_true": float(mpmath.log10(c)) if c > 0 else -math.inf,
        }


def well_posed(c_target: float, F: float, K: float) -> bool:
    """SPEC 10: ill posed where c underflows or c < eps max(F, K)."""
    return c_target >= EPS * max(F, K)


SMALLEST_NORMAL = sys.float_info.min


def informative(c_target: float, kappa: float) -> bool:
    """Post-hoc region (not pre-registered; docs/notes/study_a.md, section 4): the float64 target
    carries information about s, i.e. it is a normal float and the attainable relative error
    kappa eps is below 1. SPEC 10's rule labels in-the-money prices whose time value is lost in
    the rounding of the intrinsic value as well posed, and solvable deep out-of-the-money prices
    as ill posed."""
    return c_target >= SMALLEST_NORMAL and kappa * EPS < 1.0


# --- Solving ------------------------------------------------------------------------------


def make_solver(
    method: str, theta: int, c: float, F: float, K: float, T: float
) -> Callable[[], Any]:
    option_type = OptionType.CALL if theta == 1 else OptionType.PUT
    if method == "lets_be_rational":
        return lambda: lbr_implied_volatility(option_type, c, F=F, K=K, T=T)
    iv_method = IVMethod(method)
    return lambda: solve_black(option_type, c, F=F, K=K, T=T, method=iv_method)


def outcome(method: str, solve: Callable[[], Any], T: float) -> dict[str, Any]:
    """Outcome class, returned total volatility (or None), iterations and evaluations."""
    try:
        result = solve()
    except InfeasiblePriceError:
        return {"outcome": "INFEASIBLE", "s_hat": None, "iterations": None, "evaluations": None}
    if method == "lets_be_rational":
        s_hat = float(result) * math.sqrt(T)
        cls = "AT_LOWER_BOUND" if s_hat == 0.0 else "CONVERGED"
        return {"outcome": cls, "s_hat": s_hat, "iterations": 2, "evaluations": None}
    s_hat = result.total_vol
    return {
        "outcome": result.reason.name,
        "s_hat": s_hat,
        "iterations": result.iterations,
        "evaluations": result.function_evaluations,
    }


def errors(s_hat: float | None, s: float, kappa: float) -> dict[str, float | None]:
    """e_rel, e_ulp, e_norm (floored at one ulp of attainable error, registered) and SPEC's raw
    e_norm_spec = e_rel / (kappa eps)."""
    if s_hat is None:
        return {"e_rel": None, "e_ulp": None, "e_norm": None, "e_norm_spec": None}
    diff = abs(s_hat - s)
    e_rel = diff / s
    return {
        "e_rel": e_rel,
        "e_ulp": diff / float(np.spacing(s)),
        "e_norm": e_rel / (max(kappa, 1.0) * EPS),
        "e_norm_spec": e_rel / (kappa * EPS) if kappa > 0 else math.inf,
    }


def median_time_ns(solve: Callable[[], Any], repeats: int) -> float:
    solve()  # warm-up
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        solve()
        samples.append(time.perf_counter_ns() - start)
    return float(np.median(samples))


def run(config: dict[str, Any]) -> list[dict[str, Any]]:
    F, T, digits = config["F"], config["T"], config["mp_digits"]
    xs, ss = grid_axes(config)
    rows = []
    for name in config["option_types"]:
        theta = 1 if name == "CALL" else -1
        for x in xs:
            K = F * math.exp(-x)
            for s in ss:
                t = truth(theta, F, K, s, digits)
                row: dict[str, Any] = {
                    "option_type": name,
                    "x": x,
                    "s": s,
                    "K": K,
                    **t,
                    "well_posed": well_posed(t["c_target"], F, K),
                    "informative": informative(t["c_target"], t["kappa"]),
                }
                for method in config["methods"]:
                    solve = make_solver(method, theta, t["c_target"], F, K, T)
                    o = outcome(method, solve, T)
                    e = errors(o["s_hat"], s, t["kappa"])
                    timed = (
                        median_time_ns(solve, config["timing"]["repeats"])
                        if o["outcome"] != "INFEASIBLE"
                        else None
                    )
                    row |= {f"{method}_{k}": v for k, v in {**o, **e, "time_ns": timed}.items()}
                rows.append(row)
    return rows


# --- Analysis -----------------------------------------------------------------------------


def _values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows if r[key] is not None], dtype=float)


def _median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def analyse(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Registered analysis on SPEC 10's well-posed region, and the same checks on the post-hoc
    informative region."""
    return {
        "registered": _evaluate(config, rows, "well_posed"),
        "post_hoc_informative": _evaluate(config, rows, "informative"),
    }


def _at_least(value: float | None, threshold: float) -> bool:
    """A statistic over an empty region (None) supports nothing. Explicit None checks: a value of
    exactly 0.0 is falsy, and an earlier `value or default` form misjudged it."""
    return value is not None and value >= threshold


def _at_most(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def _evaluate(config: dict[str, Any], rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    checks, methods = config["checks"], config["methods"]
    wp = [r for r in rows if r[key]]
    ill = [r for r in rows if not r[key]]

    per_method: dict[str, Any] = {}
    for m in methods:
        times = _values(wp, f"{m}_time_ns")
        enorm = _values(wp, f"{m}_e_norm")
        per_method[m] = {
            "outcomes_well_posed": dict(Counter(r[f"{m}_outcome"] for r in wp)),
            "outcomes_ill_posed": dict(Counter(r[f"{m}_outcome"] for r in ill)),
            "e_norm_median": _median(enorm),
            "e_norm_p99": float(np.percentile(enorm, 99)) if enorm.size else None,
            "e_ulp_median": _median(_values(wp, f"{m}_e_ulp")),
            "iterations_median": _median(_values(wp, f"{m}_iterations")),
            "iterations_max": float(_values(wp, f"{m}_iterations").max())
            if m != "lets_be_rational"
            else 2.0,
            "time_median_us": float(np.median(times)) / 1e3,
            "time_iqr_us": [
                float(np.percentile(times, 25)) / 1e3,
                float(np.percentile(times, 75)) / 1e3,
            ],
        }

    def region(rs: list[dict[str, Any]], **bounds: float) -> list[dict[str, Any]]:
        def inside(r: dict[str, Any]) -> bool:
            return (
                r["s"] <= bounds.get("s_max", math.inf)
                and r["s"] >= bounds.get("s_min", 0)
                and abs(r["x"]) >= bounds.get("abs_x_min", 0)
                and abs(r["x"]) <= bounds.get("abs_x_max", math.inf)
            )

        return [r for r in rs if inside(r)]

    def fail_fraction(rs: list[dict[str, Any]], m: str) -> float | None:
        """Share of a region where the method failed; None for an empty region."""
        return sum(r[f"{m}_outcome"] in FAILURES for r in rs) / len(rs) if rs else None

    a1_fail = region(
        wp, s_max=checks["a1_fail_region"]["s_max"], abs_x_min=checks["a1_fail_region"]["abs_x_min"]
    )
    a1_calm = region(
        wp, s_min=checks["a1_calm_region"]["s_min"], abs_x_max=checks["a1_calm_region"]["abs_x_max"]
    )
    small_s = region(wp, s_max=checks["a2_small_s"])
    a5 = region(wp, s_max=checks["a5_region"]["s_max"], abs_x_max=checks["a5_region"]["abs_x_max"])

    zero = [r for r in rows if r["c_target"] == 0.0]
    positive_ill = [r for r in ill if r["c_target"] > 0.0]
    robust = ("brent", "safeguarded_newton", "lets_be_rational")
    comparable = [r for r in positive_ill if all(r[f"{m}_s_hat"] is not None for m in robust)]

    def agree(r: dict[str, Any]) -> bool:
        # Errors of exactly 0 cannot be compared by ratio, so each is floored at eps (one ulp).
        errs = [max(r[f"{m}_e_rel"], EPS) for m in robust]
        return max(errs) / min(errs) <= checks["a4_agreement_factor"]

    stats = {
        "n_points": len(rows),
        "n_well_posed": len(wp),
        "n_ill_posed": len(ill),
        "n_zero_target": len(zero),
        "n_positive_ill_posed": len(positive_ill),
        "a1_fail_fraction": fail_fraction(a1_fail, "naive_newton"),
        "a1_fail_n": len(a1_fail),
        "a1_calm_fraction": fail_fraction(a1_calm, "naive_newton"),
        "a1_calm_n": len(a1_calm),
        "a2_bad_outcomes": sum(
            r["brent_outcome"] in ("MAX_ITERATIONS", "BRACKET_FAILURE") for r in wp
        ),
        "a2_small_s_brent_median_enorm": _median(_values(small_s, "brent_e_norm")),
        "a4_zero_recovered": sum(
            any((r[f"{m}_s_hat"] or 0.0) > 0.0 for m in methods) for r in zero
        ),
        "a4_comparable_n": len(comparable),
        "a4_agreeing_fraction": (
            sum(agree(r) for r in comparable) / len(comparable) if comparable else None
        ),
        "a5_n": len(a5),
        "a5_newton_median_enorm": _median(_values(a5, "safeguarded_newton_e_norm")),
        "a5_lbr_median_enorm": _median(_values(a5, "lets_be_rational_e_norm")),
    }
    lbr = per_method["lets_be_rational"]
    verdicts = {
        "A1": _at_least(stats["a1_fail_fraction"], checks["a1_fail_region"]["min_fraction"])
        and _at_most(stats["a1_calm_fraction"], checks["a1_calm_region"]["max_fraction"]),
        "A2a": stats["a2_bad_outcomes"] == 0,
        "A2b": per_method["brent"]["iterations_median"] >= checks["a2_min_median_iterations"],
        "A2c": _at_least(stats["a2_small_s_brent_median_enorm"], checks["a2_min_median_enorm"]),
        "A3": lbr["e_norm_median"] <= checks["a3_max_median_enorm"]
        and lbr["e_norm_p99"] <= checks["a3_max_p99_enorm"],
        "A4i": stats["a4_zero_recovered"] == 0,
        "A4ii": stats["a4_agreeing_fraction"] is not None
        and stats["a4_agreeing_fraction"] >= checks["a4_min_agreeing_fraction"],
        "A5": _at_least(stats["a5_newton_median_enorm"], checks["a5_min_median_newton"])
        and _at_most(stats["a5_lbr_median_enorm"], checks["a5_max_median_lbr"]),
    }
    return {"per_method": per_method, "stats": stats, "verdicts": verdicts}


# --- Figures ------------------------------------------------------------------------------


def _matrix(
    rows: list[dict[str, Any]], xs: list[float], ss: list[float], option: str, key: str
) -> np.ndarray:
    """Grid of values (rows: s, columns: x); NaN marks 'no value' for masking in plots only."""
    index = {(r["x"], r["s"]): r for r in rows if r["option_type"] == option}
    out = np.full((len(ss), len(xs)), np.nan)
    for i, s in enumerate(ss):
        for j, x in enumerate(xs):
            v = index[(x, s)][key]
            if v is not None and not isinstance(v, str):
                out[i, j] = float(v)
    return out


def _region_mask(
    rows: list[dict[str, Any]], xs: list[float], ss: list[float], option: str, key: str
) -> np.ndarray:
    index = {(r["x"], r["s"]): r for r in rows if r["option_type"] == option}
    return np.array([[1.0 if index[(x, s)][key] else 0.0 for x in xs] for s in ss])


def _boundaries(
    ax: Any, rows: list[dict[str, Any]], xs: list[float], ss: list[float], option: str
) -> None:
    """Solid: SPEC 10 well-posed boundary. Dashed: post-hoc informative boundary."""
    log_s = np.log10(ss)
    for key, style in (("well_posed", "solid"), ("informative", "dashed")):
        mask = _region_mask(rows, xs, ss, option, key)
        ax.contour(xs, log_s, mask, levels=[0.5], colors=INK, linewidths=1.0, linestyles=style)


def _log_with_floor(values: np.ndarray, floor: float) -> np.ma.MaskedArray:
    """log10 for display; values below `floor` (including exact 0) are shown at the floor."""
    shown = np.ma.masked_invalid(values)
    return np.ma.log10(np.ma.maximum(shown, floor))


def _informative_only(
    rows: list[dict[str, Any]],
    xs: list[float],
    ss: list[float],
    option: str,
    key: str,
    floor: float,
) -> np.ma.MaskedArray:
    """log10 of a metric for display, masked where no value was returned or the point is not
    informative; values below `floor` (including exact 0) are shown at the floor."""
    shown = _log_with_floor(_matrix(rows, xs, ss, option, key), floor)
    uninformative = _region_mask(rows, xs, ss, option, "informative") == 0.0
    return np.ma.masked_where(uninformative, shown)


def plot_heatmaps(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    key: str,
    floor: float,
    title: str,
    path: Path,
) -> None:
    xs, ss = grid_axes(config)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP).with_extremes(bad=FAILED)
    pale = ListedColormap([UNINFORMATIVE])
    log_s = np.log10(ss)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), facecolor=SURFACE, sharex=True, sharey=True)
    data = {
        (o, m): _informative_only(rows, xs, ss, o, f"{m}_{key}", floor)
        for o in config["option_types"]
        for m in config["methods"]
    }
    high = max(float(v.max()) for v in data.values() if v.count())
    for i, option in enumerate(config["option_types"]):
        uninformative = _region_mask(rows, xs, ss, option, "informative") == 0.0
        for j, m in enumerate(config["methods"]):
            ax = axes[i, j]
            values = data[(option, m)]
            # Failed points inside the informative region show the colormap's "bad" colour.
            filled = np.ma.masked_where(uninformative, values.filled(np.nan))
            mesh = ax.pcolormesh(
                xs, log_s, filled, cmap=cmap, shading="nearest", vmin=math.log10(floor), vmax=high
            )
            ax.pcolormesh(
                xs,
                log_s,
                np.ma.masked_where(~uninformative, np.ones_like(values)),
                cmap=pale,
                shading="nearest",
            )
            _boundaries(ax, rows, xs, ss, option)
            ax.set_title(f"{LABEL[m]}, {option.lower()}s", color=INK, fontsize=10, loc="left")
            ax.tick_params(colors=INK_2, labelsize=8)
            if i == 1:
                ax.set_xlabel("log-moneyness x = ln(F / K)", color=INK, fontsize=9)
            if j == 0:
                ax.set_ylabel("log10 total volatility s", color=INK, fontsize=9)
    bar = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.01)
    bar.set_label(
        f"log10 {key}  (values below {floor:g}, including exact, shown at {floor:g})",
        color=INK,
        fontsize=9,
    )
    bar.ax.tick_params(colors=INK_2, labelsize=8)
    handles = [
        Patch(color=UNINFORMATIVE, label="not informative: c = 0, subnormal, or kappa eps >= 1"),
        Patch(color=FAILED, label="no value returned"),
    ]
    fig.legend(handles=handles, loc="lower left", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(
        f"{title}. Solid line: SPEC 10 well-posed boundary (c >= eps max(F, K)); "
        "dashed: informative region (post hoc).",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_outcomes(config: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    xs, ss = grid_axes(config)
    log_s = np.log10(ss)
    codes = {"CONVERGED": 0, "AT_LOWER_BOUND": 1, "INFEASIBLE": 3}
    cmap = ListedColormap([NEUTRAL, BLUE, ORANGE, AQUA])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    present: set[int] = set()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), facecolor=SURFACE, sharex=True, sharey=True)
    for i, option in enumerate(config["option_types"]):
        index = {(r["x"], r["s"]): r for r in rows if r["option_type"] == option}
        for j, m in enumerate(config["methods"]):
            grid = np.array([[codes.get(index[(x, s)][f"{m}_outcome"], 2) for x in xs] for s in ss])
            present |= {int(v) for v in np.unique(grid)}
            ax = axes[i, j]
            ax.pcolormesh(xs, log_s, grid, cmap=cmap, norm=norm, shading="nearest")
            _boundaries(ax, rows, xs, ss, option)
            ax.set_title(f"{LABEL[m]}, {option.lower()}s", color=INK, fontsize=10, loc="left")
            ax.tick_params(colors=INK_2, labelsize=8)
            if i == 1:
                ax.set_xlabel("log-moneyness x", color=INK, fontsize=9)
            if j == 0:
                ax.set_ylabel("log10 total volatility s", color=INK, fontsize=9)
    labels = {
        0: (NEUTRAL, "converged"),
        1: (BLUE, "returned s = 0 (lower bound)"),
        2: (ORANGE, "failed: no value (diverged, zero derivative, cap)"),
        3: (AQUA, "rounded target infeasible"),
    }
    handles = [Patch(color=labels[c][0], label=labels[c][1]) for c in sorted(present)]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    fig.suptitle(
        "Solver outcomes over the grid. Solid line: SPEC 10 well-posed boundary; "
        "dashed: informative region (post hoc).",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_iterations(config: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE, sharey=True)
    informative_rows = [r for r in rows if r["informative"]]
    for ax, m in zip(axes, ("brent", "naive_newton", "safeguarded_newton"), strict=True):
        converged = [r for r in informative_rows if r[f"{m}_outcome"] == "CONVERGED"]
        its = _values(converged, f"{m}_iterations")
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.hist(its, bins=np.arange(-0.5, max(its.max(), 2) + 1.5), color=BLUE, rwidth=0.85)
        ax.axvline(2, color=INK, linewidth=1, linestyle="--")
        ax.annotate(
            "Let's Be Rational: 2",
            (2, 0.97),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=INK,
        )
        ax.set_title(
            f"{LABEL[m]} (n = {its.size}, median {np.median(its):g})",
            color=INK,
            fontsize=10,
            loc="left",
        )
        ax.set_xlabel("iterations", color=INK, fontsize=9)
        ax.tick_params(colors=INK_2, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    axes[0].set_ylabel("converged informative points", color=INK, fontsize=9)
    fig.suptitle(
        "Iterations to converge, informative region (post hoc)",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_surfaces(config: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    xs, ss = grid_axes(config)
    X, Y = np.meshgrid(xs, np.log10(ss))
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    surfaces = {
        m: _informative_only(rows, xs, ss, "CALL", f"{m}_e_norm", 0.1) for m in config["methods"]
    }
    high = max(float(z.max()) for z in surfaces.values() if z.count())
    fig = plt.figure(figsize=(17, 5.4), facecolor=SURFACE)
    for k, m in enumerate(config["methods"]):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        ax.plot_surface(
            X,
            Y,
            surfaces[m].filled(np.nan),
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            vmin=-1.0,
            vmax=high,
        )
        ax.set_zlim(-1.0, high)
        ax.set_facecolor(SURFACE)
        ax.set_xlabel("x", color=INK, fontsize=9)
        ax.set_ylabel("log10 s", color=INK, fontsize=9)
        ax.set_zlabel("log10 e_norm", color=INK, fontsize=9)
        ax.set_title(f"{LABEL[m]} (calls)", color=INK, fontsize=10)
        ax.view_init(elev=30, azim=-60)
        ax.tick_params(colors=INK_2, labelsize=7)
    fig.suptitle(
        "Normalized error over the informative region, 3D view of the e_norm heatmaps. "
        "Gaps: not informative or no value returned; e_norm below 0.1 shown at 0.1.",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --- Main ---------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    env = environment(LIBRARIES)

    rows = run(config)
    analysis = analyse(config, rows)

    out = config["output"]
    table, figures = Path(out["table_csv"]), Path(out["figure_dir"])
    table.parent.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {"config": config, "environment": env, **analysis}
    Path(out["summary_json"]).write_text(json.dumps(record, indent=2, default=str) + "\n")
    plot_heatmaps(
        config,
        rows,
        "e_ulp",
        0.5,
        "Error in units in the last place of s",
        figures / "study_a_e_ulp.png",
    )
    plot_heatmaps(
        config,
        rows,
        "e_norm",
        0.1,
        "Normalized error e_norm = e_rel / (max(kappa, 1) eps)",
        figures / "study_a_e_norm.png",
    )
    plot_outcomes(config, rows, figures / "study_a_outcomes.png")
    plot_iterations(config, rows, figures / "study_a_iterations.png")
    plot_surfaces(config, rows, figures / "study_a_e_norm_3d.png")

    print(f"commit {env['commit'][:7]}{' (dirty)' if env['dirty'] else ''}")
    for label, part in analysis.items():
        print(f"=== {label} ===")
        print(json.dumps(part["stats"], indent=1, default=str))
        for m, v in part["per_method"].items():
            print(f"{m:20s} region {v['outcomes_well_posed']}")
            print(
                f"{'':20s} e_norm median {v['e_norm_median']}, p99 {v['e_norm_p99']}; iterations "
                f"median {v['iterations_median']}; time median {v['time_median_us']:.1f} us "
                f"IQR {v['time_iqr_us'][0]:.1f}-{v['time_iqr_us'][1]:.1f}"
            )
        for name, passed in part["verdicts"].items():
            print(f"{name}: {'SUPPORTED' if passed else 'REFUTED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
