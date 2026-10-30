"""Finite-difference bump study (SPEC 7).

Pre-registered in docs/notes/phase_3_bump_study.md. Run from the repository root:

    python -m experiments.greek_bumps --config experiments/configs/greek_bumps.toml

For each Greek it sweeps the relative bump eta, compares the finite difference with the analytic
Greek, fits the slopes of the two branches of the error "V", locates the optimal eta, and checks
one-sided differences near T = 0. It writes a table, a JSON summary with the registered checks,
and three figures: the V-shape, a 3D error surface (with its 2D heatmap), and the one-sided
comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from experiments.provenance import environment
from pricing import black_scholes as bs
from pricing.contracts import OptionType
from pricing.finite_diff import (
    central_first,
    central_second,
    forward_first,
    forward_first_order2,
    relative_step,
)

EPS = sys.float_info.epsilon
LIBRARIES = ("numpy", "scipy", "matplotlib")
GREEKS = ("delta", "gamma", "vega", "theta", "rho")
VARIABLE = {"delta": "S", "gamma": "S", "vega": "sigma", "theta": "T", "rho": "r"}
PREDICTED = {"first": EPS ** (1 / 3), "second": EPS ** (1 / 4)}  # SPEC 7

# Reference palette of the data-viz method (light surface); categorical slots in fixed order,
# validated for colour-vision deficiency. Sequential blue ramp for magnitude.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {
    "delta": "#2a78d6",
    "gamma": "#eb6834",
    "vega": "#1baf7a",
    "theta": "#eda100",
    "rho": "#e87ba4",
}
MARKERS = {"delta": "o", "gamma": "s", "vega": "^", "theta": "D", "rho": "v"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def eta_grid(config: dict[str, Any]) -> list[float]:
    e = config["eta_exponents"]
    n = round((e["stop"] - e["start"]) / e["step"]) + 1
    return [10.0 ** (e["start"] + i * e["step"]) for i in range(n)]


def order(greek: str) -> str:
    return "second" if greek == "gamma" else "first"


def analytic(greek: str, option_type: OptionType, market: dict[str, float]) -> float:
    return float(getattr(bs, greek)(option_type, **market))


def finite_difference(
    greek: str, option_type: OptionType, market: dict[str, float], eta: float, x_scale: float
) -> float:
    """Central difference for one Greek (SPEC 7); theta = -dV/dT."""
    name = VARIABLE[greek]
    x = market[name]

    def price(value: float) -> float:
        return float(bs.price(option_type, **{**market, name: value}))

    h = relative_step(x, eta=eta, x_scale=x_scale)
    if greek == "gamma":
        return central_second(price, x, h)
    derivative = central_first(price, x, h)
    return -derivative if greek == "theta" else derivative


def relative_error(estimate: float, reference: float) -> float:
    return abs(estimate - reference) / abs(reference)


def slope(log_eta: list[float], log_err: list[float]) -> float:
    """Least-squares slope of log10(error) on log10(eta)."""
    x, y = np.asarray(log_eta), np.asarray(log_err)
    xc = x - x.mean()
    return float(xc @ (y - y.mean()) / (xc @ xc))


def run(config: dict[str, Any]) -> dict[str, Any]:
    option_type = OptionType[config["option_type"]]
    market, x_scale, etas = dict(config["market"]), config["x_scale"], eta_grid(config)
    rows: list[dict[str, Any]] = []

    main: dict[str, list[float]] = {}
    for greek in GREEKS:
        ref = analytic(greek, option_type, market)
        errors = []
        for eta in etas:
            err = relative_error(finite_difference(greek, option_type, market, eta, x_scale), ref)
            errors.append(err)
            rows.append(
                {
                    "part": "main",
                    "greek": greek,
                    "S": market["S"],
                    "T": market["T"],
                    "method": "central",
                    "eta": eta,
                    "relative_error": err,
                }
            )
        main[greek] = errors

    surface: dict[str, np.ndarray] = {}
    for greek in ("delta", "gamma"):
        grid = np.empty((len(config["surface"]["S"]), len(etas)))
        for i, S in enumerate(config["surface"]["S"]):
            case = {**market, "S": S}
            ref = analytic(greek, option_type, case)
            for j, eta in enumerate(etas):
                fd = finite_difference(greek, option_type, case, eta, x_scale)
                grid[i, j] = relative_error(fd, ref)
                rows.append(
                    {
                        "part": "surface",
                        "greek": greek,
                        "S": S,
                        "T": market["T"],
                        "method": "central",
                        "eta": eta,
                        "relative_error": grid[i, j],
                    }
                )
        surface[greek] = grid

    short = {**market, "T": config["short_maturity"]["T"]}
    theta_ref = analytic("theta", option_type, short)

    def price_in_T(T: float) -> float:
        return float(bs.price(option_type, **{**short, "T": T}))

    one_sided: dict[str, list[float | None]] = {
        "forward_first": [],
        "forward_order2": [],
        "central": [],
    }
    for eta in etas:
        h = relative_step(short["T"], eta=eta, x_scale=x_scale)
        estimates: dict[str, float | None] = {
            "forward_first": -forward_first(price_in_T, short["T"], h),
            "forward_order2": -forward_first_order2(price_in_T, short["T"], h),
            # A central difference needs T - h > 0; otherwise it is unavailable (not computed).
            "central": -central_first(price_in_T, short["T"], h) if h < short["T"] else None,
        }
        for method, est in estimates.items():
            err = None if est is None else relative_error(est, theta_ref)
            one_sided[method].append(err)
            rows.append(
                {
                    "part": "short_maturity",
                    "greek": "theta",
                    "S": short["S"],
                    "T": short["T"],
                    "method": method,
                    "eta": eta,
                    "relative_error": err,
                }
            )
    return {"etas": etas, "main": main, "surface": surface, "one_sided": one_sided, "rows": rows}


def analyse(config: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    checks, etas = config["checks"], data["etas"]
    log_eta = [math.log10(e) for e in etas]

    def window_slope(errors: list[float], window: list[float]) -> float:
        idx = [i for i, le in enumerate(log_eta) if any(abs(le - w) < 1e-9 for w in window)]
        return slope([log_eta[i] for i in idx], [math.log10(errors[i]) for i in idx])

    per_greek = {}
    for greek, errors in data["main"].items():
        best = int(np.argmin(errors))
        predicted = PREDICTED[order(greek)]
        per_greek[greek] = {
            "order": order(greek),
            "right_slope": window_slope(errors, checks["right_window"]),
            "left_slope": window_slope(errors, checks["left_window"]),
            "optimal_eta": etas[best],
            "predicted_eta": predicted,
            "optimum_ratio": etas[best] / predicted,
            "best_error": errors[best],
        }

    one_sided = {
        method: window_slope([e for e in errs if e is not None], checks["one_sided_window"])
        if method != "central"
        else None
        for method, errs in data["one_sided"].items()
    }
    lo, hi = checks["right_slope_band"]
    first = [g for g in GREEKS if order(g) == "first"]
    factor = checks["optimum_factor"]
    results = {
        "D1_right_slopes_in_band": all(lo <= v["right_slope"] <= hi for v in per_greek.values()),
        "D1_left_slopes_first_in_band": all(
            checks["left_slope_band_first"][0]
            <= per_greek[g]["left_slope"]
            <= checks["left_slope_band_first"][1]
            for g in first
        ),
        "D1_left_slope_gamma_in_band": checks["left_slope_band_second"][0]
        <= per_greek["gamma"]["left_slope"]
        <= checks["left_slope_band_second"][1],
        "D2_optima_within_factor": all(
            1 / factor <= v["optimum_ratio"] <= factor for v in per_greek.values()
        ),
        "D3_best_error_first": all(
            per_greek[g]["best_error"] <= checks["best_error_first"] for g in first
        ),
        "D3_best_error_gamma": per_greek["gamma"]["best_error"] <= checks["best_error_second"],
        "D4_forward_first_slope": checks["forward_first_band"][0]
        <= one_sided["forward_first"]
        <= checks["forward_first_band"][1],
        "D4_forward_order2_slope": checks["forward_order2_band"][0]
        <= one_sided["forward_order2"]
        <= checks["forward_order2_band"][1],
    }
    return {"per_greek": per_greek, "one_sided_slopes": one_sided, "checks": results}


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def plot_vshape(
    config: dict[str, Any], data: dict[str, Any], analysis: dict[str, Any], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor=SURFACE)
    _style(ax)
    etas = data["etas"]
    for greek in GREEKS:
        errs = data["main"][greek]
        ax.loglog(
            etas,
            errs,
            color=SERIES[greek],
            linewidth=2,
            marker=MARKERS[greek],
            markersize=5,
            label=greek,
        )
        ax.annotate(
            greek,
            (etas[0], errs[0]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    for kind, label in (
        ("first", "predicted optimum, first derivatives  eps^(1/3)"),
        ("second", "predicted optimum, gamma  eps^(1/4)"),
    ):
        ax.axvline(
            PREDICTED[kind], color=INK_2, linewidth=1, linestyle="--" if kind == "first" else ":"
        )
        ax.annotate(
            label,
            (PREDICTED[kind], 0.5),
            xycoords=("data", "axes fraction"),
            rotation=90,
            fontsize=8,
            color=INK_2,
            ha="right",
            va="center",
        )
    ax.set_xlabel("relative bump size eta  (h = eta x max(|x|, 1))", color=INK)
    ax.set_ylabel("relative error vs analytic Greek", color=INK)
    m = config["market"]
    ax.set_title(
        f"Finite-difference error vs bump size: call, S = K = {m['K']:g}, T = {m['T']:g}, "
        f"sigma = {m['sigma']:g}",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    ax.set_xlim(etas[-1] / 2, etas[0] * 6)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_surface(
    config: dict[str, Any], data: dict[str, Any], path_3d: Path, path_2d: Path
) -> None:
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    log_eta = np.log10(data["etas"])
    spots = np.asarray(config["surface"]["S"])
    X, Y = np.meshgrid(log_eta, spots)

    fig = plt.figure(figsize=(12, 5.5), facecolor=SURFACE)
    for k, greek in enumerate(("delta", "gamma")):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        Z = np.log10(data["surface"][greek])
        ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95)
        ax.set_facecolor(SURFACE)
        ax.set_xlabel("log10 eta", color=INK, fontsize=9)
        ax.set_ylabel("spot S", color=INK, fontsize=9)
        ax.set_zlabel("log10 relative error", color=INK, fontsize=9)
        ax.set_title(f"{greek}: error surface", color=INK, fontsize=11)
        ax.view_init(elev=28, azim=-60)
        ax.tick_params(colors=INK_2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path_3d, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    for ax, greek in zip(axes, ("delta", "gamma"), strict=True):
        Z = np.log10(data["surface"][greek])
        mesh = ax.pcolormesh(log_eta, spots, Z, cmap=cmap, shading="nearest")
        ax.set_xlabel("log10 eta", color=INK)
        ax.set_ylabel("spot S", color=INK)
        ax.set_title(
            f"{greek}: log10 relative error (same data as the 3D view)",
            color=INK,
            fontsize=10,
            loc="left",
        )
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.axvline(math.log10(PREDICTED[order(greek)]), color=INK, linewidth=1, linestyle="--")
        fig.colorbar(mesh, ax=ax).ax.tick_params(colors=INK_2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path_2d, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_one_sided(config: dict[str, Any], data: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor=SURFACE)
    _style(ax)
    etas = data["etas"]
    styles = {
        "forward_first": ("#2a78d6", "o", "forward, first order (one-sided)"),
        "forward_order2": ("#eb6834", "s", "forward, three-point (one-sided)"),
        "central": ("#1baf7a", "^", "central (needs T - h > 0)"),
    }
    for method, (color, marker, label) in styles.items():
        pts = [
            (e, err)
            for e, err in zip(etas, data["one_sided"][method], strict=True)
            if err is not None
        ]
        ax.loglog(
            *zip(*pts, strict=True),
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
            label=label,
        )
        ax.annotate(
            label.split(",")[0].split(" (")[0],
            pts[0],
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            va="center",
        )
    T = config["short_maturity"]["T"]
    ax.axvline(T, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate(
        "h = T: central difference unavailable to the right",
        (T, 0.5),
        xycoords=("data", "axes fraction"),
        rotation=90,
        fontsize=8,
        color=INK_2,
        ha="right",
        va="center",
    )
    ax.set_xlabel("relative bump size eta  (h = eta, since T < 1)", color=INK)
    ax.set_ylabel("relative error of theta", color=INK)
    ax.set_title(
        "Theta one day before expiry: one-sided vs central differences",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    env = environment(LIBRARIES)

    data = run(config)
    analysis = analyse(config, data)

    out = config["output"]
    table = Path(out["table_csv"])
    figures = Path(out["figure_dir"])
    table.parent.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data["rows"][0]))
        writer.writeheader()
        writer.writerows(data["rows"])
    record = {"config": config, "environment": env, **analysis}
    Path(out["summary_json"]).write_text(json.dumps(record, indent=2) + "\n")
    plot_vshape(config, data, analysis, figures / "greek_bumps_vshape.png")
    plot_surface(
        config,
        data,
        figures / "greek_bumps_surface_3d.png",
        figures / "greek_bumps_surface_heatmap.png",
    )
    plot_one_sided(config, data, figures / "greek_bumps_one_sided.png")

    print(f"commit {env['commit'][:7]}{' (dirty)' if env['dirty'] else ''}")
    for greek, v in analysis["per_greek"].items():
        print(
            f"{greek:6s} right slope {v['right_slope']:+.2f}  left slope {v['left_slope']:+.2f}"
            f"  optimal eta {v['optimal_eta']:.1e} (predicted {v['predicted_eta']:.1e}, ratio "
            f"{v['optimum_ratio']:.2f})  best error {v['best_error']:.1e}"
        )
    for method, s in analysis["one_sided_slopes"].items():
        if s is not None:
            print(f"one-sided {method}: right slope {s:+.2f}")
    for name, passed in analysis["checks"].items():
        print(f"{name}: {'SUPPORTED' if passed else 'REFUTED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
