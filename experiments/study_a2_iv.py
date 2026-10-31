"""Study A2: do the solver fixes close Study A's gaps? (docs/notes/study_a2.md)

Run from the repository root:

    python -m experiments.study_a2_iv --config experiments/configs/study_a2_iv.toml

It runs Study A's experiment code unchanged (`experiments.study_a_iv.run`) on two grids, Study
A's own and a new one, then evaluates the six registered hypotheses on the informative points of
each grid. For grid A it also reads Study A's committed results to show before and after on
identical points.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.provenance import environment
from experiments.study_a_iv import load_config, plot_heatmaps, plot_surfaces, run

LIBRARIES = ("numpy", "scipy", "mpmath", "matplotlib", "lets-be-rational")
SN = "safeguarded_newton"


def grid_config(config: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    """Study A's config shape for one grid."""
    shared = {k: config[k] for k in ("option_types", "mp_digits", "methods", "timing")}
    return {**shared, **{k: grid[k] for k in ("F", "T", "x", "s")}}


def _informative(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["informative"] in (True, "True")]


def _floats(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows if r[key] not in (None, "")], dtype=float)


def _stat(values: np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if values.size else None


def evaluate(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The registered statistics on one grid's informative points."""
    checks = config["checks"]
    info = _informative(rows)
    region = checks["h1_region"]
    near = [
        r
        for r in info
        if abs(float(r["x"])) <= region["abs_x_max"] and float(r["s"]) <= region["s_max"]
    ]
    lbr_solved = [r for r in info if r["lets_be_rational_outcome"] == "CONVERGED"]
    missed = [
        {
            "option_type": r["option_type"],
            "x": r["x"],
            "s": r["s"],
            "sn_outcome": r[f"{SN}_outcome"],
        }
        for r in lbr_solved
        if r[f"{SN}_outcome"] != "CONVERGED"
    ]
    bad = set(checks["h4_bad_outcomes"])
    per_method = {}
    for m in config["methods"]:
        e = _floats(info, f"{m}_e_norm")
        it = _floats(info, f"{m}_iterations")
        t = _floats(info, f"{m}_time_ns") / 1e3
        outcomes: dict[str, int] = {}
        for r in info:
            outcomes[r[f"{m}_outcome"]] = outcomes.get(r[f"{m}_outcome"], 0) + 1
        per_method[m] = {
            "outcomes_informative": outcomes,
            "e_norm_median": _stat(e, 50),
            "e_norm_p99": _stat(e, 99),
            "iterations_median": _stat(it, 50),
            "time_median_us": _stat(t, 50),
        }
    near_sn = _floats(near, f"{SN}_e_norm")
    return {
        "n_points": len(rows),
        "n_informative": len(info),
        "h1_n": len(near),
        "h1_sn_median_enorm": _stat(near_sn, 50),
        "h1_lbr_median_enorm": _stat(_floats(near, "lets_be_rational_e_norm"), 50),
        "h2_lbr_converged": len(lbr_solved),
        "h2_sn_missed": missed,
        "h4_bad": {m: sum(r[f"{m}_outcome"] in bad for r in info) for m in ("brent", SN)},
        "per_method": per_method,
    }


def _within(value: float | None, limit: float) -> bool:
    """A statistic that could not be computed (no converged solves) does not pass."""
    return value is not None and value <= limit


def verdicts(config: dict[str, Any], stats: dict[str, dict[str, Any]]) -> dict[str, bool]:
    """H1 to H4 and H6 must hold on every grid in `stats`; H5 on the grids never used before
    (config `fresh_grids`, default the new grid), when any are present."""
    c = config["checks"]
    grids = list(stats.values())
    sn = [g["per_method"][SN] for g in grids]
    fresh = [stats[name] for name in config.get("fresh_grids", ["new_grid"]) if name in stats]
    result = {
        "H1": all(_within(g["h1_sn_median_enorm"], c["h1_max_median_enorm"]) for g in grids),
        "H2": all(not g["h2_sn_missed"] for g in grids),
        "H3": all(
            _within(s["e_norm_median"], c["h3_max_median_enorm"])
            and _within(s["e_norm_p99"], c["h3_max_p99_enorm"])
            for s in sn
        ),
        "H4": all(sum(g["h4_bad"].values()) == 0 for g in grids),
        "H6": all(_within(s["iterations_median"], c["h6_max_median_iterations"]) for s in sn),
    }
    if fresh:
        result["H5"] = all(
            _within(g["per_method"]["lets_be_rational"]["e_norm_p99"], c["h5_max_p99_enorm_lbr"])
            for g in fresh
        )
    return dict(sorted(result.items()))


def before(config: dict[str, Any]) -> dict[str, Any]:
    """Study A's committed results on grid A, evaluated the same way (read, not recomputed)."""
    with Path(config["before_table"]).open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v == "":
                r[k] = None
    return evaluate(config, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    env = environment(LIBRARIES)
    out = config["output"]
    figures = Path(out["figure_dir"])
    prefix = out.get("figure_prefix", "study_a2")
    figures.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, Any]] = {}
    for grid in config["grids"]:
        gconfig = grid_config(config, grid)
        rows = run(gconfig)
        table = Path(out["table_csv_pattern"].format(grid=grid["name"]))
        table.parent.mkdir(parents=True, exist_ok=True)
        with table.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        stats[grid["name"]] = evaluate(config, rows)
        plot_heatmaps(
            gconfig,
            rows,
            "e_norm",
            0.1,
            f"Study A2 ({grid['name']}): e_norm = e_rel / (max(kappa, 1) eps)",
            figures / f"{prefix}_{grid['name']}_e_norm.png",
        )
        plot_surfaces(gconfig, rows, figures / f"{prefix}_{grid['name']}_e_norm_3d.png")

    record: dict[str, Any] = {
        "config": config,
        "environment": env,
        "grids": stats,
        "verdicts": verdicts(config, stats),
        "verdicts_by_grid": {name: verdicts(config, {name: s}) for name, s in stats.items()},
    }
    if "before_table" in config:
        record["before_grid_a"] = before(config)
    if "followup" in config:  # registered hypotheses judged on one grid (note, section 9)
        f = config["followup"]
        judged = record["verdicts_by_grid"][f["grid"]]
        record["followup_verdicts"] = {name: judged[h] for name, h in f["hypotheses"].items()}
    Path(out["summary_json"]).write_text(json.dumps(record, indent=2, default=str) + "\n")

    print(f"commit {env['commit'][:7]}{' (dirty)' if env['dirty'] else ''}")
    shown = [*stats.items()]
    if "before_grid_a" in record:
        shown.append(("before (Study A, grid A)", record["before_grid_a"]))
    for name, s in shown:
        sn = s["per_method"][SN]
        print(f"=== {name}: {s['n_informative']} informative of {s['n_points']}")
        print(f"  H1 near money SN median e_norm {s['h1_sn_median_enorm']} (n={s['h1_n']})")
        print(f"  H2 LBR converged {s['h2_lbr_converged']}, SN missed {len(s['h2_sn_missed'])}")
        print(f"  H3 SN e_norm median {sn['e_norm_median']}, p99 {sn['e_norm_p99']}")
        print(f"  H4 bad outcomes {s['h4_bad']}; H6 SN iterations median {sn['iterations_median']}")
    for label, table in (("all grids", record["verdicts"]), *record["verdicts_by_grid"].items()):
        print(label, {k: "SUPPORTED" if ok else "REFUTED" for k, ok in table.items()})
    for name, ok in record.get("followup_verdicts", {}).items():
        print(f"{name}: {'SUPPORTED' if ok else 'REFUTED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
