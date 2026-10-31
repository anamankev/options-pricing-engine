"""Helpers of experiments/study_a2_iv.py (definitions registered in docs/notes/study_a2.md)."""

from pathlib import Path
from typing import Any

from experiments.study_a2_iv import evaluate, grid_config, verdicts
from experiments.study_a_iv import grid_axes, load_config

CONFIG = load_config(Path(__file__).resolve().parents[1] / "experiments/configs/study_a2_iv.toml")
METHODS = ("brent", "naive_newton", "safeguarded_newton", "lets_be_rational")


def test_grids_match_the_registration() -> None:
    sizes = {}
    for grid in CONFIG["grids"]:
        xs, ss = grid_axes(grid_config(CONFIG, grid))
        sizes[grid["name"]] = (len(xs), len(ss))
        if grid["name"] == "new_grid":
            on_grid_a = {round(x / 0.25, 9) % 1 == 0 for x in xs}
            assert on_grid_a == {False}  # no x shared with Study A's multiples of 0.25
    assert sizes == {"grid_a": (49, 60), "new_grid": (48, 57)}


def _row(x: float, s: float, informative: bool, sn: str, lbr: str, sn_e: float) -> dict[str, Any]:
    row: dict[str, Any] = {"option_type": "CALL", "x": x, "s": s, "informative": informative}
    for m in METHODS:
        outcome = (
            sn if m == "safeguarded_newton" else (lbr if m == "lets_be_rational" else "CONVERGED")
        )
        row |= {
            f"{m}_outcome": outcome,
            f"{m}_e_norm": sn_e if outcome == "CONVERGED" else None,
            f"{m}_iterations": 5,
            f"{m}_time_ns": 1000.0,
        }
    return row


def test_evaluate_scores_only_informative_points_and_finds_misses() -> None:
    rows = [
        _row(0.0, 1e-3, True, "CONVERGED", "CONVERGED", 1.0),  # near money (H1 region)
        _row(3.0, 0.5, True, "AT_LOWER_BOUND", "CONVERGED", 0.0),  # an H2 miss
        _row(0.0, 1e-3, False, "CONVERGED", "CONVERGED", 999.0),  # not informative: ignored
    ]
    stats = evaluate(CONFIG, rows)
    assert stats["n_informative"] == 2
    assert stats["h1_n"] == 1
    assert stats["h1_sn_median_enorm"] == 1.0
    assert [m["x"] for m in stats["h2_sn_missed"]] == [3.0]


def test_verdicts_apply_the_registered_thresholds() -> None:
    good = evaluate(CONFIG, [_row(0.0, 1e-3, True, "CONVERGED", "CONVERGED", 1.0)])
    bad = evaluate(CONFIG, [_row(0.0, 1e-3, True, "CONVERGED", "CONVERGED", 683.0)])
    assert all(verdicts(CONFIG, {"grid_a": good, "new_grid": good}).values())
    result = verdicts(CONFIG, {"grid_a": bad, "new_grid": good})
    assert not result["H1"]  # 683 > 4 on grid A
    assert not result["H3"]


FOLLOWUP = load_config(
    Path(__file__).resolve().parents[1] / "experiments/configs/study_a2_followup.toml"
)


def _points(config: dict[str, Any], name: str) -> set[tuple[float, float]]:
    grid = next(g for g in config["grids"] if g["name"] == name)
    xs, ss = grid_axes(grid_config(config, grid))
    return {(round(x, 12), round(s, 15)) for x in xs for s in ss}


def test_confirmation_grid_shares_no_point_with_earlier_grids() -> None:
    confirm = _points(FOLLOWUP, "confirm_grid")
    assert len(confirm) == 35 * 53  # registered: 35 x 53 points per option type
    assert not confirm & _points(CONFIG, "grid_a")
    assert not confirm & _points(CONFIG, "new_grid")


def test_followup_hypotheses_are_judged_on_the_confirmation_grid_only() -> None:
    good = evaluate(FOLLOWUP, [_row(0.0, 1e-3, True, "CONVERGED", "CONVERGED", 1.0)])
    missed = evaluate(FOLLOWUP, [_row(3.0, 0.5, True, "AT_LOWER_BOUND", "CONVERGED", 0.0)])
    by_grid = {
        "grid_a": verdicts(FOLLOWUP, {"grid_a": missed}),  # post hoc: does not count
        "confirm_grid": verdicts(FOLLOWUP, {"confirm_grid": good}),
    }
    assert by_grid["confirm_grid"]["H2"]
    assert "H5" in by_grid["confirm_grid"]  # the fresh grid
    assert "H5" not in by_grid["grid_a"]
