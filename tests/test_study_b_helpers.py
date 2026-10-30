"""Helpers of experiments/study_b_trees.py (definitions registered in docs/notes/study_b.md,
section 3)."""

import pytest
from experiments.study_b_trees import (
    american_cases,
    best_within_budget,
    envelope,
    fit_order,
    oscillation,
    step_counts,
)

EPS = 2.220446049250313e-16
# A least-squares slope over an exact power law: a few roundings of logs and dot products.
FIT_TOL = 1e-12


def test_step_counts_follow_spec_11() -> None:
    steps = step_counts(
        {
            "steps": {
                "crr": {"start": 10, "stop": 2000},
                "lr": {"start": 11, "stop": 2001},
                "bbs_k": {"start": 4, "stop": 13},
            }
        }
    )
    assert steps["crr"][0] == 10
    assert steps["crr"][-1] == 2000
    assert len(steps["crr"]) == 1991
    assert all(n % 2 for n in steps["lr"])
    assert steps["lr"][-1] == 2001
    assert steps["bbs"] == [2**k for k in range(4, 14)]


def test_fit_order_recovers_power_law_and_excludes_unresolved_errors() -> None:
    ns = [100, 200, 400, 800, 1600]
    errors = [3.0 * n**-2.0 for n in ns]
    fit = fit_order(ns, errors, resolution=1e-30)
    assert fit["slope"] == pytest.approx(-2.0, abs=FIT_TOL)
    assert fit["slope_se"] <= FIT_TOL
    # The last error (1.2e-6) is below a resolution of 2e-6, so it is excluded.
    assert fit_order(ns, errors, resolution=2e-6)["n_excluded"] == 1


def test_oscillation_metric() -> None:
    # Alternating signs give |e_N - e_{N+1}| / max = 2; a smooth sequence gives a small ratio.
    assert oscillation([1, 2, 3, 4], [1.0, -1.0, 1.0, -1.0], 1, 3) == 2.0
    assert oscillation([1, 2, 3], [1.0, 0.9, 0.81], 1, 2) == pytest.approx(0.1, rel=1e-12)


def test_envelope_takes_block_maxima() -> None:
    centres, peaks = envelope(list(range(10, 30)), [0.1 * (-1) ** n for n in range(10, 30)], 10, 10)
    assert centres == [15, 25]
    assert peaks == [0.1, 0.1]


def test_best_within_budget_picks_largest_affordable_n() -> None:
    times = {"crr": {10: 0.5, 20: 2.0}, "bbsr": {16: 3.0}}
    errors = {"crr": {10: 0.1, 20: -0.05}, "bbsr": {16: 0.001}}
    assert best_within_budget(times, errors, 1.0) == {"crr": 0.1, "bbsr": None}
    assert best_within_budget(times, errors, 5.0) == {"crr": 0.05, "bbsr": 0.001}


def test_american_cases_are_the_18_ju_zhong_values() -> None:
    from pathlib import Path

    from experiments.study_b_trees import load_config

    config = load_config(
        Path(__file__).resolve().parents[1] / "experiments/configs/study_b_trees.toml"
    )
    cases = american_cases(config)
    assert len(cases) == 18
    assert sum(c["type"] == "PUT" for c in cases) == 13
    assert cases[2]["T"] == 210 / 360  # R4c: printed 0.5833, represented exactly as 7/12
