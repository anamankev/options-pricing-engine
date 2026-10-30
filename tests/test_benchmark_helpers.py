"""Helpers of experiments/benchmarks.py (definitions registered in docs/notes/benchmarks.md,
section 3)."""

import gc

import pytest
from experiments.benchmarks import hardware_and_threads, loglog_slope, measure

# A least-squares slope over an exact power law: a few roundings of logs and dot products.
FIT_TOL = 1e-12


def test_measure_reports_ordered_quantiles_and_restores_gc() -> None:
    calls = []
    gc.enable()
    result = measure(lambda: calls.append(1), repeats=20, warmup=3)
    assert len(calls) == 23  # warm-up calls run but are not timed
    assert result["repeats"] == 20
    assert 0.0 <= result["p25_s"] <= result["median_s"] <= result["p75_s"]
    assert gc.isenabled()  # collector state restored


def test_measure_leaves_a_disabled_gc_disabled() -> None:
    gc.disable()
    try:
        measure(lambda: None, repeats=2, warmup=0)
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_loglog_slope_recovers_quadratic_scaling() -> None:
    ns = [400, 800, 1600, 3200]
    assert loglog_slope(ns, [3e-9 * n**2 for n in ns]) == pytest.approx(2.0, abs=FIT_TOL)


def test_environment_record_has_spec_12_fields() -> None:
    env = hardware_and_threads()
    assert set(env) == {"cpu", "logical_cpus", "os", "blas", "thread_variables"}
    assert env["cpu"]
    assert "OMP_NUM_THREADS" in env["thread_variables"]
