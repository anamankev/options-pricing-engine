"""The checks `python -m verify` runs. Each returns a Result: what was checked, what was measured,
the limit it was held to, and PASS, FAIL or SKIP.

The quick rechecks recompute headline results on this machine and hold them to the same named
tolerances the tests use, so a PASS here means the same thing as a passing test.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from verify.ledger import ROOT, check_all, uncovered_numbers

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
TABLES = ROOT / "experiments" / "results" / "tables"
ORACLE_HINT = 'QuantLib not installed (install it with: pip install -e ".[oracle]")'


@dataclass(frozen=True)
class Result:
    name: str
    status: str
    measured: str
    limit: str = ""
    detail: str = ""
    seconds: float = 0.0


def oracle_available() -> bool:
    return importlib.util.find_spec("QuantLib") is not None


def _timed(fn: Callable[[], Result]) -> Result:
    start = time.perf_counter()
    result = fn()
    return Result(**{**result.__dict__, "seconds": time.perf_counter() - start})


def _committed(file: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((TABLES / file).read_text())
    return data


def tracked_test_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "tests/test_*.py"], capture_output=True, text=True, cwd=ROOT
    ).stdout.split()
    return out or sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("tests/test_*.py"))


# --- 1. The test suite -------------------------------------------------------------------------


def test_suite() -> Result:
    marker = [] if oracle_available() else ["-m", "not oracle"]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *marker,
            *tracked_test_files(),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-300:]
    summary = re.sub(r"\s+in [\d.]+s.*$", "", summary.strip("= "))
    note = "" if oracle_available() else "oracle tests not run: " + ORACLE_HINT
    status = PASS if proc.returncode == 0 else FAIL
    detail = note if status == PASS else proc.stdout[-3000:]
    return Result("Test suite (every test in the repository)", status, summary, "", detail)


# --- 2. The README's numbers -------------------------------------------------------------------


def readme_claims() -> Result:
    outcomes = check_all()
    failed = [o for o in outcomes if not o.passed]
    missing = uncovered_numbers()
    measured = f"{len(outcomes) - len(failed)} of {len(outcomes)} numbers match their source"
    problems = [f"{o.claim.id}: {o.detail}" for o in failed]
    problems += [f"number {n!r} has no source, in: {line.strip()}" for n, line in missing]
    status = PASS if not problems else FAIL
    detail = "\n".join(problems) if problems else "no number in Status or Results lacks a source"
    return Result("README numbers (docs/claims.toml)", status, measured, "all", detail)


# --- 3. Quick rechecks of headline results ------------------------------------------------------


def bsm_against_mpmath() -> Result:
    from experiments.validation import bsm_vs_mpmath
    from tests.test_black_scholes import ROUNDING_FACTOR

    got = bsm_vs_mpmath()
    worst = got["worst_error_eps_scale"]
    committed = _committed("validation_summary.json")["bsm_vs_mpmath"]["worst_error_eps_scale"]
    return Result(
        f"Black-Scholes prices vs 50-digit mpmath ({got['cases']} cases)",
        PASS if worst <= ROUNDING_FACTOR else FAIL,
        f"worst error {worst:.2f} eps x scale (README: {committed:.2f})",
        f"<= {ROUNDING_FACTOR}",
    )


def greeks_against_finite_differences() -> Result:
    from experiments.validation import greeks_vs_finite_differences
    from tests.test_black_scholes_greeks import FD_RTOL

    got = greeks_vs_finite_differences()
    worst = got["worst_relative_error_any"]
    return Result(
        f"Six Greeks vs finite differences ({got['cases_per_greek']} cases each)",
        PASS if worst <= FD_RTOL else FAIL,
        f"worst relative error {worst:.1e}",
        f"<= {FD_RTOL:.0e}",
    )


def implied_vol_round_trips() -> Result:
    from experiments.validation import iv_round_trips
    from tests.test_implied_vol import IV_ROUNDING_FACTOR

    got = iv_round_trips()
    methods = ("safeguarded_newton", "brent")
    worst = max(got[m]["worst_error_in_rounding_units"] for m in methods)
    leaked = sum(got[m]["value_returned_without_convergence"] for m in methods)
    ok = worst <= IV_ROUNDING_FACTOR and leaked == 0
    return Result(
        f"Implied-vol round trips ({got['cases']} cases, two solvers)",
        PASS if ok else FAIL,
        f"worst error {worst:.1f} rounding units; unconverged values returned: {leaked}",
        f"<= {IV_ROUNDING_FACTOR}; 0",
    )


def american_benchmarks() -> Result:
    from experiments.validation import juzhong
    from tests.test_reference_values import JUZHONG_ABS_TOL

    got = juzhong()
    worst = got["worst_abs_difference"]
    return Result(
        f"American prices vs Ju and Zhong (1999) ({got['cases']} published values)",
        PASS if worst <= JUZHONG_ABS_TOL else FAIL,
        f"worst difference {worst:.1e} (Leisen-Reimer and BBSR)",
        f"<= {JUZHONG_ABS_TOL:.0e}",
    )


def european_against_quantlib(sample: int = 50) -> Result:
    name = f"Prices and Greeks vs QuantLib ({sample} of the 1800 reconciliation cases)"
    if not oracle_available():
        return Result(name, SKIP, "", "", ORACLE_HINT)
    from experiments.quantlib_reconciliation import (
        QUANTITIES,
        compare_case,
        iter_cases,
        load_config,
    )
    from tests.test_oracles import ORACLE_ABS_EPS_SCALE, ORACLE_RTOL

    cases = list(iter_cases(load_config(ROOT / "experiments/configs/quantlib_reconciliation.toml")))
    picked = cases[:: max(1, len(cases) // sample)][:sample]
    disagreements, medians = 0, []
    for case in picked:
        row = compare_case(case)
        for q in QUANTITIES:
            close = row[f"{q}_rel_err"] <= ORACLE_RTOL
            if not (close or row[f"{q}_abs_err_eps_scale"] <= ORACLE_ABS_EPS_SCALE):
                disagreements += 1
            medians.append(row[f"{q}_rel_err"])
    median = float(np.median([m for m in medians if math.isfinite(m)]))
    return Result(
        name,
        PASS if disagreements == 0 else FAIL,
        f"{disagreements} disagreements; median relative difference {median:.0e}",
        f"each within {ORACLE_RTOL:.0e} relative or {ORACLE_ABS_EPS_SCALE:g} eps x scale",
    )


def trees_against_quantlib() -> Result:
    name = "Trees vs QuantLib's trees (192 cases)"
    if not oracle_available():
        return Result(name, SKIP, "", "", ORACLE_HINT)
    from experiments.validation import trees_vs_quantlib
    from tests.test_oracles import TREE_ORACLE_RTOL

    got = trees_vs_quantlib()
    lr = got["lr_worst_relative_difference"]
    crr = got["crr_with_quantlib_probability_worst_relative_difference"]
    return Result(
        name,
        PASS if max(lr, crr) <= TREE_ORACLE_RTOL else FAIL,
        f"Leisen-Reimer {lr:.1e}; CRR with QuantLib's probability {crr:.1e}",
        f"<= {TREE_ORACLE_RTOL:.0e}",
    )


# Monte Carlo: one cell of the coverage study, fewer replications. With R replications the
# coverage estimate has standard error sqrt(0.95 x 0.05 / R); 4 standard errors either side of
# 0.95 is the band (a false FAIL has probability about 6e-5).
MC_REPLICATIONS = 400
MC_PATHS = 10_000
MC_BAND_SE = 4.0
MC_SEED = 20261030  # the coverage study's root seed


def monte_carlo_coverage() -> Result:
    from pricing import black_scholes as bs
    from pricing.contracts import OptionType
    from pricing.monte_carlo import MCMethod, monte_carlo

    market = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2}
    truth = float(bs.price(OptionType.CALL, **market))
    covered = 0
    for rep in range(MC_REPLICATIONS):
        seed = np.random.SeedSequence(MC_SEED, spawn_key=(99, rep))
        res = monte_carlo(
            OptionType.CALL, **market, n_paths=MC_PATHS, seed=seed, method=MCMethod.PLAIN
        )
        covered += res.ci_low <= truth <= res.ci_high
    coverage = covered / MC_REPLICATIONS
    half_width = MC_BAND_SE * math.sqrt(0.95 * 0.05 / MC_REPLICATIONS)
    return Result(
        f"Monte Carlo 95% intervals cover the exact price ({MC_REPLICATIONS} runs of "
        f"{MC_PATHS:,} paths, fixed seed)",
        PASS if abs(coverage - 0.95) <= half_width else FAIL,
        f"coverage {coverage:.3f}",
        f"0.95 +/- {half_width:.3f}",
    )


QUICK_CHECKS: tuple[Callable[[], Result], ...] = (
    bsm_against_mpmath,
    greeks_against_finite_differences,
    implied_vol_round_trips,
    american_benchmarks,
    european_against_quantlib,
    trees_against_quantlib,
    monte_carlo_coverage,
)


def run_quick() -> list[Result]:
    return [_timed(test_suite), _timed(readme_claims), *(_timed(c) for c in QUICK_CHECKS)]
