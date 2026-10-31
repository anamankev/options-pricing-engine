"""Check that everything this project claims is true, on this machine, with one command.

    python -m verify            # about half a minute: tests, README numbers, quick rechecks
    python -m verify --full     # also reruns every experiment and compares with the committed
                                # results (about 3 minutes with the oracle extra installed)
    python -m verify --report verification.md   # also save the report as Markdown

Each line says what was checked, what was measured, the limit it was held to, and PASS, FAIL or
SKIP (with the reason). The command exits with status 1 if anything failed.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from verify.checks import FAIL, ORACLE_HINT, PASS, SKIP, Result, oracle_available, run_quick
from verify.ledger import ROOT

RESULTS = "experiments/results"

# The README's reproduction table: (experiment, summary file, needs QuantLib).
EXPERIMENTS = (
    ("validation", "validation_summary.json", False),
    ("greek_bumps", "greek_bumps_summary.json", False),
    ("mc_coverage", "mc_coverage_summary.json", False),
    ("benchmarks", "benchmarks_summary.json", False),
    ("quantlib_reconciliation", "quantlib_reconciliation_summary.json", True),
    ("study_a_iv", "study_a_iv_summary.json", True),
    ("study_b_trees", "study_b_summary.json", True),
)
# Never compared: provenance, and anything measured in seconds, which depends on the machine and
# its load. Benchmarks are compared on their verdicts only. Study B's B4 ranks methods by error at
# a fixed runtime budget, so it is timing-dependent too.
NEVER_COMPARED = ("environment", "config")
TIMING_WORDS = ("time", "timing", "b4_pairs", "B4_")
COMPARE_ONLY = {"benchmarks": ("verdicts",)}
RELATIVE_TOLERANCE = 1e-9  # same code, same inputs: differences beyond rounding are real changes


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=ROOT, check=False
    ).stdout.strip()


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out |= _flatten(value, f"{prefix}.{key}" if prefix else str(key))
        return out
    if isinstance(node, list):
        out = {}
        for i, value in enumerate(node):
            out |= _flatten(value, f"{prefix}[{i}]")
        return out
    return {prefix: node}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a == b:
            return True
        scale = max(abs(a), abs(b))
        return abs(a - b) <= RELATIVE_TOLERANCE * scale
    return bool(a == b)


def compare_summaries(
    experiment: str, committed: dict[str, Any], fresh: dict[str, Any]
) -> list[str]:
    """Keys whose regenerated value differs from the committed one."""
    keep = COMPARE_ONLY.get(experiment)
    old, new = _flatten(committed), _flatten(fresh)
    differences = []
    for key in sorted(set(old) | set(new)):
        top = key.split(".")[0].split("[")[0]
        if top in NEVER_COMPARED or any(word in key for word in TIMING_WORDS):
            continue
        if keep is not None and top not in keep:
            continue
        if key not in old or key not in new or not _same(old[key], new[key]):
            differences.append(f"{key}: committed {old.get(key)!r}, now {new.get(key)!r}")
    return differences


def rerun_experiments() -> list[Result]:
    """Run every README reproduction command, compare with the committed results, then restore
    the committed files so the working copy is left as it was."""
    if _git("status", "--porcelain", "--", RESULTS):
        return [
            Result(
                "Rerun every experiment",
                FAIL,
                "",
                "",
                f"{RESULTS} has uncommitted changes; commit or discard them first",
            )
        ]
    results = []
    try:
        for experiment, summary, needs_oracle in EXPERIMENTS:
            name = f"Rerun {experiment}"
            command = [
                sys.executable,
                "-m",
                f"experiments.{experiment}",
                "--config",
                f"experiments/configs/{experiment}.toml",
            ]
            if needs_oracle and not oracle_available():
                results.append(Result(name, SKIP, "", "", ORACLE_HINT))
                continue
            print(f"  running {' '.join(command[1:])} ...", flush=True)
            start = time.perf_counter()
            proc = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            seconds = time.perf_counter() - start
            if proc.returncode != 0:
                results.append(
                    Result(name, FAIL, "command failed", "", proc.stderr[-2000:], seconds)
                )
                continue
            path = f"{RESULTS}/tables/{summary}"
            committed = json.loads(_git("show", f"HEAD:{path}"))
            fresh = json.loads((ROOT / path).read_text())
            if experiment == "validation" and not oracle_available():
                fresh["trees_vs_quantlib"] = committed["trees_vs_quantlib"]  # skipped, see above
            differences = compare_summaries(experiment, committed, fresh)
            measured = (
                "matches the committed results"
                if not differences
                else f"{len(differences)} values differ"
            )
            results.append(
                Result(
                    name,
                    PASS if not differences else FAIL,
                    measured,
                    f"relative {RELATIVE_TOLERANCE:.0e}; timings not compared",
                    "\n".join(differences[:10]),
                    seconds,
                )
            )
    finally:
        _git("checkout", "--", RESULTS)  # put the committed results back
    return results


def render(results: list[Result], markdown: bool) -> str:
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    quantlib = "installed" if oracle_available() else "not installed"
    header = (
        f"Commit {commit} · Python {platform.python_version()} · {platform.platform()} · "
        f"QuantLib {quantlib}"
    )
    counts = {s: sum(r.status == s for r in results) for s in (PASS, FAIL, SKIP)}
    summary = f"{counts[PASS]} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped"
    if markdown:
        lines = [
            "# Verification report",
            "",
            header,
            "",
            "| Result | Check | Measured | Limit | Time |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {r.status} | {r.name} | {r.measured} | {r.limit} | {r.seconds:.1f} s |"
            for r in results
        ]
        details = [f"- **{r.name}**: {r.detail}" for r in results if r.detail]
        return "\n".join([*lines, "", *(["Notes:", *details, ""] if details else []), summary, ""])
    lines = ["Options pricing engine: verification report", header, ""]
    for r in results:
        limit = f"   (limit: {r.limit})" if r.limit else ""
        lines.append(f"[{r.status}] {r.name}")
        if r.measured:
            lines.append(f"       {r.measured}{limit}   [{r.seconds:.1f} s]")
        if r.detail:
            lines += [f"       {d}" for d in r.detail.splitlines()]
    lines += ["", f"Summary: {summary}"]
    if counts[FAIL]:
        lines.append("Something failed: the lines marked [FAIL] say what and by how much.")
    elif counts[SKIP]:
        lines.append("Everything that ran passed; [SKIP] lines say what was not run and why.")
    else:
        lines.append("Everything passed.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="also rerun every experiment")
    parser.add_argument("--report", type=Path, help="also save the report as Markdown here")
    args = parser.parse_args(argv)

    print("Checking... (the test suite takes about 10 seconds)", flush=True)
    results = run_quick()
    if args.full:
        print("Rerunning every experiment...", flush=True)
        results += rerun_experiments()
    print(render(results, markdown=False))
    if args.report:
        args.report.write_text(render(results, markdown=True))
        print(f"Report saved to {args.report}")
    return 1 if any(r.status == FAIL for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
