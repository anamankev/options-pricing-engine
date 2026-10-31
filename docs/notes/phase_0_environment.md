# Phase 0: environment and tooling

Status: COMPLETE (tag `phase-0`). This is a phase-gate record, not a study: the hypotheses below restate the SPEC §14 exit gate as checkable claims and were not registered before the checks ran.

## 1. Question
Can someone with a fresh clone, following only the README, reproduce every check (lint, format, types, tests) on the supported Python versions?

## 2. Hypotheses
- H1: A fresh clone passes `ruff check .`, `ruff format --check .`, `mypy src` and `pytest -q` on Python 3.11 and 3.12, using only the README commands.
- H2: No tracked file contains a machine-specific path.

## 3. Design
- Fresh clones into a temporary directory: Python 3.11.17 with `.[dev,oracle]` and `HYPOTHESIS_PROFILE=ci`; plain `python3` (3.14.8 on the development Mac) with `.[dev]`. GitHub Actions `ubuntu-latest`: 3.11 and 3.12 with `.[dev]`; manual full suite on 3.12 with `.[dev,oracle]`.
- A local post-edit check (lint and fast tests after every edit) was also set up. It is personal editor tooling and is kept outside the repository.
- Paths: `git grep -nIE '/Users/|kelvinanaman|/home/|C:\\\\'` over tracked files.

## 4. Results
- Commits: `b1c840b` to `cb7f7c0` (5 units). CI: push run 41070958691 (checks on 3.11 and 3.12: success); manual run 41071119106 (checks and full: success).
- Environment, development Mac (macOS, arm64): Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1, matplotlib 3.11.2, mpmath 1.4.1, pytest 9.2.0, Hypothesis 6.170.13, Ruff 0.17.3, mypy 2.4.1, QuantLib 1.44, py_lets_be_rational 1.1.2, vollib 1.0.11.
- On Python 3.11, pip resolves NumPy 2.4.6 and SciPy 1.17.1, because NumPy 2.5 and SciPy 1.18 require Python 3.12 or later.
- H1: all four checks passed in every environment listed in §3; 17 tests, of which 15 covered the local editor tooling and are kept outside the repository with it.
- H2: no matches.

## 5. Verdicts
- H1: SUPPORTED (3.11 and 3.12 on Linux CI, 3.11 and 3.14 on macOS).
- H2: SUPPORTED.

## 6. Interpretation
Treating warnings as errors paid off before any pricing code existed: it would have failed every test that imported `py_vollib`, which now emits a `DeprecationWarning` on import. A silently deprecated dependency is the kind of drift that breaks a study months later. Using fixed Hypothesis examples in CI (`derandomize=True`) means a CI failure reproduces locally, which matters once property tests guard parity and bounds.

## 7. Limitations and threats to validity
- Dependency ranges, not a lock file, define the environment, so later installs may resolve newer versions. Each experiment must record exact versions.

## 8. Next questions
- Phase 1: add `scipy-stubs` when `src/` first imports SciPy, so `mypy --strict` checks those calls.
- Phase 3: choose between `py_lets_be_rational` and `lets_be_rational` as the Study A oracle (SPEC §17).
