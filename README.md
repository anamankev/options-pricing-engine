# Options Pricing Engine

A small, transparent derivatives-pricing library in Python. The goal is that every number it produces can be defended: model assumptions stated, numerical error quantified, results checked against published values and independent implementations, and experiments reproducible from one command.

Planned components: Black-Scholes-Merton closed form and Greeks; binomial trees (Cox-Ross-Rubinstein, Leisen-Reimer, BBS and BBSR) with American exercise; Monte Carlo with antithetic and control variates; safeguarded implied-volatility solvers. They serve two pre-registered studies: implied-volatility solver accuracy (Study A) and American-option tree convergence (Study B).

The full technical standard is [`docs/SPEC.md`](docs/SPEC.md).

## Status

Phase 0 of 6 (environment and tooling). No pricing code yet. See SPEC §14 for the phase plan.

## Scope

In scope: European calls and puts (closed form, Monte Carlo); American calls and puts (binomial trees); analytic and finite-difference Greeks; implied volatility for European options; Studies A and B; benchmarks.

Out of scope: market data, volatility-surface calibration, stochastic or local volatility, jumps, discrete cash dividends, transaction costs, funding and collateral, trading connectivity.

## Setup from a fresh clone

Requires Python 3.11 or newer and git. Everything used is free and open source.

macOS or Linux:

```bash
git clone https://github.com/anamankev/options-pricing-engine.git
cd options-pricing-engine
python3 -m venv .venv            # or python3.12 / python3.11 to pick a version
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"          # add the oracle extra for oracle tests: ".[dev,oracle]"
```

Windows (untested; CI runs on Linux):

```powershell
git clone https://github.com/anamankev/options-pricing-engine.git
cd options-pricing-engine
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

The `oracle` extra installs QuantLib, py_lets_be_rational and vollib, used only to cross-check results.

Dependency versions differ by Python version: NumPy 2.5 and SciPy 1.18 need Python 3.12+, so on 3.11 pip installs older releases. CI tests both. Every experiment records the exact versions it ran with.

## Commands

Run from the project folder with the environment activated.

| Task | Command |
|---|---|
| Lint | `ruff check .` |
| Format check | `ruff format --check .` |
| Type check (strict) | `mypy src` |
| Fast tests | `pytest -q -m "not slow and not statistical and not oracle"` |
| Full tests | `pytest -q` (needs the `oracle` extra once oracle tests exist) |
| Experiments | `python -m experiments.<name> --config experiments/configs/<name>.toml` (from Phase 2) |

Test markers: `slow`, `statistical` (Monte Carlo checks with fixed seeds), `oracle` (comparisons with QuantLib or Let's Be Rational). Any warning fails a test. Hypothesis property tests use fixed examples when `HYPOTHESIS_PROFILE=ci` is set, as in CI.

CI (GitHub Actions) runs lint, format, types and fast tests on Python 3.11 and 3.12 for every push and pull request. The full suite runs on demand: Actions > ci > Run workflow.

## Conventions

Full table: SPEC §2. In short: volatility `sigma` is an annualized decimal (0.20 means 20%); time `T` is in years; rates `r` and `q` are continuously compounded annualized decimals. Greeks are per 1.00 change in the input. Theta is `-dV/dT` per year. Helpers convert to per day (/365) and per volatility point (/100).

## Layout

```
src/pricing/      library code (pricing modules arrive from Phase 1)
tests/            pytest suite; conftest.py holds the Hypothesis profiles
docs/SPEC.md      technical standard
docs/notes/       research notes (template: TEMPLATE.md)
.github/          CI workflow
```

## How it was built


## Results

None yet. Study results will appear here with the commit, configuration and environment that produced them.

## What this project does not claim

- Constant-volatility Black-Scholes-Merton is a baseline model, not a description of listed-options markets.
- Synthetic implied-volatility round trips validate solvers; they say nothing about market calibration.
- Monte Carlo intervals quantify sampling error under the model, not model risk.
- One machine's benchmarks are not evidence of production performance.
- Nothing here is a trading strategy, alpha, or investment recommendation.

## License

MIT. See [LICENSE](LICENSE).
