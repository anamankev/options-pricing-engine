# Options Pricing Engine

A small, transparent derivatives-pricing library in Python. The goal is that every number it produces can be defended: model assumptions stated, numerical error quantified, results checked against published values and independent implementations, and experiments reproducible from one command.

Planned components: Black-Scholes-Merton closed form and Greeks; binomial trees (Cox-Ross-Rubinstein, Leisen-Reimer, BBS and BBSR) with American exercise; Monte Carlo with antithetic and control variates; safeguarded implied-volatility solvers. They serve two pre-registered studies: implied-volatility solver accuracy (Study A) and American-option tree convergence (Study B).

The full technical standard is [`docs/SPEC.md`](docs/SPEC.md).

## Status

Phases 0 and 1 are complete: tooling, and the Black-Scholes-Merton reference engine (European prices and Greeks). Phase 2 (binomial trees with American exercise, Monte Carlo) is next. See SPEC §14 for the phase plan and `docs/notes/` for each phase's record.

## Usage

```python
from pricing import black_scholes as bs
from pricing.contracts import OptionType

# Hull, Options, Futures, and Other Derivatives, Example 15.6
bs.price(OptionType.CALL, S=42.0, K=40.0, T=0.5, r=0.10, q=0.0, sigma=0.20)  # 4.759422392871536
bs.vega(OptionType.CALL, S=42.0, K=40.0, T=0.5, r=0.10, q=0.0, sigma=0.20)  # 8.81 per 1.00 of sigma
```

Market inputs are keyword-only, so r and q cannot be swapped by position. Arrays broadcast like NumPy and return arrays; scalars return a float. Invalid inputs raise `PricingInputError` naming the parameter, and Greeks that do not exist (for example delta at expiry with S = K) raise `UndefinedGreekError`.

## Scope

In scope: European calls and puts (closed form, Monte Carlo); American calls and puts (binomial trees); analytic and finite-difference Greeks; implied volatility for European options; Studies A and B; benchmarks.

Out of scope: market data, volatility-surface calibration, stochastic or local volatility, jumps, discrete cash dividends, transaction costs, funding and collateral, trading connectivity.

## Setup from a fresh clone

Requires Python 3.11 or newer and git. Everything used is free and open source.

macOS or Linux:

```bash
git clone https://github.com/anamankev/options-pricing-engine.git
cd options-pricing-engine
python3.12 -m venv .venv         # CI tests 3.11 and 3.12; plain python3 may be newer
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"          # add the oracle extra for oracle tests: ".[dev,oracle]"
```

Windows (untested; CI runs on Linux):

```powershell
git clone https://github.com/anamankev/options-pricing-engine.git
cd options-pricing-engine
py -3.12 -m venv .venv
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
src/pricing/      library: contracts, errors, black_scholes, finite_diff, oracles
tests/            pytest suite; conftest.py holds the Hypothesis profiles
docs/SPEC.md      technical standard
docs/notes/       research notes (template: TEMPLATE.md)
experiments/      runnable studies, configs/, results/
.github/          CI workflow
```

## How it was built


## Results

Validation of the Phase 1 engine (studies A and B have not run yet):

- **Against exact values:** prices agree with a 50-digit mpmath evaluation on 288 cases. The worst error is 0.76 times eps times the formula's error scale, inside a budget of 64 (`tests/test_black_scholes.py`).
- **Greeks:** all six agree with finite differences to 5e-8 relative or better (`tests/test_black_scholes_greeks.py`).
- **Against QuantLib 1.44, 1800 cases:** median relative agreement is 2e-16 to 1e-15. In far tails (|d1| >= 3.2) QuantLib loses relative precision, up to 100% against the exact value, while this library stays within 2.5e-10. Details: [`docs/notes/phase_1_quantlib_reconciliation.md`](docs/notes/phase_1_quantlib_reconciliation.md).

## What this project does not claim

- Constant-volatility Black-Scholes-Merton is a baseline model, not a description of listed-options markets.
- Synthetic implied-volatility round trips validate solvers; they say nothing about market calibration.
- Monte Carlo intervals quantify sampling error under the model, not model risk.
- One machine's benchmarks are not evidence of production performance.
- Nothing here is a trading strategy, alpha, or investment recommendation.

## License

MIT. See [LICENSE](LICENSE).
