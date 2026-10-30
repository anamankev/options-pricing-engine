# Options Pricing Engine

A small, transparent derivatives-pricing library in Python. The goal is that every number it produces can be defended: model assumptions stated, numerical error quantified, results checked against published values and independent implementations, and experiments reproducible from one command.

Planned components: Black-Scholes-Merton closed form and Greeks; binomial trees (Cox-Ross-Rubinstein, Leisen-Reimer, BBS and BBSR) with American exercise; Monte Carlo with antithetic and control variates; safeguarded implied-volatility solvers. They serve two pre-registered studies: implied-volatility solver accuracy (Study A) and American-option tree convergence (Study B).

The full technical standard is [`docs/SPEC.md`](docs/SPEC.md).

## Status

Phases 0 to 5 are complete:

- tooling;
- the Black-Scholes-Merton reference engine (European prices and Greeks);
- binomial trees (CRR, Leisen-Reimer, BBS, BBSR) with American exercise, and Monte Carlo with antithetic and control variates;
- implied-volatility solvers (Brent, naive and safeguarded Newton) with Let's Be Rational as the oracle, and the finite-difference bump study;
- Study A, the accuracy of those solvers across the (log-moneyness, total-volatility) plane;
- Study B, the convergence of the trees for European and American options.

Phase 6 (benchmarks and v1.0) is next. See SPEC §14 for the phase plan and `docs/notes/` for each phase's record.

## Usage

```python
from pricing import black_scholes as bs
from pricing.contracts import OptionType

# Hull, Options, Futures, and Other Derivatives, Example 15.6
bs.price(OptionType.CALL, S=42.0, K=40.0, T=0.5, r=0.10, q=0.0, sigma=0.20)  # 4.759422392871536
bs.vega(OptionType.CALL, S=42.0, K=40.0, T=0.5, r=0.10, q=0.0, sigma=0.20)  # 8.81 per 1.00 of sigma
```

```python
from pricing import binomial as bt
from pricing.contracts import ExerciseStyle, OptionType
from pricing.monte_carlo import MCMethod, monte_carlo

# Ju and Zhong (1999), Exhibit 3: American put, published 1.990
bt.bbsr(
    OptionType.PUT,
    S=40.0,
    K=40.0,
    T=0.5833,
    r=0.0488,
    q=0.0,
    sigma=0.2,
    n_steps=2048,
    exercise=ExerciseStyle.AMERICAN,
).value  # 1.99047...

res = monte_carlo(
    OptionType.CALL,
    S=100.0,
    K=100.0,
    T=1.0,
    r=0.05,
    q=0.02,
    sigma=0.2,
    n_paths=100_000,
    seed=1,
    method=MCMethod.CONTROL_VARIATE,
    pilot_paths=10_000,
)
res.estimate, res.std_error, (res.ci_low, res.ci_high)
```

```python
from pricing import black_scholes as bs
from pricing.contracts import OptionType
from pricing.implied_vol import implied_volatility, implied_volatility_result

market = dict(S=42.0, K=40.0, T=0.5, r=0.10, q=0.0)
price = bs.price(OptionType.CALL, **market, sigma=0.20)
implied_volatility(OptionType.CALL, price, **market)  # 0.20000000000000015, i.e. 0.2 to 1 ulp
implied_volatility_result(OptionType.CALL, price, **market).iterations  # full diagnostics
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
src/pricing/      library: contracts, errors, black_scholes, binomial, monte_carlo, implied_vol,
                  finite_diff, oracles
tests/            pytest suite; conftest.py holds the Hypothesis profiles
docs/SPEC.md      technical standard
docs/notes/       research notes (template: TEMPLATE.md)
experiments/      runnable studies, configs/, results/
.github/          CI workflow
```

## How it was built


## Results

Validation so far (studies A and B have not run yet):

- **Against exact values:** prices agree with a 50-digit mpmath evaluation on 288 cases. The worst error is 0.76 times eps times the formula's error scale, inside a budget of 64 (`tests/test_black_scholes.py`).
- **Greeks:** all six agree with finite differences to 5e-8 relative or better (`tests/test_black_scholes_greeks.py`).
- **Against QuantLib 1.44, 1800 cases:** median relative agreement is 2e-16 to 1e-15. In far tails (|d1| >= 3.2) QuantLib loses relative precision, up to 100% against the exact value, while this library stays within 2.5e-10. Details: [`docs/notes/phase_1_quantlib_reconciliation.md`](docs/notes/phase_1_quantlib_reconciliation.md).
- **American options:** all 18 Ju and Zhong (1999) benchmark values are reproduced within 2e-3 by two methods (Leisen-Reimer, N = 2001; BBSR, N = 2048). QuantLib's high-precision American engine agrees with BBSR at N = 16384 within 1.5e-6.
- **Trees vs QuantLib:** Leisen-Reimer matches QuantLib's to 8.4e-12 relative. CRR differs by the O(1/N) error because QuantLib uses a different up-probability convention; with that convention substituted, ours matches to 8.1e-12. Details: [`docs/notes/phase_2_numerical_methods.md`](docs/notes/phase_2_numerical_methods.md).
- **Implied volatility** (SPEC §6.4 grid, 520 cases): the safeguarded Newton and Brent solvers recover every well-posed case to within a few eps x the price's rounding error divided by vega. Naive Newton fails on 202 of them and reports each failure; no solver ever returns an unconverged value. Details: [`docs/notes/phase_3_iv_and_fd.md`](docs/notes/phase_3_iv_and_fd.md).
- **Finite-difference Greeks:** the error traces the predicted V in bump size, with truncation slope +2.00 for every Greek. The best bump lies within a factor of 2.1 of the refined prediction, and within a factor of 4 of the textbook rule eps^(1/3). Figures, including 3D error surfaces: [`experiments/results/figures/`](experiments/results/figures/); details: [`docs/notes/phase_3_bump_study.md`](docs/notes/phase_3_bump_study.md).
- **Study A, implied-volatility solvers** (5880 problems, 50-digit targets):
  - Let's Be Rational is as accurate as conditioning allows wherever the price carries information: median 1 ulp, 99th-percentile normalized error 1.99, in 2 iterations.
  - Our safeguarded Newton converges everywhere Brent does, with fewer iterations (median 9 vs 12). Near the money at small volatility, cancellation in the price formula costs it accuracy (median normalized error 683).
  - The SPEC's well-posedness rule mislabels points in both directions. A conditioning-based rule fixes it; it was applied post hoc and is labelled as such.
  - Details and figures: [`docs/notes/study_a.md`](docs/notes/study_a.md).
- **Study B, tree convergence** (R1 European and the 18 Ju-Zhong American cases, against QuantLib QdFp at a validated strict setting):
  - CRR is first order and oscillates between odd and even N; Richardson on it helps only 32% of the time.
  - Leisen-Reimer is second order for European options (slope -1.995) but first order for American puts (median -1.00).
  - BBSR has the smallest error at a given runtime in 85% of comparisons (median order -1.24 on American puts), though its error changes sign as N doubles.
  - Details and figures: [`docs/notes/study_b.md`](docs/notes/study_b.md).
- **Monte Carlo intervals** (2000 replications per cell): 95% coverage holds for n >= 1000; at n = 100 deep out of the money it drops to 0.83. Control variates cut the cost of a given accuracy 44-fold in the money, but only 1.2-fold deep out of the money. Details: [`docs/notes/phase_2_mc_coverage.md`](docs/notes/phase_2_mc_coverage.md).

## What this project does not claim

- Constant-volatility Black-Scholes-Merton is a baseline model, not a description of listed-options markets.
- Synthetic implied-volatility round trips validate solvers; they say nothing about market calibration.
- Monte Carlo intervals quantify sampling error under the model, not model risk.
- One machine's benchmarks are not evidence of production performance.
- Nothing here is a trading strategy, alpha, or investment recommendation.

## License

MIT. See [LICENSE](LICENSE).
