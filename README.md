# Options Pricing Engine

A small, transparent derivatives-pricing library in Python. The goal is that every number it produces can be defended: model assumptions stated, numerical error quantified, results checked against published values and independent implementations, and every experiment reproducible from one command.

Components:
- Black-Scholes-Merton prices and analytic Greeks;
- binomial trees (Cox-Ross-Rubinstein, Leisen-Reimer, BBS and BBSR) with American exercise;
- Monte Carlo with antithetic and control variates;
- implied-volatility solvers (Brent, naive and safeguarded Newton) with full diagnostics.

These serve two pre-registered numerical studies: implied-volatility solver accuracy (Study A) and American-option tree convergence (Study B).

The technical standard the code is built to is [`docs/SPEC.md`](docs/SPEC.md). Each phase's record is in [`docs/notes/`](docs/notes/).

## Status

All seven phases of SPEC §14 are complete.
- 458 tests: deterministic, statistical, and oracle tests against QuantLib, Let's Be Rational and vollib.
- CI on Python 3.11 and 3.12.
- 9 experiments, each reproducible from one command (see [Reproducing the results](#reproducing-the-results)).
- Every number in this README is checked against the result file it comes from (`docs/claims.toml`), so a stale or mistyped number fails the tests.

## Results

### Study A: implied-volatility solvers ([note](docs/notes/study_a.md))
5880 problems across log-moneyness and total volatility, with targets computed at 50 digits:
- **Let's Be Rational** (Jaeckel 2015) is as accurate as the problem's conditioning allows wherever the price carries information about volatility: median error 1 ulp, 99th-percentile normalized error 1.99, in 2 iterations.
- **Our safeguarded Newton** converges everywhere Brent does, in fewer iterations (median 9 against 12). Near the money at small volatility, cancellation in the price formula costs it accuracy (median normalized error 683).
- **SPEC §10's well-posedness rule** mislabels problems in both directions. A conditioning-based rule classifies them correctly. It was applied post hoc and is labelled as such.

### Study B: binomial-tree convergence ([note](docs/notes/study_b.md))
R1 (European) and the 18 Ju and Zhong (1999) American cases, against QuantLib's QdFp engine at a validated strict setting:
- CRR is first order and oscillates between odd and even N; Richardson extrapolation on it helps only 32% of the time.
- Leisen-Reimer is second order for European options (slope -1.995) but first order for American puts (median -1.00).
- BBSR has the smallest error at a given runtime in 85% of comparisons on American puts (median order -1.24), although its error changes sign as N doubles. This ranking uses measured runtimes, so it depends on the machine and its load.

### Validation
- **Exact values:** prices agree with a 50-digit mpmath evaluation on 288 cases. The worst error is 0.76 eps times the formula's error scale, against a budget of 64.
- **Greeks:** all six agree with finite differences to 5e-8 relative or better.
- **QuantLib 1.44, 1800 European cases:** median relative agreement 2e-16 to 1e-15. In far tails (|d1| >= 3.2) QuantLib loses relative precision, up to 100% against the exact value, while this library's worst relative error is 2.5e-10 ([note](docs/notes/phase_1_quantlib_reconciliation.md)).
- **American benchmarks:** all 18 Ju-Zhong values are reproduced within 2e-3 by Leisen-Reimer and by BBSR (worst difference 1.1e-3).
  - Leisen-Reimer matches QuantLib's tree to 8.4e-12.
  - CRR differs from QuantLib's CRR only because QuantLib uses a different up-probability; with QuantLib's probability substituted, ours matches to 8.1e-12 ([note](docs/notes/phase_2_numerical_methods.md)).
- **Implied volatility** (SPEC §6.4 grid, 520 cases): safeguarded Newton and Brent recover every well-posed case, with errors at most 1.1 (safeguarded Newton) and 1.0 (Brent) times eps x (price rounding / vega). No solver ever returns an unconverged value (0 of 520 cases for either solver) ([note](docs/notes/phase_3_iv_and_fd.md)).
- **Finite-difference Greeks:** the error traces the predicted V in bump size. The best bump is within a factor of 2.1 of the refined prediction ([note](docs/notes/phase_3_bump_study.md)).
- **Monte Carlo intervals** (2000 replications per cell): 95% coverage holds for n >= 1000, and drops to 0.83 at n = 100 deep out of the money. Control variates cut the cost of a given accuracy 44-fold in the money but only 1.2-fold deep out of the money ([note](docs/notes/phase_2_mc_coverage.md)).

### Benchmarks ([note](docs/notes/benchmarks.md))
Pure Python and NumPy, single-threaded, on an Apple M5 (one machine; see the limitations). Median, with the interquartile range in the note:

| What | Size | Time |
|---|---|---|
| BSM price, scalar call | 1 option | 23 us |
| BSM price, vectorized | 1,000,000 options | 21 ms (4.8e7 options per second) |
| American put, CRR or Leisen-Reimer | N ~ 3200 | 30 to 31 ms |
| American put, BBSR | N = 3200 | 41 ms |
| Monte Carlo, plain / antithetic / control variate | per path, n = 1e6 | 9.1 / 7.4 / 10.6 ns |

Tree runtime grows like N^1.1 to N^1.4 up to N = 3200: per-layer overhead still dominates the O(N^2) work there. The registered N^2 hypothesis was refuted and the reason is diagnosed in the note.

Figures for every study are in [`experiments/results/figures/`](experiments/results/figures/), including 3D error surfaces paired with their heatmaps.

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
implied_volatility(OptionType.CALL, price, **market)  # 0.19999999999999996, i.e. 0.2 to 2 ulps
implied_volatility_result(OptionType.CALL, price, **market).iterations  # full diagnostics
```

How the API behaves:
- Market inputs are keyword-only, so r and q cannot be swapped by position.
- Arrays broadcast like NumPy and return arrays; scalars return a float.
- Invalid inputs raise `PricingInputError` naming the parameter.
- Greeks that do not exist (for example delta at expiry with S = K) raise `UndefinedGreekError`.
- Solvers never return an unconverged value: `implied_volatility` raises `ConvergenceError`, and `implied_volatility_result` reports why the solver stopped.

## Scope

In scope:
- European calls and puts (closed form, Monte Carlo);
- American calls and puts (binomial trees);
- analytic and finite-difference Greeks;
- implied volatility for European options;
- Studies A and B, and benchmarks.

Out of scope: market data, volatility-surface calibration, stochastic or local volatility, jumps, discrete cash dividends, transaction costs, funding and collateral, and trading connectivity.

## Conventions

The full table is SPEC §2. In short:
- volatility `sigma` is an annualized decimal (0.20 means 20%);
- time `T` is in years;
- rates `r` and `q` are continuously compounded annualized decimals;
- Greeks are per 1.00 change in the input, and theta is `-dV/dT` per year;
- helpers convert to per day (/365) and per volatility point (/100).

## Setup from a fresh clone

Requires Python 3.11 or newer and git. Everything used is free and open source.

macOS or Linux:

```bash
git clone https://github.com/anamankev/options-pricing-engine.git
cd options-pricing-engine
python3.12 -m venv .venv         # CI tests 3.11 and 3.12; plain python3 may be newer
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,oracle]"   # drop ",oracle" to skip QuantLib, lets-be-rational and vollib
```

Windows (untested; CI runs on Linux):

```powershell
git clone https://github.com/anamankev/options-pricing-engine.git
cd options-pricing-engine
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev,oracle]"
```

The `oracle` extra installs QuantLib, lets-be-rational and vollib, which are used only to cross-check results. NumPy 2.5 and SciPy 1.18 need Python 3.12 or newer, so on 3.11 pip installs older releases; CI tests both, and every experiment records the exact versions it ran with.

## Commands

Run from the project folder with the environment activated.

| Task | Command |
|---|---|
| Lint | `ruff check .` |
| Format check | `ruff format --check .` |
| Type check (strict) | `mypy src` |
| Fast tests | `pytest -q -m "not slow and not statistical and not oracle"` |
| Full tests (needs the `oracle` extra) | `pytest -q` |

Test markers:
- `slow`;
- `statistical`: Monte Carlo checks with fixed seeds;
- `oracle`: comparisons with QuantLib, Let's Be Rational or vollib.

Any warning fails a test. Hypothesis property tests use fixed examples when `HYPOTHESIS_PROFILE=ci` is set, as in CI. CI (GitHub Actions) runs lint, format, types and fast tests on Python 3.11 and 3.12 for every push and pull request; the full suite runs on demand under Actions > ci > Run workflow.

## Reproducing the results

Each experiment is one command that writes its tables and figures to `experiments/results/`, recording the commit, library versions and platform. Each study's hypotheses were committed before it first ran; the validation summary only records measurements.

| Result | Command | Needs `oracle` |
|---|---|---|
| QuantLib reconciliation | `python -m experiments.quantlib_reconciliation --config experiments/configs/quantlib_reconciliation.toml` | yes |
| Monte Carlo coverage | `python -m experiments.mc_coverage --config experiments/configs/mc_coverage.toml` | no |
| Finite-difference bump study | `python -m experiments.greek_bumps --config experiments/configs/greek_bumps.toml` | no |
| Study A | `python -m experiments.study_a_iv --config experiments/configs/study_a_iv.toml` | yes |
| Study B | `python -m experiments.study_b_trees --config experiments/configs/study_b_trees.toml` | yes |
| Study A2 (the solver fixes, on Study A's grid and a new one) | `python -m experiments.study_a2_iv --config experiments/configs/study_a2_iv.toml` | yes |
| Study A2 follow-up (time-value cutoff 0, confirmed on a third grid) | `python -m experiments.study_a2_iv --config experiments/configs/study_a2_followup.toml` | yes |
| Benchmarks | `python -m experiments.benchmarks --config experiments/configs/benchmarks.toml` | no |
| Validation numbers (this README's Validation section) | `python -m experiments.validation --config experiments/configs/validation.toml` | for the tree comparison |

## Limitations

Model:
- Constant volatility, rates and dividend yield. There are no market data, smiles or discrete dividends (see Scope).

Numerical (each documented in its note):
- Our implied-volatility solvers lose accuracy near the money at very small total volatility, where the Black formula subtracts two nearly equal terms.
- On 30 informative in-the-money prices they return zero total volatility, because the time value is below their lower-bound tolerance; Let's Be Rational recovers 16 of these. Use Let's Be Rational where those regimes matter.
- Leisen-Reimer is only first order for American options.
- BBSR's error changes sign as N doubles, so a single BBSR price can be luckier or unluckier than its convergence order suggests.

Performance:
- Pure Python and NumPy, single-threaded; there are no compiled kernels.
- Benchmarks come from one machine (an Apple M5).

Validation:
- Most experiments ran on macOS arm64. CI repeats the tests on Linux.
- Some citations' page or equation numbers are not yet checked against the books (SPEC §17).
- Three reference values (R2, R3, R4b) are reproduced numerically but not yet checked against the printed primary source, and R1's Hull edition and page are unrecorded (SPEC §9.1).
- Analyses added after seeing data are labelled post hoc in their notes and are exploratory, for example:
  - Study A's informative region (the source of its headline accuracy numbers);
  - the bump study's refined optimum;
  - the benchmark scaling model.

References:
- QuantLib loses relative precision in far tails.
- QuantLib's default "high precision" American scheme is accurate only to about 6e-8; Study B uses a stricter, validated setting.

## What this project does not claim

- Constant-volatility Black-Scholes-Merton is a baseline model, not a description of listed-options markets.
- Synthetic implied-volatility round trips validate solvers; they say nothing about market calibration.
- Monte Carlo intervals quantify sampling error under the model, not model risk.
- One machine's benchmarks are not evidence of production performance.
- Nothing here is a trading strategy, alpha, or investment recommendation.

## Layout

```
src/pricing/      library: contracts, errors, black_scholes, binomial, monte_carlo, implied_vol,
                  finite_diff, oracles
tests/            pytest suite; conftest.py holds the Hypothesis profiles
experiments/      runnable studies, configs/, results/{tables,figures}
docs/SPEC.md      technical standard
docs/notes/       research notes (template: TEMPLATE.md)
.github/          CI workflow
```

## How it was built

The project was built in the seven phases of SPEC §14, under four working rules:
- plan before each phase;
- one commit per unit;
- register hypotheses before running experiments;
- never loosen a tolerance without a diagnosis.

## License

MIT. See [LICENSE](LICENSE).
