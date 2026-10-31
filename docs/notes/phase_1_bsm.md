# Phase 1: Black-Scholes-Merton reference engine

Status: COMPLETE (tag `phase-1`). This is a phase-gate record, not a study. The hypotheses restate the SPEC 14 exit gate and the SPEC 9.2 test categories as checkable claims; they were not registered before the work. The QuantLib reconciliation, which was registered, has its own note: `phase_1_quantlib_reconciliation.md`.

## 1. Question
Does the library price European options and their Greeks under BSM correctly, to the precision double-precision arithmetic allows? Do invalid states fail loudly instead of returning plausible wrong numbers?

## 2. Hypotheses
- H1: Prices and Greeks match independent references: published values R1 to R3 (SPEC 9.1), a 50-digit mpmath evaluation, finite differences, and QuantLib.
- H2: The no-arbitrage identities, bounds, monotonicity and Greek signs of SPEC 9.2 hold across the valid domain, within the rounding budget.
- H3: Boundaries follow SPEC 3.4: T = 0, sigma = 0, small total volatility, extreme moneyness. Invalid inputs raise the SPEC 2.4 exceptions, naming the parameter.

## 3. Design
- Code: `src/pricing/{errors,contracts,finite_diff,black_scholes,oracles}.py`.
- Tests: 185, all deterministic except the Hypothesis properties, which are derandomized in CI:

  | Category | Tests | Where |
  |---|---|---|
  | Input validation, contracts, errors | 50 | `test_contracts.py`, `test_errors.py` |
  | Finite-difference tool | 12 | `test_finite_diff.py` |
  | Prices: types, forms, mpmath, boundaries | 29 | `test_black_scholes.py` |
  | Greeks: FD, identities, anchors, boundaries | 54 | `test_black_scholes_greeks.py` |
  | Reference values R1 to R3 | 4 | `test_reference_values.py` |
  | Properties | 9 | `test_properties.py` |
  | Oracle (QuantLib) and metric definitions | 10 | `test_oracles.py`, `test_reconciliation_metrics.py` |
  | Phase 0 package and local tooling | 17 | `test_package.py` (2); 15 tooling tests kept outside the repository |

- Rounding model behind every price tolerance: absolute error <= 64 eps x scale, with scale = D F N(theta d1) + D K N(theta d2) + D F phi(d1) (1 + |d1| + |d2|).

## 4. Results
- Commits: `e8c7dff` (errors, validation) to `47716e5` (named tolerances). CI at `12776f5`: fast checks on Python 3.11 and 3.12, and the full suite with QuantLib on Linux, all passed. On 3.11 pip resolved NumPy 2.4.6 and scipy-stubs 1.17.1.5; on 3.12, NumPy 2.5.4 and 1.18.1.1.
- Prices against mpmath at 50 digits (144 markets x call and put): worst error 0.76 eps x scale, against a budget of 64.
- Greeks against finite differences (72 cases x 2 types): worst relative error 2e-9 (theta) to 5e-8 (vega), against a tolerance of 1e-6.
- Properties: 8 Hypothesis properties held on 20,000 random markets each (160,000 in total) in a one-off stress run, with warnings as errors; CI runs 100 per property.
- QuantLib: agreement to machine precision except in far tails, where QuantLib is the inaccurate one (see the reconciliation note). The R1 call is bit-identical.
- R1 to R3: printed values reproduced; R1's unrounded 4.759422 and 0.808599 confirmed by QuantLib.

Two diagnoses, made when a first version of a test failed. Neither changed library code.
1. The price rounding budget first counted only the two terms the formula subtracts. At d1 = -11.4 the test's independent spot-form price was off by 63 eps x terms, while ours was off by 1.8. In a far tail, a rounding error of order eps |d| in d changes N(d) by a relative amount of about |d| times that, since phi(d) is about |d| N(d) there (the Mills ratio). The budget gained the D F phi(d1) (1 + |d1| + |d2|) term derived from this; it was not loosened to fit.
2. Gamma was first checked as a second difference of prices. Its error is h^2 gamma''/12 + 4 eps V / h^2. At S = 120, T = 0.25, where V = 21 and gamma = 0.0014, the best attainable is about 1.0e-6 relative. A sweep over eta traced the SPEC 7 V-shape: 9.7e-5 at eta = 1e-3, 1.0e-6 at 1e-4, 8.7e-6 at 1e-5. Gamma is now checked as the first difference of analytic delta, which is itself checked against prices; worst error 2e-8.

## 5. Verdicts
- H1: SUPPORTED, with the reconciliation note's qualification: in far tails QuantLib, not ours, is inaccurate.
- H2: SUPPORTED (720-case parity grid; 9 properties; 160,000-example stress run).
- H3: SUPPORTED (boundary, continuity, underflow, and validation tests in `test_black_scholes*.py` and `test_contracts.py`).

## 6. Interpretation
- The forward form is not a cosmetic choice. With N(theta d) evaluated directly, tail prices and Greeks keep their relative precision to 4e-13. A widely used library loses it entirely in the same places.
- Accuracy statements need a model of the error. "64 eps" means nothing until you say 64 eps of what. Both diagnoses were cases where the first model was incomplete, and the fix was to complete the model, not to widen the number.
- Where conditioning is poor, a finite-difference check of a Greek can be less accurate than the Greek it checks. Choosing the better-conditioned check is part of the method.

## 7. Limitations and threats to validity
- Single option type per call: a mixed book needs two calls.
- Inputs that overflow intermediate values (e.g. (r - q) T above ~700) produce NumPy overflow warnings, not a named exception.
- Some citations' numbering is not yet checked against the books (SPEC 17).
- Hypothesis domain: |ln(S/K)| <= 2, T <= 10 years, sigma <= 2; behaviour beyond is covered only by the boundary tests.

## 8. Next questions
- Phase 2: trees and Monte Carlo will be checked against this engine. Its verified accuracy sets the floor for their convergence studies.
- Study A: tail prices must come from mpmath (SPEC 8.3); Phase 1 showed why.
