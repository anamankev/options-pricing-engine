# Phase 1: QuantLib reconciliation of European prices and Greeks

Status: COMPLETE. Sections 1 to 3 registered at commit `c10ab40`, before the reconciliation code existed. Results from commit `df6c321`.

## 1. Question
Do our closed-form BSM prices and Greeks (`pricing.black_scholes`) agree with QuantLib's independent implementation to the precision two double-precision closed forms should reach? Where they differ, is the cause numerical conditioning or a formula, unit or convention error? The answer decides the oracle-test tolerance (SPEC 8.1) and the status of R1 to R3 (SPEC 9.1).

## 2. Hypotheses
- H1: For every case in the grid, the price relative error |ours - QuantLib| / |QuantLib| is at most 1e-11, the SPEC 8.1 starting tolerance.
- H2: QuantLib's `AnalyticEuropeanEngine` reports vega per 1.00 of volatility, theta per year (calendar decay, `-dV/dT`), and rho and dividend rho per 1.00, matching SPEC 2.2. Checked on R1: vega near 8.81 and theta near -4.559 confirm it. Values near 0.0881 or -0.0125 would mean per-point or per-day units. With units matched, each of delta, gamma, vega, theta, rho and phi agrees within relative 1e-11 in every case.
- H3: Any relative error above 1e-11 occurs only where the quantity is small compared with the terms that produce it: far tails, where rounding in d is amplified by about d^2, or results that underflow. A formula, unit or convention difference would instead show up at ordinary, near-the-money cases. One exceedance at a case with |d1| < 3 refutes H3.
- H4: With T represented exactly (QuantLib `Actual360`, 180 and 90 days give T = 0.5 and 0.25), QuantLib reproduces R1's unrounded values 4.759422 (call) and 0.808599 (put) to 6 decimals, R2's 2.4648 and R3's 2.1334 to 4 decimals, and our values to relative 1e-11.

## 3. Design
- Methods compared: `pricing.black_scholes` (price, delta, gamma, vega, theta, rho, phi) against QuantLib `AnalyticEuropeanEngine` (`NPV`, `delta`, `gamma`, `vega`, `theta`, `rho`, `dividendRho`).
- QuantLib settings (SPEC 8.1): evaluation date fixed at 30 October 2026; maturity = evaluation date + n days; `Actual365Fixed`, so our T = n / 365 exactly; `FlatForward` curves for r and q with continuous compounding; `BlackConstantVol`; `BlackScholesMertonProcess`.
- Grid, 1800 cases (SPEC 8.1 asks for at least 500): K = 100; S in {70, 90, 100, 110, 130}; n in {7, 30, 91, 365, 1825} days; r in {-0.01, 0, 0.05}; q in {-0.01, 0, 0.03}; sigma in {0.10, 0.25, 0.50, 1.00}; call and put.
- Metrics, for each case and quantity: absolute error |ours - QuantLib|; relative error |ours - QuantLib| / |QuantLib|, defined as 0 when both are exactly 0 and infinite when only QuantLib's is 0; |d1|, for H3. Summary per quantity: maximum, 99th percentile and median of the relative error, the count above 1e-11, and the worst case.
- Ground truth: neither implementation is ground truth. Agreement between two independent double-precision implementations bounds both errors. Absolute accuracy against exact values is tested separately against mpmath at 50 digits (`tests/test_black_scholes.py`).
- Seeds: none. The grid is deterministic.
- Config: `experiments/configs/quantlib_reconciliation.toml`. Output: `experiments/results/tables/quantlib_reconciliation.csv` (every case) and `..._summary.json` (summary plus commit hash, dirty flag, library versions and platform).
- Decision rule: if H1 and H2 hold, the oracle test asserts relative 1e-11. Otherwise the tolerance is set only after a recorded diagnosis, never loosened to fit.

## 4. Results
- Code: commit `df6c321`, clean tree. Command: `python -m experiments.quantlib_reconciliation --config experiments/configs/quantlib_reconciliation.toml`.
- Environment: macOS 26.6 arm64, Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1, QuantLib 1.44, mpmath 1.4.1.
- Outputs: `experiments/results/tables/quantlib_reconciliation.csv` (1800 rows) and `quantlib_reconciliation_summary.json`.
- History: the first run, at `cb19df4`, crashed while printing its summary (an infinite relative error could not be formatted) and recorded `dirty: true` because it checked git status after writing its own output. Both were fixed in `df6c321` before the run reported here; the case data were the same.

Relative error against QuantLib (1800 cases):

| Quantity | Max | 99th pct | Median | Cases > 1e-11 | Smallest abs(d1) among them |
|---|---|---|---|---|---|
| price | inf (18 cases where QuantLib returns exactly 0) | inf | 4.1e-16 | 116 | 3.2 |
| delta | 4.9e13 | 1.7e8 | 2.3e-16 | 73 | 4.9 |
| gamma | 1.2e-10 | 4.7e-11 | 3.8e-16 | 36 | 18.9 |
| vega | 1.2e-10 | 4.7e-11 | 3.2e-16 | 36 | 18.9 |
| theta | 3.7e-2 | 4.0e-6 | 9.7e-16 | 49 | 5.1 |
| rho | 1.7e13 | 1.7e8 | 2.7e-16 | 71 | 4.9 |
| phi | 1.7e13 | 1.7e8 | 2.3e-16 | 73 | 4.9 |

Post-hoc diagnosis (not pre-registered; added after the first run showed relative errors far beyond rounding). In every case above 1e-11, both values were scored against mpmath at 50 digits, and the absolute difference was expressed in units of eps times the quantity's natural scale (S for price and theta, 1 for delta, 1/S for gamma, S sqrt(T) for vega, K T for rho, S T for phi):

| Quantity | Ours, max rel. error vs exact | QuantLib, max rel. error vs exact | Max abs(ours - QuantLib) / (eps x scale) |
|---|---|---|---|
| price | 2.5e-10 | 1.0 | 0.65 |
| delta | 4.2e-13 | 1.0 | 0.37 |
| gamma | 3.7e-13 | 1.2e-10 | 0.00 |
| vega | 3.7e-13 | 1.2e-10 | 0.00 |
| theta | 3.7e-13 | 3.8e-2 | 0.02 |
| rho | 4.3e-13 | 1.0 | 0.37 |
| phi | 4.2e-13 | 1.0 | 0.37 |

Example: put, S = 130, K = 100, 30 days, sigma = 0.10, abs(d1) = 9.2. Exact delta -1.8859e-20; ours -1.8859e-20; QuantLib -3.85e-34.

## 5. Verdicts
- H1: REFUTED. 116 of 1800 prices differ by more than relative 1e-11; in 18, QuantLib returns exactly 0 for a positive price (e.g. 2.0e-81).
- H2: units SUPPORTED (R1: vega 8.813 per 1.00, theta -4.559 per year, matching SPEC 2.2). Agreement within relative 1e-11 REFUTED for all six Greeks (36 to 73 cases each).
- H3: SUPPORTED. No exceedance at abs(d1) < 3; the smallest is 3.2. Scored against exact values, every exceedance is QuantLib's error (up to 100% relative), while ours stays within 2.5e-10 (price) and 4.3e-13 (Greeks).
- H4: SUPPORTED. QuantLib gives 4.759422 and 0.808599 for R1 at 6 decimals, 2.4648 for R2 and 2.1334 for R3. Ours agrees within relative 1e-11, and the R1 call matches to the last bit.

## 6. Interpretation
Where both implementations are well conditioned they agree to machine precision (medians 2e-16 to 1e-15). The disagreements are all in far tails, and there QuantLib is the inaccurate one. Its tail values look like forms such as N(d) - 1, which cancel to zero or noise once N(d) rounds to 1; we inferred this from its outputs and did not read its source. Ours evaluates N(theta d) directly (SPEC 3.4), so it keeps relative precision.

In absolute terms the differences are negligible (at most 0.65 eps times scale), which is all a hedging desk would see. Relative precision in the tails matters for implied volatility: a deep out-of-the-money price returned as 0 instead of 2e-81 has no implied volatility at all. That is the regime Study A measures.

Decision (section 3 rule, after the diagnosis above): the oracle test passes a case if the relative error is at most 1e-11, or the absolute error is at most 4 eps x natural scale (observed maximum 0.65, so a 6x margin). Our tail accuracy is held to relative precision separately, by the mpmath tests.

## 7. Limitations and threats to validity
- One platform for this run. The oracle tests repeat it on Linux in CI's manual full suite.
- The grid stops at sigma = 1 and T = 5 years, and at abs(ln(S/K)) <= 0.36.
- The natural scales are a modelling choice; the absolute criterion depends on them.
- The mpmath scoring covers only cases above 1e-11. Elsewhere agreement at 1e-16 is taken as joint correctness; `tests/test_black_scholes.py` checks ours against mpmath on its own grid.
- QuantLib's mechanism is inferred, not read from source.

## 8. Next questions
- Study A: price targets will come from mpmath (SPEC 8.3). This run shows a double-precision library cannot be trusted for tail targets.
- Does QuantLib's tail behaviour also affect its implied-volatility routines? Out of scope here; worth a line in Study A's limitations.
