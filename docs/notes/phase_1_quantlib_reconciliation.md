# Phase 1: QuantLib reconciliation of European prices and Greeks

Status: REGISTERED. This section was committed before the reconciliation code existed or ran.

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
(Added after the run.)

## 5. Verdicts

## 6. Interpretation

## 7. Limitations and threats to validity

## 8. Next questions
