# Phase 3: finite-difference bump study

Status: REGISTERED. Sections 1 to 3 were committed before the experiment code existed or ran.

## 1. Question
How does the error of a finite-difference Greek depend on the bump size? Does the optimal bump match the theoretical prediction of SPEC 7: eta about eps^(1/3) = 6.1e-6 for first derivatives and eps^(1/4) = 1.2e-4 for second derivatives? A desk that computes Greeks by bumping a pricer has to choose this bump. Too large a bump gives truncation error, too small a bump amplifies rounding. This study measures where the trade-off sits.

## 2. Hypotheses
Error model for a central difference with bump h on a function computed with relative rounding error about eps:
- first derivative: E(h) ~ (h^2 / 6) |f'''| + eps |f| / h;
- second derivative: E(h) ~ (h^2 / 12) |f''''| + 4 eps |f| / h^2.

In log-log axes this traces a V. The right branch (truncation) has slope +2. The left branch (rounding) has slope -1 for first derivatives and -2 for second derivatives. The minimum lies near h ~ eps^(1/3) and eps^(1/4) on the variable's own scale.

- D1, V-shape slopes, from least-squares fits of log10(relative error) on log10(eta):
  - right branch over eta in {1e-3, 10^-2.5, 1e-2}: slope in [1.7, 2.3] for delta, gamma, vega, theta and rho;
  - left branch over eta in {1e-12, 10^-11.5, 1e-11, 10^-10.5, 1e-10}: slope in [-1.5, -0.5] for the first derivatives and [-2.5, -1.5] for gamma.
- D2, optimum location: the eta with the smallest error lies within a factor of 10 of the SPEC prediction, i.e. in [6.1e-7, 6.1e-5] for delta, vega, theta and rho, and in [1.2e-5, 1.2e-3] for gamma.
- D3, best attainable error: the minimum relative error is at most 1e-8 for each first derivative and at most 1e-6 for gamma. Theory gives about eps^(2/3) = 3.7e-11 and eps^(1/2) = 1.5e-8, times problem-dependent constants.
- D4, one-sided differences near T = 0: for theta at T = 1 day, where a central difference is unavailable once h >= T, the right-branch slope over eta in {1e-5, 10^-4.5, 1e-4, 10^-3.5} lies in [0.7, 1.3] for the forward first-order difference and [1.7, 2.3] for the three-point forward difference (truncation orders 1 and 2).

## 3. Design
- Contract: European call, K = 100, T = 0.5, r = 0.05, q = 0.02, sigma = 0.25; main case S = 100.
- Greeks and bumped variable:
  - delta: central first difference in S;
  - gamma: central second difference in S;
  - vega: central first difference in sigma;
  - theta: minus the central first difference in T;
  - rho: central first difference in r.
- Bumps: h = eta x max(|x|, 1), with x the bumped variable's value (SPEC 7, x_scale = 1). eta runs over 23 values, 10^-1, 10^-1.5, ..., 10^-12.
- Reference: the analytic Greeks of `pricing.black_scholes`, verified in Phase 1 against mpmath, finite differences and QuantLib to about 1e-13 relative or better here.
- Metric: relative error |FD - analytic| / |analytic|. The optimum is the grid eta with the smallest error.
- D4 case: T = 1/365, theta by `forward_first` and `forward_first_order2` (labelled one-sided), and by the central difference where T - h > 0.
- Figures (reporting only, not hypotheses):
  - the V-shape for all five Greeks, with predicted optima marked;
  - for delta and gamma, a 3D surface of log10(relative error) over (log10 eta, S) for S = 70 to 130 in steps of 5, with the same data as a 2D heatmap;
  - the one-sided comparison at T = 1 day.
- No randomness. Config: `experiments/configs/greek_bumps.toml`. Outputs: tables and figures under `experiments/results/`, with commit, dirty flag, library versions and platform.

## 4. Results
(Added after the run.)

## 5. Verdicts

## 6. Interpretation

## 7. Limitations and threats to validity

## 8. Next questions
