# Phase 2: Monte Carlo interval coverage, bias and error scaling

Status: REGISTERED. Sections 1 to 3 were committed before the experiment code existed or ran.

## 1. Question
Do the Monte Carlo estimators of `pricing.monte_carlo` (plain, antithetic, control variate with an independent pilot) give unbiased estimates and 95% intervals that contain the true price 95% of the time? Where do the normal-approximation intervals fail, and how much do the variance-reduction methods save at equal cost? The answers decide how far the intervals can be trusted (SPEC 5.1, 5.5).

## 2. Hypotheses
With R = 2000 replications per cell, the binomial standard error of an empirical coverage near 0.95 is sqrt(0.95 x 0.05 / 2000) = 0.0049.
- C1, calibration: for n >= 1000, empirical coverage lies in [0.935, 0.965] (0.95 +/- 3 x 0.0049) in all 27 cells (3 strikes x 3 methods x n in {1e3, 1e4, 1e5}).
- C2, small-sample undercoverage: at n = 100, deep out of the money (K = 140), plain Monte Carlo covers less than 0.935. The payoff is zero about 95% of the time, so the sample mean is strongly skewed and the normal approximation is poor (SPEC 5.1).
- C3, no bias: in every one of the 36 cells, |mean estimate - BSM| <= 4 x sd(estimates) / sqrt(R). The 4-sigma threshold keeps the chance of a false alarm across 36 cells near 0.2%.
- C4, error scaling: for each method and strike, the least-squares slope of log(mean reported SE) on log(n), over n in {1e2, 1e3, 1e4, 1e5}, lies in [-0.52, -0.48] (theory: -1/2).
- C5, variance reduction at n = 1e4, with VRF = Var_plain(estimate) cost_plain / (Var_method(estimate) cost_method), using the empirical variance of the 2000 estimates and counting pilot paths in cost:
  - antithetic VRF > 1 at every strike (the payoff is monotone in Z);
  - control-variate VRF > 1 in and at the money, largest in the money;
  - control-variate VRF lower deep out of the money than at the money.

## 3. Design
- Contract: European call, S = 100, T = 1, r = 0.05, q = 0.02, sigma = 0.20. Strikes K = 80 (in the money), 100 (at the money), 140 (deep out of the money; P(S_T > K) is about 5%).
- Truth: the BSM closed form, verified in Phase 1 against mpmath and QuantLib to about 1e-15 relative.
- Grid: n in {1e2, 1e3, 1e4, 1e5} paths; 3 methods x 3 strikes x 4 n = 36 cells. Control variate: pilot m = max(100, n // 10) paths from an independent stream.
- Replications: R = 2000 per cell. The replication seed is `SeedSequence(20261030, spawn_key=(method, strike, n, replication))` (indices into the config lists), which is what `SeedSequence(20261030).spawn(...)` produces per cell. The root seed is recorded.
- Metrics per cell:
  - coverage: the fraction of intervals containing the BSM price;
  - bias = mean(estimate) - BSM, with its standard error sd(estimate) / sqrt(R);
  - RMSE = sqrt(mean((estimate - BSM)^2));
  - calibration ratio: mean reported SE / sd(estimate);
  - VRF as defined in C5.
- Config: `experiments/configs/mc_coverage.toml`. Outputs: `experiments/results/tables/mc_coverage.csv` and `mc_coverage_summary.json`, recording commit, dirty flag, library versions and platform.

## 4. Results
(Added after the run.)

## 5. Verdicts

## 6. Interpretation

## 7. Limitations and threats to validity

## 8. Next questions
