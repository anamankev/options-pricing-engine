# Phase 2: Monte Carlo interval coverage, bias and error scaling

Status: COMPLETE. Sections 1 to 3 registered at commit `fb102e0`, before the experiment code existed. Results from commit `f5c0d9b`.

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
- Code: commit `f5c0d9b`, clean tree. Command: `python -m experiments.mc_coverage --config experiments/configs/mc_coverage.toml` (21 s).
- Environment: macOS 26.6 arm64, Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1.
- Outputs: `experiments/results/tables/mc_coverage.csv` (36 cells) and `mc_coverage_summary.json` (cells, fits, VRFs, check outcomes).

Coverage of the 95% interval (R = 2000; binomial SE 0.0049):

| Method | K | n = 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|---|
| plain | 80 | 0.9475 | 0.9485 | 0.9560 | 0.9460 |
| plain | 100 | 0.9415 | 0.9445 | 0.9500 | 0.9565 |
| plain | 140 | **0.8305** | 0.9385 | 0.9405 | 0.9500 |
| antithetic | 80 | **0.9075** | 0.9470 | 0.9440 | 0.9470 |
| antithetic | 100 | 0.9435 | 0.9405 | 0.9490 | 0.9540 |
| antithetic | 140 | **0.8220** | 0.9395 | 0.9465 | 0.9525 |
| control variate | 80 | **0.9160** | 0.9475 | 0.9495 | 0.9545 |
| control variate | 100 | 0.9450 | 0.9465 | 0.9515 | 0.9440 |
| control variate | 140 | **0.8815** | 0.9385 | 0.9500 | 0.9485 |

- Bias: all 36 cells within 4 standard errors; the largest is 2.10 SE (plain, K = 100, n = 1e5).
- Calibration (mean reported SE / sd of estimates): 0.97 to 1.02 for n >= 1000. At n = 100 it is lower in the five undercovering cells: 0.921, 0.957, 0.882, 0.962, 0.910.
- log(SE) slope on log(n), with its standard error: plain -0.4997 +/- 0.0001, -0.4996 +/- 0.0001, -0.4868 +/- 0.0065 (K = 80, 100, 140); antithetic -0.4963 +/- 0.0019, -0.4990 +/- 0.0003, -0.4831 +/- 0.0085; control variate -0.4988 +/- 0.0029, -0.5018 +/- 0.0010, -0.4980 +/- 0.0057.
- VRF at n = 1e4, pilot cost included:

  | Method | K = 80 | K = 100 | K = 140 |
  |---|---|---|---|
  | antithetic | 7.38 | 1.88 | 1.08 |
  | control variate | 44.41 | 5.50 | 1.22 |

## 5. Verdicts
- C1: SUPPORTED. Every cell with n >= 1000 lies in [0.935, 0.965]. Deep out of the money sits near the lower edge at n = 1e3 to 1e4 (0.9385 to 0.9405 for plain and control variate).
- C2: SUPPORTED. Plain, K = 140, n = 100 covers 0.8305.
- C3: SUPPORTED (largest bias 2.10 SE).
- C4: SUPPORTED. All slopes in [-0.52, -0.48]; deep out of the money is the flattest, at -0.4831 and -0.4868.
- C5: SUPPORTED. Antithetic VRF > 1 at every strike (7.38, 1.88, 1.08). Control-variate VRF > 1 in and at the money, largest in the money (44.41), and lower deep out of the money (1.22) than at the money (5.50).

Post-hoc observations (not hypothesized):
1. At n = 100 the antithetic and control-variate methods also undercover: in the money 0.9075 and 0.9160, deep out of the money 0.8220 and 0.8815. Antithetic at n = 100 has only 50 independent pairs. A control variate's pilot of 100 paths gives a noisy beta.
2. Deep out of the money, the slopes are flatter than -1/2 because the n = 100 point is pulled down. There the sample sd underestimates the true spread (calibration 0.88 to 0.92), so a fit over n >= 1000 would sit closer to -1/2.

## 6. Interpretation
The intervals can be trusted once n reaches about 1000 for these contracts. Below that, the normal approximation fails where the per-path quantity is skewed or the effective sample is small. The deep out-of-the-money call pays nothing on about 95% of paths. Its sample sd then usually underestimates the true one, so the interval is too narrow exactly when it is most needed. This is the standard central-limit-theorem caveat, measured. The reported SE itself is unbiased at large n (calibration 0.97 to 1.02).

Variance reduction tracks how much of the payoff's variance the auxiliary quantity explains. In the money, the call moves almost one-for-one with S_T, so the control S_T removes nearly all variance (VRF 44). Deep out of the money, the payoff is a rare event only weakly correlated with S_T, so neither method helps much (VRF 1.08 and 1.22). Importance sampling, which shifts paths toward the strike, would address that case; it is not in this project's scope.

## 7. Limitations and threats to validity
- One contract family (call, S = 100, T = 1, sigma = 0.2) and three strikes; puts and other volatilities are not covered.
- R = 2000 resolves coverage to +/- 0.0049; differences smaller than about 0.01 between cells are within noise.
- The slope fit uses four points, so its standard error rests on two degrees of freedom.
- The pilot rule m = max(100, n / 10) is one choice. A larger pilot at small n would likely improve the control variate's coverage there.
- Single machine; results are deterministic given the root seed, so this affects only timing.

## 8. Next questions
- Coverage of a Student-t or bootstrap interval at n = 100 deep out of the money.
- A fit of the SE slope over n >= 1000 only.
- Importance sampling for deep out-of-the-money contracts (stretch, outside SPEC 1 scope).
