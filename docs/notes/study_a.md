# Study A: implied-volatility solver accuracy, robustness and cost

Status: REGISTERED. Sections 1 to 3 were committed before the study code existed or ran.

## 1. Question
(SPEC 10.) Across the (log-moneyness, total-volatility) plane, how do Brent, naive Newton and safeguarded Newton compare with Jaeckel's Let's Be Rational (LBR) in accuracy, robustness and cost, when accuracy is measured against what the problem's conditioning allows? The answer tells a desk which solver it can trust, and where, when it backs volatilities out of prices.

## 2. Hypotheses
Definitions are in section 3. "Well posed" means the float64 target price c satisfies c >= eps max(F, K), so it has neither underflowed nor fallen below the price scale's rounding. Every other grid point is "ill posed".

- A1, naive Newton fails regionally. It fails (reason DIVERGED, ZERO_DERIVATIVE or MAX_ITERATIONS) on at least 50% of well-posed points with s <= 0.1 and |x| >= 1. It fails on at most 10% of well-posed points with s >= 0.2 and |x| <= 0.5.
- A2, Brent is robust but slow and tolerance-limited:
  - (a) Brent never returns MAX_ITERATIONS or BRACKET_FAILURE at a well-posed point;
  - (b) its median iteration count over well-posed points is at least 6, three times LBR's fixed 2 Householder iterations;
  - (c) on well-posed points with s <= 1e-2 its median e_norm is at least 10, because its absolute stopping tolerance xtol = 1e-15 is large relative to small s.
- A3, LBR is as accurate as the problem permits. Over well-posed points, LBR's e_norm has median <= 2 and 99th percentile <= 20.
- A4, the ill-posed region is set by representability, not method:
  - (i) where the target price is exactly 0, no method recovers s: each returns s = 0 or fails;
  - (ii) where the target is positive but ill posed, the relative errors of Brent, safeguarded Newton and LBR agree within a factor of 10 at >= 90% of the points where all three return a value.
- A5, added in Phase 4 from the Phase 3 finding. Near the money at small total volatility our solvers are limited by cancellation in evaluating c(s), not by kappa. On well-posed points with |x| <= 0.5 and s <= 1e-2, safeguarded Newton's median e_norm is at least 10, while LBR's is at most 2.

## 3. Design
- Problems: forward F = 100, T = 1 (so sigma = s), strike K = F exp(-x) in float64.
  - x from -6 to 6 in steps of 0.25: 49 values.
  - s: 60 values log-spaced from 1e-4 to 4.
  - Calls and puts: 5880 problems.
- Ground truth (SPEC 8.3): for each float64 (F, K, s), mpmath at 50 significant digits computes the undiscounted price c_true and dc/ds = F phi(d1). The target is c_true rounded once to float64.
- Solvers:
  - ours, through `pricing.implied_vol.solve_black(option_type, c, F=F, K=K, T=1, method=...)` for Brent, naive Newton and safeguarded Newton (SPEC 6.3 as amended in Phase 3);
  - LBR, through `pricing.oracles.lbr_implied_volatility` (`lets-be-rational` 1.1.2, 2 Householder iterations).
- Metrics, per solver and point:
  - outcome class: CONVERGED, AT_LOWER_BOUND (returned s = 0, including LBR returning 0), MAX_ITERATIONS, BRACKET_FAILURE, DIVERGED, ZERO_DERIVATIVE, or INFEASIBLE (the feasibility check rejected the rounded target);
  - for any returned s_hat: e_rel = |s_hat - s| / s and e_ulp = |s_hat - s| / spacing(s);
  - kappa = |c_true / (s dc/ds)| (mpmath);
  - e_norm = e_rel / (max(kappa, 1) eps). The max with 1 is a registered deviation from SPEC 10's e_rel / (kappa eps): s_hat is a float64, so one unit in the last place is the floor for any method, and deep out of the money kappa ~ s^2 / x^2 falls far below 1. SPEC's raw version is reported too.
  - iterations (ours: from IVResult; LBR: 2 by construction) and function evaluations (ours).
- Timing: for every point and solver, one warm-up call, then 50 timed calls (`time.perf_counter_ns`). Per point, the median; per solver, the median and interquartile range over well-posed points (SPEC 12: never the fastest run). All four solvers are pure Python plus SciPy special functions, so the comparison is like for like, but not representative of compiled implementations.
- Deliverables (SPEC 10):
  - heatmaps of log10(e_ulp) and log10(e_norm) per solver, calls and puts;
  - failure maps;
  - iteration histograms;
  - a timing table;
  - a verdict on each hypothesis.
  
  Because the user likes 3D views, log10(e_norm) is also drawn as a 3D surface over (x, log10 s) per solver, paired with the heatmaps.
- No randomness. Config: `experiments/configs/study_a_iv.toml`. Outputs under `experiments/results/` with commit, dirty flag, library versions and platform.

## 4. Results
(Added after the run.)

## 5. Verdicts

## 6. Interpretation

## 7. Limitations and threats to validity

## 8. Next questions
