# Study A: implied-volatility solver accuracy, robustness and cost

Status: COMPLETE. Sections 1 to 3 registered at commit `f814c1a`, before the study code existed. Results from commit `67539c5`.

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
- Code: commit `67539c5`, clean tree. Command: `python -m experiments.study_a_iv --config experiments/configs/study_a_iv.toml` (about 6 s, including 50 timed repeats per point and solver).
- Environment: macOS 26.6 arm64, Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1, mpmath 1.4.1, matplotlib 3.11.2, lets_be_rational 1.1.2.
- Outputs:
  - tables: `experiments/results/tables/study_a_iv.csv` (5880 rows) and `study_a_iv_summary.json`;
  - figures in `experiments/results/figures/`: `study_a_e_ulp.png`, `study_a_e_norm.png`, `study_a_e_norm_3d.png`, `study_a_outcomes.png`, `study_a_iterations.png`.
- History: the script was committed at `9a9f2f5` and first run there. Three later commits changed only analysis and plotting, not the solver or truth code:
  - `9226224` added the post-hoc analysis described below;
  - `32bbf76` fixed a verdict bug: a median of exactly 0.0 was treated as missing (`value or inf`), which wrongly refuted post-hoc A5;
  - `fcec7de` and `67539c5` corrected the figures (uninformative points were dominating the colour scale) and handled empty regions.

  The registered verdicts are identical in every run.

### Registered analysis (SPEC 10's well-posed region: c >= eps max(F, K))

| | Points |
|---|---|
| Grid | 5880 |
| Well posed | 3688 |
| Ill posed | 2192, of which 1752 have target exactly 0 and 440 a positive target |

| Solver | Well-posed outcomes | e_norm median / p99 |
|---|---|---|
| Brent | 1492 converged, 2196 returned s = 0 | 0 / 201 |
| naive Newton | 230 converged, 2196 s = 0, 1104 zero derivative, 94 diverged, 64 cap | 0 / 16.5 |
| safeguarded Newton | 1492 converged, 2196 s = 0 | 0 / 123 |
| Let's Be Rational | 1508 converged, 2180 s = 0 | 0 / 1.62 |

Diagnosis: all 2196 well-posed points where our solvers return s = 0 are in the money, and 2166 of them have kappa eps >= 1. These are prices whose time value lies below the rounding of the intrinsic value, so the float64 target carries no information about s. They are 60% of SPEC 10's "well-posed" region and set every median there to 0: e_norm 0 because kappa is enormous, iterations 0, time 0.9 us for an immediate lower-bound return. In the other direction, 432 points that SPEC 10 calls ill posed are deep out-of-the-money prices below eps max(F, K) that are fully informative.

### Post-hoc analysis (informative region: a normal float target and kappa eps < 1; not pre-registered)
The same thresholds as section 2, applied where the target determines s:
- 1954 informative points;
- of the 3926 uninformative points, 1752 have a target of 0 and 2174 a positive target.

| Solver | Informative outcomes | e_norm median / p99 | Median e_ulp | Converged iterations, median | Time per solve, median (IQR) |
|---|---|---|---|---|---|
| Brent | 1924 converged, 30 s = 0 | 0.65 / 640 | 5 | 12 | 9.96 us (8.38 to 13.12) |
| naive Newton | 292 converged, 30 s = 0, 1276 zero derivative, 260 cap, 96 diverged | 1.37 / 269 | 6 | 7 | 2.42 us (2.38 to 3.67) |
| safeguarded Newton | 1924 converged, 30 s = 0 | 0.86 / 1087 | 6 | 9 | 5.50 us (3.96 to 8.92) |
| Let's Be Rational | 1940 converged, 14 s = 0 | 0.16 / 1.99 | 1 | 2 (fixed) | 6.62 us (5.58 to 7.38) |

Region statistics (registered thresholds):

| Statistic | Registered region | Post-hoc region |
|---|---|---|
| A1 naive failure share, s <= 0.1 and abs(x) >= 1 | 0.0 (n = 1638) | 0.914 (n = 70) |
| A1 naive failure share, s >= 0.2 and abs(x) <= 0.5 | 0.306 (n = 170) | 0.306 (n = 170) |
| A2c Brent median e_norm, s <= 1e-2 | 0.0 | 486 |
| A4(ii) share where the three robust methods agree within 10x | 0.266 (n = 440) | 0.996 (n = 2174) |
| A5 safeguarded Newton / LBR median e_norm, abs(x) <= 0.5 and s <= 1e-2 | 0.0 / 0.0 (n = 156) | 683 / 0.0 (n = 56) |

Naive Newton's failures in A1's "calm" region are structured. For every s from 0.46 to 4 (13 values), it fails at x = -0.5 and x = +0.5, for both calls and puts (13 failures in each of the four combinations); it converges at x = 0 and +/-0.25. At s = 0.46 the failure is a zero derivative and above it divergence. Starting at s0 = 0.2 with |x| = 0.5, d1 = +/-2.6, so the slope is small and the first step overshoots to a huge s, where the slope is tiny again; the next step then goes negative.

On 30 informative in-the-money points our solvers return s = 0, because the time value is below their 8 eps max(F, K) lower-bound tolerance. On 16 of these, with kappa eps from 0.003 to 0.07, LBR recovers s with relative error at most 0.4%.

## 5. Verdicts
Registered (as computed, on SPEC 10's region):
- A1: REFUTED. The hypothesized region contains only lower-bound points (failure share 0.0). The calm region's failure share is 0.306, above 0.1.
- A2a: SUPPORTED. Brent never returns MAX_ITERATIONS or BRACKET_FAILURE.
- A2b: REFUTED. The median iteration count is 0, dominated by lower-bound returns.
- A2c: REFUTED. The median e_norm at s <= 1e-2 is 0, dominated by lower-bound points.
- A3: SUPPORTED. LBR's e_norm median is 0 and its p99 1.62.
- A4(i): SUPPORTED. No method recovers s where the target is 0.
- A4(ii): REFUTED. 26.6% agreement: these "ill-posed" points are mostly solvable deep out-of-the-money problems, and on solvable problems methods differ.
- A5: REFUTED. Both medians are 0, dominated by lower-bound points.

Post hoc (exploratory, same thresholds, informative region):
- A1: REFUTED. 91.4% failure in the hypothesized region, but 30.6% in the calm region too: naive Newton's failure is not regional.
- A2a, A2b, A2c: SUPPORTED. No bad outcomes; median 12 iterations against LBR's 2; median e_norm 486 at s <= 1e-2.
- A3: SUPPORTED. Median 0.16, p99 1.99.
- A4(i), A4(ii): SUPPORTED. 99.6% agreement where the target carries no information.
- A5: SUPPORTED. Safeguarded Newton 683 against LBR 0.0 near the money at small s.

## 6. Interpretation
- **SPEC 10's well-posedness rule measures the wrong quantity.** It looks at the size of the price c, but the information about s sits in the time value c - iota, and its recoverability is exactly what kappa measures. An in-the-money price with its time value lost in rounding looks "large and well posed" and determines nothing. A deep out-of-the-money price of 1e-20 looks "tiny and ill posed" and determines s to within an ulp. The study's own framework, attainable error kappa eps, is the right classifier. With it, the registered hypotheses behave as expected except A1.
- **Let's Be Rational is as accurate as the problem allows everywhere it is informative.** Its median is 1 ulp and its p99 e_norm 1.99, in 2 iterations. Our safeguarded Newton is robust (it converges everywhere Brent does) and faster than Brent (9 against 12 iterations, 5.5 against 10 us). Its accuracy is also limited near the money at small s, with e_norm in the hundreds. There c(s) = F N(d1) - K N(d2) is a difference of two nearly equal terms, so evaluating it loses accuracy, which kappa does not capture. LBR evaluates a normalized price designed to avoid that cancellation. This is the A5 mechanism predicted from Phase 3.
- **Brent's accuracy is set by its stopping rule at small s** (median e_norm 486), as A2 predicted: xtol = 1e-15 is a large fraction of s = 1e-4.
- **Naive Newton fails wherever it starts far from the root and off the money.** That covers the predicted corner and also large s at moderate |x|. The fixed start sigma = 0.2 is the cause, not the corner itself.
- Timing (pure Python, one machine): naive Newton is fastest because it fails fast. LBR (6.6 us) and safeguarded Newton (5.5 us) are close. Brent is slowest (10 us). LBR's cost would fall further in compiled form; that is not measured here.
- **Our lower-bound tolerance gives up too early in the money.** LBR, which works through the out-of-the-money counterpart, recovers s on 16 points where we return 0.

## 7. Limitations and threats to validity
- The post-hoc analysis was designed after seeing the registered results. It is exploratory and should be confirmed by a fresh registration on a new grid.
- Single forward (F = 100) and T = 1. Total volatility s and x are the only parameters that matter in Black form, but scale effects of F on the rounding tolerances are untested.
- Timing is pure Python on one machine (SPEC 15); medians are per point over 50 repeats, so between-point variation dominates the IQR.
- LBR's iteration count is fixed at 2 by construction; its function evaluations are not exposed, so cost is compared by time.
- e_norm uses max(kappa, 1) (registered); SPEC's raw e_norm_spec is in the table.

## 8. Next questions
- Register and test the informative-region classifier on an independent grid (different F, T and spacing).
- Improve our solvers where Let's Be Rational wins:
  - solve in-the-money problems through the out-of-the-money counterpart (put-call parity) to avoid losing the time value;
  - evaluate c(s) without cancellation near the money at small s (for example with a normalized form or Jaeckel's asymptotic expansions).
- Feed this into Phase 6's benchmarks once compiled options are considered (SPEC 12).
