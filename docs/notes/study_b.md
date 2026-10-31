# Study B: convergence of binomial trees for European and American options

Status: COMPLETE. Sections 1 to 3 registered at commit `abc9ca2`, before the study code existed. Results from commit `71a75d9`.

## 1. Question
(SPEC 11.) What convergence order and what error-versus-runtime trade-off do CRR, Leisen-Reimer (LR), BBS and BBSR achieve for European and American options? Do they reproduce published benchmarks? The answer tells a desk which tree to use for American options, and at what step count, for a given error budget.

## 2. Hypotheses
Error e_N = V_N - V_ref. An "order fit" is the least-squares slope of log10|e_N| on log10 N, with its standard error, over the window given in section 3.

- B1, CRR European (R1 call):
  - (a) the slopes fitted separately over odd and over even N >= 100 both lie in [-1.2, -0.8];
  - (b) it oscillates between odd and even N: over N in [100, 1999], the median of |e_N - e_{N+1}| / max(|e_N|, |e_{N+1}|) is at least 0.5;
  - (c) naive Richardson extrapolation V_RE(N) = 2 V(2N) - V(N) (order 1) is unreliable: for N in [100, 1000] it improves on V(2N), meaning |e_RE| < |e_{2N}|, at fewer than 90% of N.
- B2, LR European (R1 call): the slope over odd N >= 101 lies in [-2.2, -1.8].
- B3, LR American puts (the 13 put cases of R4, R4b and R6): the median of the per-case slopes over odd N >= 101 lies in [-1.5, -0.5]. The order falls toward 1 because the early-exercise boundary does not align with the nodes; the mechanism is not tested directly.
- B4, BBSR has the lowest error at fixed runtime for American puts:
  - budgets of 1, 5 and 25 ms;
  - for each of the 13 put cases and each budget, take each method's absolute error at the largest timed N whose median runtime fits the budget;
  - BBSR has the smallest error in at least 75% of the 39 (case, budget) pairs.

## 3. Design
- Cases:
  - European: R1 call (S = 42, K = 40, T = 0.5, r = 0.10, q = 0, sigma = 0.2), reference the BSM closed form (Phase 1: about 1e-15 relative).
  - American: the 18 Ju and Zhong cases (R4 including R4b, R5, R6) at maturities QuantLib can represent exactly:
    - R4: Actual360 with 30, 120, 210 days, i.e. T = 1/12, 1/3, 7/12 for the printed 0.0833, 0.3333, 0.5833;
    - R5: 180/360;
    - R6: 1095/365.

    The trees use the same T.
- American reference: QuantLib `QdFpAmericanEngine` with `QdFpTanhSinhIterationScheme(20, 60, 1e-14)`.
  - Found during design: QuantLib's built-in `highPrecisionScheme()` differs from stricter schemes by up to 6e-8 on these cases, too coarse for tree errors near 1e-7.
  - Gate 1: this scheme agrees with (40, 100, 1e-15) within 1e-9 on all 18 cases.
  - Gate 2 (SPEC 11): it agrees with Ju-Zhong's published values within 2e-3.
  - If either gate fails, the SPEC 8.1 fallback (BBSR N = 2^15 cross-checked with LR N = 32769) is used instead.
  - Tree errors below 1e-8 are beyond the reference's resolution: they are excluded from order fits and counted.
- Step counts (SPEC 11): CRR every N from 10 to 2000; LR odd N from 11 to 2001; BBS and BBSR N = 2^k for k = 4 to 13.
- Order-fit windows:
  - CRR: N >= 100, odd and even separately; also the envelope, the maximum |e_N| in each block of 10 consecutive N starting at 100;
  - LR: odd N >= 101;
  - BBS and BBSR: N >= 64.
- Runtime: on one representative case (R4, K = 40, sigma = 0.2, T = 7/12, American put), one warm-up and 7 timed repeats, median, at:
  - CRR N in {10, 20, 40, 80, 160, 320, 640, 1280, 2000};
  - LR N in {11, 21, 41, 81, 161, 321, 641, 1281, 2001};
  - BBS and BBSR N = 2^k.

  Runtime depends on N and method, not on the case's parameters. Error sweeps run in parallel processes; timings run alone.
- Deliverables (SPEC 11):
  - log-log error plots;
  - an order table with standard errors;
  - an error-versus-runtime Pareto plot;
  - a Ju-Zhong reproduction table;
  - verdicts.

  Reporting only, not a hypothesis: a 3D surface of CRR's European error over (strike, N) with its 2D heatmap, showing how the odd/even pattern depends on where the strike falls between nodes.
- No randomness. Config: `experiments/configs/study_b_trees.toml`. Outputs under `experiments/results/` with commit, dirty flag, library versions and platform.

## 4. Results
- Code: commit `71a75d9`, clean tree. Command: `python -m experiments.study_b_trees --config experiments/configs/study_b_trees.toml`. The error sweeps run in 9 parallel processes; the timings run alone.
- Environment: macOS 26.6 arm64, Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1, matplotlib 3.11.2, QuantLib 1.44.
- Outputs:
  - tables in `experiments/results/tables/`: `study_b_errors.csv` (every tree price and error), `study_b_orders.csv` (order fits) and `study_b_summary.json` (gates, Ju-Zhong table, timings, B4 pairs, verdicts);
  - figures in `experiments/results/figures/`: `study_b_european.png`, `study_b_american_puts.png`, `study_b_pareto.png`, and `study_b_crr_surface_3d.png` with `study_b_crr_surface_heatmap.png`.

**Reference gates.**
- Gate 1: the scheme (20, 60, 1e-14) agrees with (40, 100, 1e-15) within 7e-10 on all 18 cases.
- Gate 2: it agrees with Ju-Zhong within 6.4e-4 (worst case R5e).

Both pass.

**Order fits** (slope of log10|e| on log10 N, with its standard error):

| Case | CRR odd / even / envelope | LR | BBS | BBSR |
|---|---|---|---|---|
| R1 European call | -1.079 / -0.914 / -1.013 (0.022) | -1.995 (7e-5; 907 points, 44 below 1e-8 excluded) | -0.998 | -1.017 (0.092) |
| American puts (13), median and range | envelope -1.02 (-1.11 to -1.00) | -1.00 (-1.13 to +0.17) | -0.95 (-1.04 to -0.75) | -1.24 (-1.45 to -0.74) |
| American calls (5), range | envelope -1.02 to -1.00 | -1.97 to -0.91 | -1.00 to -0.89 | -1.99 to -1.13 |

**B1 details (R1).**
- Odd-N slope -1.079, even-N slope -0.914.
- Oscillation statistic 1.28: consecutive errors typically differ by more than their own size, i.e. they change sign.
- Naive Richardson on CRR improves on V(2N) at only 32.4% of N in [100, 1000].

**The one outlying LR slope, R4b (+0.17).** Its error changes sign near N = 220:
- -3.6e-5 at N = 101 and -4.4e-7 at N = 201;
- then positive, peaking at +5.2e-6 near N = 801 and down to +3.0e-6 at N = 2001.

Two error components of opposite sign cancel, and the registered window N >= 101 straddles the crossing. Post hoc, over N >= 801 the slope is -0.46 (0.008): this case is still pre-asymptotic at N = 2001.

**Runtime** (representative case R4c, American, median of 7 repeats, pure Python):

| N | CRR | LR | BBS | BBSR |
|---|---|---|---|---|
| ~160 | 0.64 ms (160) | 0.71 ms (161) | 0.55 ms (128) | 0.90 ms (128) |
| ~640 | 3.6 ms (640) | 3.4 ms (641) | 2.4 ms (512) | 3.8 ms (512) |
| ~2000 | 16.1 ms (2000) | 15.4 ms (2001) | 15.5 ms (2048) | 21.3 ms (2048) |
| 8192 | - | - | 161 ms | 206 ms |

**B4.** BBSR had the smallest error in 33 of the 39 (case, budget) pairs (84.6%). The exceptions:
- LR won 5: R4b at all three budgets, during its zero crossing; R4f at 1 and 5 ms.
- CRR won 1: R6e at 25 ms, with 2.6e-5 against BBSR's 2.6e-5.

On R4c (Pareto figure) LR and BBSR are nearly tied up to about 10 ms; BBSR leads at larger budgets.

**Ju-Zhong reproduction** (reference minus published; tree error at the largest N):
- reference minus published lies between 1.1e-5 and 6.4e-4 for all 18 cases;
- at the largest N the worst tree errors are:
  - CRR (N = 2000): 1.6e-3;
  - LR (N = 2001): 9.2e-4;
  - BBS (N = 8192): 1.5e-4;
  - BBSR (N = 8192): 5.8e-6.

  The full table is in `study_b_summary.json` (`juzhong`).

## 5. Verdicts
- B1: SUPPORTED.
  - (a) odd and even slopes -1.079 and -0.914, both in [-1.2, -0.8];
  - (b) oscillation statistic 1.28 >= 0.5;
  - (c) Richardson improves at only 32.4% of N, below 90%.
- B2: SUPPORTED. LR European slope -1.995 (standard error 7e-5).
- B3: SUPPORTED. The median LR slope over the 13 American puts is -1.00. Twelve cases lie in [-1.13, -0.89]; R4b is pre-asymptotic (see section 4).
- B4: SUPPORTED. BBSR is best in 84.6% of pairs, threshold 75%.

## 6. Interpretation
- **The payoff kink sets the order.** For a European option CRR is first order and oscillates (B1). LR, which centres the tree on the strike, reaches second order (B2) and is the best European tree by far: 9e-9 at N = 2001, against about 1e-4 for CRR.
- **For American puts, LR's advantage disappears (B3).** The early-exercise boundary moves through the tree and cannot sit on nodes at every step, so the error is first order again. LR then behaves like a smoother CRR: order -1.00 with no odd/even scatter.
- **BBSR is the most accurate American method at a given cost**, with a median order of -1.24 and the smallest error in 85% of budget comparisons. Two caveats follow.
  - BBSR's error is not a clean power law. It changes sign as N doubles, so on the European R1 its order is only -1.02 and its error at a given N is partly luck. Richardson assumes one smooth error term; BBS has a small oscillating component that survives the extrapolation.
  - At small budgets (1 to 5 ms) LR is as good as BBSR or better on some cases. BBSR's lead appears at larger N.
- **"High precision" needs checking.** QuantLib's built-in `highPrecisionScheme()` is good only to about 6e-8, which would have been the noise floor for BBSR's best errors. A stricter scheme, validated to 7e-10, was needed.
- **The strike's position decides CRR's pattern.** The (strike, N) surface shows CRR's error as a checkerboard over odd and even N, with a different band at the money, where the strike sits on a node for every even N.

## 7. Limitations and threats to validity
- Pure Python on one machine. The runtime ranking could change in compiled code, where BBS's per-node Black-Scholes evaluations cost relatively more or less.
- Order fits over a fixed window can straddle a sign change (R4b) or include pre-asymptotic points. The registered B3 statistic, a median over cases, is robust to that; single-case slopes need care.
- N >= 64 for BBS and BBSR gives only 8 points per fit. Their slopes' standard errors range from below 0.001 to 0.37; the large ones come from BBSR's sign changes, so single-case BBSR slopes are uncertain.
- B4 compares methods at timed N only, so each budget sees a discrete set of N per method.
- R4 maturities use the exact twelfths (1/12, 1/3, 7/12), not the printed 0.0833, 0.3333, 0.5833; Phase 2 measured the effect at up to 1.6e-4.

## 8. Next questions
- Smooth BBSR's sign-changing error, for example with Richardson over three levels, or by smoothing BBS's last layer differently.
- Joshi (2009)'s Tian tree with truncation and smoothing, a SPEC 14 stretch goal, against LR and BBSR on the same cases.
- Compiled trees (SPEC 12) to see whether the runtime ranking holds.
