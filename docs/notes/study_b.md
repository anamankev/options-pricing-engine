# Study B: convergence of binomial trees for European and American options

Status: REGISTERED. Sections 1 to 3 were committed before the study code existed or ran.

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
(Added after the run.)

## 5. Verdicts

## 6. Interpretation

## 7. Limitations and threats to validity

## 8. Next questions
