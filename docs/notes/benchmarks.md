# Phase 6: performance benchmarks

Status: REGISTERED. Sections 1 to 3 were committed before the benchmark code existed or ran.

## 1. Question
(SPEC 12.) How fast is the library on one machine? Specifically: the latency and throughput of vectorized BSM at batch sizes 1, 1e3 and 1e6; tree runtime against step count; Monte Carlo cost per path. Does each scale as its algorithm predicts? A user sizing a batch or choosing N needs these numbers, with their spread and the environment that produced them.

## 2. Hypotheses
- P1: tree runtime scales as N^2 (SPEC 4: O(N^2) time). For CRR, LR, BBS and BBSR, European and American, the least-squares slope of log(median runtime) on log N over N >= 400 lies in [1.7, 2.3].
- P2: vectorization amortizes Python overhead. BSM's median time per option at batch 1e6 is at least 100 times lower than at batch 1, for both price and delta.
- P3: Monte Carlo cost is linear in paths once overhead is amortized. For each method (plain, antithetic, control variate), the median time per path varies by less than a factor of 2 across n = 1e4, 1e5 and 1e6.
- P4: antithetic paths cost no more than plain ones. At n = 1e6, antithetic's median time per path is at most plain's, since it draws half as many normal variates.

## 3. Design
- Timer: `time.perf_counter_ns`, with the garbage collector paused during each timed block (as `timeit` does). Each measurement has a warm-up, then repeated timed calls; we report the median and interquartile range, never the fastest run (SPEC 12).
- BSM: `black_scholes.price` and `delta` for a call.
  - Batch 1 is a scalar call.
  - Batches of 1e3 and 1e6 use arrays: S evenly spaced in [50, 150], K = 100, T = 1, r = 0.03, q = 0.01, sigma = 0.2.
  - Repeats: 2000, 300 and 15.
  - Reported: median batch time, its IQR, time per option, and options per second.
- Trees: CRR, LR (N + 1, odd), BBS and BBSR on R4c (American put, S = K = 40, sigma = 0.2, T = 7/12, r = 0.0488), European and American, at N in {50, 100, 200, 400, 800, 1600, 3200}. Repeats: 15 for N <= 400 and 7 above.
- Monte Carlo: `monte_carlo` for an at-the-money call (S = K = 100, T = 1, r = 0.05, q = 0.02, sigma = 0.2) at n in {1e3, 1e4, 1e5, 1e6}. The control variate's pilot is n / 10. Repeats: 50, 30, 10, 5. Reported: time per path and per payoff evaluation.
- Environment: CPU model and core count, OS, Python, NumPy and SciPy versions, NumPy's BLAS configuration, thread-related environment variables, commit and dirty flag.
- One command: `python -m experiments.benchmarks --config experiments/configs/benchmarks.toml`, writing `experiments/results/tables/benchmarks.csv`, `benchmarks_summary.json` and `experiments/results/figures/benchmarks.png`.
- Non-claim (SPEC 15): one machine's benchmarks are not evidence of production performance.

## 4. Results
(Added after the run.)

## 5. Verdicts

## 6. Interpretation

## 7. Limitations and threats to validity

## 8. Next questions
