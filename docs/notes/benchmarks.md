# Phase 6: performance benchmarks

Status: COMPLETE. Sections 1 to 3 registered at commit `7e727fb`, before the benchmark code existed. Results from commit `7d3d864`.

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
- Code: commit `7d3d864`, clean tree. Command: `python -m experiments.benchmarks --config experiments/configs/benchmarks.toml` (about 5 s).
- Environment:
  - hardware and OS: Apple M5, 10 logical CPUs, macOS 26.6 arm64;
  - software: Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1, matplotlib 3.11.2, BLAS Accelerate;
  - thread variables (OMP, OpenBLAS, MKL, vecLib, numexpr): all unset.
- Outputs: `experiments/results/tables/benchmarks.csv`, `benchmarks_summary.json` and `experiments/results/figures/benchmarks.png`.
- History: a first run at `1680145` gave the same verdicts and numbers within a few percent. The figure then gained slope guides (`7d3d864`), and the run reported here is from that commit.

Vectorized BSM, call (median, with IQR):

| Function | Batch | Batch time | Per option | Options per second |
|---|---|---|---|---|
| price | 1 (scalar) | 23.0 us (22.7 to 23.2) | 23,000 ns | 4.35e4 |
| price | 1,000 | 50.2 us (49.7 to 50.6) | 50.2 ns | 1.99e7 |
| price | 1,000,000 | 21.0 ms (20.8 to 21.1) | 21.0 ns | 4.76e7 |
| delta | 1 (scalar) | 23.8 us (23.5 to 24.0) | 23,800 ns | 4.21e4 |
| delta | 1,000 | 45.5 us (45.0 to 45.7) | 45.5 ns | 2.20e7 |
| delta | 1,000,000 | 16.1 ms (16.0 to 16.1) | 16.1 ns | 6.22e7 |

Trees (median time per price; R4c, European / American):

| Method | N ~ 100 | N ~ 3200 | Fitted exponent, N >= 400 |
|---|---|---|---|
| CRR | 0.11 / 0.40 ms | 5.1 / 30.4 ms | 1.18 / 1.36 |
| LR | 0.12 / 0.41 ms | 5.1 / 30.6 ms | 1.18 / 1.36 |
| BBS | 0.15 / 0.44 ms | 5.4 / 30.5 ms | 1.15 / 1.35 |
| BBSR | 0.24 / 0.68 ms | 7.6 / 41.4 ms | 1.11 / 1.31 |

Monte Carlo (median time per path, at-the-money call):

| Method | n = 1e4 | n = 1e6 | Paths per second at 1e6 |
|---|---|---|---|
| plain | 10.4 ns | 9.1 ns | 1.09e8 |
| antithetic | 9.1 ns | 7.4 ns | 1.35e8 |
| control variate (pilot n / 10 included) | 13.9 ns | 10.6 ns | 9.44e7 |

Post-hoc analysis of tree scaling (not pre-registered).
- Local exponents between consecutive N rise steadily:
  - CRR European: 0.72, 0.94, 1.00, 1.10, 1.17, 1.28;
  - CRR American: 0.94, 1.09, 1.07, 1.22, 1.34, 1.54.
- A fit of t = a N + b N^2 to American trees gives:
  - b = 1.6 to 2.2 ns per node;
  - a = 3.7 to 5.9 us per layer;
  - a crossover a / b of 2,000 to 2,900 steps;
  - maximum residuals of 9% to 23%.
- The same model fits European trees poorly (residuals 23% to 50%; b about 0.18 ns per node, crossover 5,900 to 8,900 steps). At small N even the per-layer cost is dominated by a fixed per-call cost that this model lacks.

## 5. Verdicts
- P1: REFUTED. The fitted exponents over N in [400, 3200] are 1.11 to 1.18 for European trees and 1.31 to 1.36 for American trees, not in [1.7, 2.3].
- P2: SUPPORTED. Time per option falls 1,095 times (price) and 1,478 times (delta) from batch 1 to batch 1e6.
- P3: SUPPORTED. Time per path varies by factors of 1.13 (plain), 1.23 (antithetic) and 1.31 (control variate) across n = 1e4 to 1e6.
- P4: SUPPORTED. At n = 1e6, antithetic costs 7.4 ns per path against plain's 9.1 ns.

## 6. Interpretation
- **The trees are O(N^2) in work but not yet in time at these step counts.** Each of the N layers runs a few NumPy operations with a fixed cost of about 1 to 6 us, plus about 0.2 ns (European) to 2 ns (American) per node.
  - Below roughly 2,000 to 3,000 steps for American trees, and several thousand for European ones, the per-layer cost dominates and runtime grows close to linearly. The local exponents climb toward 2 as N grows.
  - American layers cost about ten times more per node than European ones, because the node prices are recomputed with exp at every layer. That brings them into the quadratic regime sooner.
  - SPEC 4's O(N^2) is the asymptotic statement; the registered hypothesis wrongly assumed it would already hold at N = 400.
- **Vectorization is what makes BSM fast here.** A scalar call spends about 23 us on validation and Python overhead. At a million options the cost per option is about 21 ns (price) or 16 ns (delta), about 50 million options per second on one core, with no compiled code.
- **Monte Carlo is linear in paths once n reaches about 1e4.** Antithetic is cheapest per path because it draws half as many normals. The control variate pays for its pilot and extra arithmetic, about 15% over plain, which its variance reduction repays many times in the money (Phase 2: VRF 44).
- **Is a C++ kernel warranted (SPEC 12)?** Not on this evidence. Nothing in the project's requirements is limited by these runtimes. The clearest optimization targets would be precomputing the node-price lattice once instead of exponentiating every layer, and a compiled tree loop. Both are measurable future work, not needed for v1.0.

## 7. Limitations and threats to validity
- One machine, one Python build, single-threaded NumPy (SPEC 15): not evidence of production performance.
- Wall-clock medians with the collector paused. Other processes on the machine were not controlled beyond running the benchmark alone; The IQR was 1.5% of the median for the median measurement, 3.3% at the 90th percentile and 12.3% at worst.
- Trees were timed on one American put case; runtime depends on N and method, not on market parameters, but exercise style matters (American layers do more work).
- The post-hoc scaling model is descriptive and was chosen after seeing the data.

## 8. Next questions
- Precompute the CRR/BBS node-price lattice (it depends only on 2j - step) and remeasure the American exponent.
- Measure the crossover to quadratic scaling directly with N up to 2^14.
- A compiled kernel (pybind11, SPEC 12), only with a parity test against the Python reference.
