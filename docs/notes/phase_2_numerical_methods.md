# Phase 2: binomial trees, American exercise and Monte Carlo

Status: COMPLETE (tag `phase-2`). This is a phase-gate record, not a study. Its claims restate the SPEC 14 exit gate. The coverage experiment, which was registered, has its own note: `phase_2_mc_coverage.md`.

## 1. Question
Do the trees (CRR, LR, BBS, BBSR) and the Monte Carlo estimators price correctly? Do they agree with BSM, published American values and independent implementations? Do their errors behave as theory predicts?

## 2. Claims (exit gate)
- G1: cross-method tests pass. European trees and Monte Carlo agree with BSM at the predicted error rates. American values are at least European values. American calls equal European calls only under SPEC 4.4's conditions.
- G2: the Monte Carlo coverage note is committed, with hypotheses registered before the run.
- G3: Ju-Zhong R4 to R6 are reproduced within 2e-3.

## 3. Design
- Code: `src/pricing/binomial.py` (trees, Richardson, exercise boundary), `src/pricing/monte_carlo.py`, and QuantLib wrappers in `src/pricing/oracles.py`. `contracts.py` gained `ExerciseStyle` and scalar validation.
- Tests: 312 in total, of which 312 - 185 = 127 are new in this phase:

  | File | Tests | What |
  |---|---|---|
  | `test_binomial.py` | 71 | Parameters, validity, convergence orders, American exercise, boundary |
  | `test_monte_carlo.py` | 31 | 23 deterministic, 8 `statistical` |
  | `test_reference_values.py` | +18 | Ju-Zhong R4 to R6 |
  | `test_oracles.py` | +4 | Trees vs QuantLib binomial engines; QdFp (one marked `slow`) |
  | `test_mc_coverage_helpers.py` | 3 | Experiment helpers |

## 4. Results
- Commits: `2c0b63c` (CRR) to `d99fec7` (coverage results).
- Tree convergence on R1 and R2 (European), against BSM:
  - CRR: N |error| <= 0.4 for N <= 1001, changing sign between N = 1000 and 1001 (the odd/even oscillation of SPEC 4.5).
  - LR: N^2 x error = -0.036 (R1) and -0.151 (R2), constant from N = 101 to 801, i.e. O(1/N^2) and monotone.
  - BBS: N |error| <= 0.35 with one sign. BBSR beats BBS at every N tested, but its error alternates sign on R1 (-1.0e-4, +7.5e-5, -1.7e-5, +8.9e-6 for N = 64 to 512).
- American exercise: American >= European for every method tested. With q <= 0 and q <= r the American CRR call equals the European exactly. At r = -0.03, q = 0 the American call is worth more, as Healy (2021) predicts.
- Ju-Zhong: all 18 published values (R4, R4b, R5, R6) are reproduced by LR at N = 2001 and by BBSR at N = 2048. The worst difference is 1.15e-3 (LR, R6, S = 80); most are below 6.4e-4.
- QuantLib, 192 cases each:
  - Our LR matches QuantLib's LR within 8.4e-12 relative.
  - Our CRR does not match QuantLib's CRR. The gap is at least 2e-9 relative and of the size of the O(1/N) error. QuantLib's up-probability is the log-drift approximation 1/2 + (1/2)(r - q - sigma^2/2) sqrt(dt) / sigma; ours is the martingale probability SPEC 4.1 defines. Substituting QuantLib's probability into our backward induction reproduces QuantLib within 8.1e-12. SPEC 8.1's expectation of machine-precision agreement for CRR is therefore refuted as written; SPEC 8.1 now records why.
  - QdFp high-precision American engine: reproduces Ju-Zhong within 6.4e-4 and agrees with BBSR at N = 16384 within 1.5e-6. LR at N = 16001 is off by up to 1.1e-4 on the deep in-the-money R6 put.
- Monte Carlo: unbiased against BSM within 4 SE in every statistical test.
  - VRF at n = 1e4 (coverage run): antithetic 7.38, 1.88, 1.08 and control variate 44.41, 5.50, 1.22, for K = 80, 100, 140.
  - Coverage: every registered check supported; intervals undercover at n = 100 for skewed or small-sample cases (see the coverage note).

Diagnoses and decisions:
1. CRR probability convention (above). This was a SPEC expectation that turned out wrong, not a code error.
2. Numerical forms. The CRR probability uses expm1 and sinh; at dt = 1e-12 the textbook (g - d) / (u - d) loses about 1e-9 relative. The Peizer-Pratt inversion computes h and 1 - h without cancellation; at z = 30, n = 3, 1 - h would otherwise be exactly 0. LR's d uses g (1 - p_bar) / (1 - p) instead of (g - p u) / (1 - p).
3. Seeds (SPEC 2.3 amended). Monte Carlo takes an int or SeedSequence rather than a Generator, so the result can record the seed that reproduces it. Child streams are built without mutating the caller's SeedSequence.
4. The audit named the last inline tolerances, including two in Phase 1's `test_reconciliation_metrics.py` that the Phase 1 audit missed.

## 5. Verdicts
- G1: SUPPORTED.
- G2: SUPPORTED (registered at `fb102e0`, results at `f5c0d9b`).
- G3: SUPPORTED.

## 6. Interpretation
- Error order is a property of how a tree treats the payoff kink, not of the tree's size. CRR's oscillation comes from where the strike falls relative to the nodes; LR centres the tree on the strike; BBS smooths the last step analytically. The measured constants (0.4/N, 0.036/N^2, 0.11/N) are what Study B will fit formally.
- Two previews of Study B, measured but not yet tested as hypotheses:
  - LR's advantage shrinks for American puts (1.1e-4 off at N = 16001 on R6), consistent with B3.
  - BBSR's sign alternation means its error is not a clean power law, which bears on B4.
- "The same model" can hide different discretizations. Two libraries' CRR trees disagree by the full O(1/N) error, and isolating the probability convention took a controlled substitution, not a tolerance change.

## 7. Limitations and threats to validity
- Trees are scalar and pure Python plus NumPy per layer: O(N^2) time. N = 16384 takes about 0.75 s per BBSR price.
- Ju-Zhong comparisons use printed maturities (0.0833 rather than 1/12). Measured effect at most 1.6e-4, inside the 2e-3 tolerance.
- The Windows branch of the hook is still untested.
- Citations whose numbering is not yet checked against the source are listed in SPEC 17.

## 8. Next questions
- Phase 3: implied-volatility solvers, the Let's Be Rational oracle (choose between the two installed copies, SPEC 17), and the finite-difference bump study.
- Study B: formal order fits with standard errors for every method, using QdFp as the American reference.
