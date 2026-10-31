# Study A2: do the solver fixes close Study A's gaps?

Status: REGISTERED. Sections 1 to 3 were committed before the study code existed, and after the fixes they test (commit `0a1566d`). No A2 grid point has been solved before registration.

## 1. Question
Study A found two weaknesses in our safeguarded Newton solver:
- near the money at small total volatility, cancellation in c(s) left its median normalized error at 683, where Let's Be Rational (LBR) reaches 0;
- on 30 informative in-the-money prices it returned s = 0, and LBR solved 16 of them.

Commit `0a1566d` changed `pricing.implied_vol` in two ways:
- it solves in-the-money problems as their out-of-the-money counterpart (put-call parity), with a time-value threshold of 2 eps max(F, K) instead of 8;
- it recomputes c(s) with extra digits where the double formula cancels (SPEC 6.2 and 6.3, Study A2 amendments).

Do these changes make the safeguarded Newton as accurate as the problem's conditioning allows across the plane, on Study A's grid and on a new grid it was never tuned on? The answer decides whether the README can drop the two IV limitations.

## 2. Hypotheses
"Informative" is Study A's post-hoc rule, registered here in advance: the float64 target c is a normal number (not 0 or subnormal), and kappa eps < 1. All other definitions are Study A's (section 3 there): e_norm = e_rel / (max(kappa, 1) eps); SN is the safeguarded Newton.

Each hypothesis must hold on both grids (section 3):
- H1, near-money accuracy. On informative points with |x| <= 0.5 and s <= 1e-2, SN's median e_norm is at most 4. Study A measured 683.
- H2, in-the-money recovery. At every informative point where LBR converges (returns s > 0), SN converges too.
- H3, overall accuracy. Over informative points, SN's e_norm has median <= 2 and 99th percentile <= 20. These are the thresholds LBR met in Study A (A3); SN's 99th percentile there was 1087.
- H4, robustness. At informative points, neither SN nor Brent returns MAX_ITERATIONS, BRACKET_FAILURE or INFEASIBLE.
- H5, the informative rule transfers. Over informative points of the new grid, LBR's e_norm has 99th percentile <= 20, as on Study A's grid (1.99). This tests that the rule marks solvable problems on a grid it was not derived from.
- H6, cost. SN's median iteration count over informative points is at most 12. Study A measured 9, and the extra-digit evaluation must not buy accuracy with many more iterations.

## 3. Design
- **Grids.** Both use calls and puts and the strike K = F exp(-x) in float64.
  - A (Study A's own, for a before/after comparison on identical points): F = 100, T = 1, x from -6 to 6 in steps of 0.25 (49 values), s log-spaced from 1e-4 to 4 (60 values). 5880 problems.
  - New: F = 37.5, T = 0.4, x from -5.875 to 5.875 in steps of 0.25 (48 values, none on grid A), s log-spaced from 1.3e-4 to 3.7 (57 values). 5472 problems.
- **Ground truth, solvers, outcome classes and metrics:** exactly Study A's (`experiments/study_a_iv.py`, reused unchanged):
  - mpmath at 50 digits;
  - Brent, naive Newton, safeguarded Newton and LBR (`lets-be-rational` 1.1.2).
- **Timing.** Not part of any hypothesis. Each solve is timed with 5 repeats only to show the cost of the extra-digit path; the timings are reported, not tested.
- **Before/after.** Grid A's "before" numbers are Study A's committed results (`experiments/results/tables/study_a_iv.csv`, commit `67539c5`), read, not recomputed.
- **Outputs.**
  - Per grid: a table, e_norm heatmaps and the 3D surface (Study A's plotting functions), and a summary with the six verdicts.
  - Config: `experiments/configs/study_a2_iv.toml`.
  - Script: `experiments/study_a2_iv.py`.
- **Randomness:** none.
