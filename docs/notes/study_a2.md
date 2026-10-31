# Study A2: do the solver fixes close Study A's gaps?

Status: COMPLETE. Sections 1 to 3 registered at commit `030cea7`, after the fixes they test (`0a1566d`) and before the study code existed; results from commit `ba1d3c6`. Follow-up (section 9) registered at `61420eb`; results from `28544ae` (section 10).

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

## 4. Results
- Code: commit `ba1d3c6`, clean tree. Command: `python -m experiments.study_a2_iv --config experiments/configs/study_a2_iv.toml` (about 1 minute).
- Environment: macOS 26.6 arm64, Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1, mpmath 1.4.1, matplotlib 3.11.2, lets-be-rational 1.1.2.
- Outputs:
  - tables: `experiments/results/tables/study_a2_grid_a.csv`, `study_a2_new_grid.csv` and `study_a2_summary.json`;
  - figures: `experiments/results/figures/study_a2_{grid_a,new_grid}_e_norm.png` and `_e_norm_3d.png`.

  The figures come from Study A's plotting code, unchanged. Its caption calls the dashed informative boundary "post hoc"; in A2 that region was registered.

Informative points: 1954 of 5880 on grid A (as in Study A), 1807 of 5472 on the new grid. Safeguarded Newton (SN) before (Study A's committed results) and after:

| Statistic (informative points) | Grid A before | Grid A after | New grid |
|---|---|---|---|
| H1: SN median e_norm, \|x\| <= 0.5 and s <= 1e-2 | 683 (n = 56) | 0.0 (n = 56) | 0.0 (n = 12) |
| H2: points LBR solves that SN does not | 16 of 1940 | 12 of 1940 | 8 of 1797 |
| H3: SN e_norm median / p99 | 0.86 / 1087 | 0.018 / 4.97 | 0.0 / 3.48 |
| H4: MAX_ITERATIONS, BRACKET_FAILURE or INFEASIBLE (Brent, SN) | 0, 0 | 0, 0 | 0, 0 |
| H5: LBR e_norm p99 | 1.99 | 1.99 | 2.74 |
| H6: SN median iterations | 9 | 8 | 9 |
| SN median time per solve (reported, not tested) | 5.5 us | 220 us | 297 us |

For comparison, after the fix:
- Brent: e_norm median 0.16 / p99 23.7 on grid A, and 0.18 / 24.7 on the new grid.
- LBR: median 0.16 / 0.20, with 2 iterations and 7 us per solve.

## 5. Verdicts
- H1: SUPPORTED. The near-money median falls from 683 to 0.0 on grid A, and is 0.0 on the new grid.
- H2: REFUTED. SN still returns s = 0 at 12 (grid A) and 8 (new grid) informative in-the-money points where LBR converges, with LBR relative errors from 0.05% to 3.9%. See section 6.
- H3: SUPPORTED. SN's p99 falls from 1087 to 4.97 on grid A, and is 3.48 on the new grid.
- H4: SUPPORTED. No bad outcomes for Brent or SN at any informative point.
- H5: SUPPORTED. LBR's p99 on the new grid is 2.74, so the informative rule marks solvable problems off the grid it came from.
- H6: SUPPORTED. 8 and 9 median iterations.

## 6. Interpretation
- **The cancellation fix works.** With c(s) evaluated to a few eps everywhere, SN is now limited by the problem's conditioning, as LBR is. Its p99 (3.5 to 5.0) is within a factor of 2.5 of LBR's (2.0 to 2.7), on a grid it was never run on.
- **Why H2 failed.** All 20 misses are deep in the money, with a time value c - iota of only 0.4 to 1.2 eps max(F, K), below the 2-eps cutoff (SPEC 6.2, Study A2 amendment).
  - That is less than an ulp of the price, yet the price still pins s. The time value is so sensitive to s (elasticity ~ x^2 / s^2) that an O(1) relative error in it moves s by under 4%.
  - Study A's informative rule (kappa eps < 1) counts these points correctly; the cutoff does not. Any positive time value carries information about s; how much is set by conditioning, not by the price's rounding.
  - The cutoff should be 0, as in LBR: solve whenever c exceeds the intrinsic value. That conclusion was drawn after seeing these results, so the change is tested as a post-hoc follow-up and confirmed on a freshly registered grid (section 9).
- **The cost is real.** SN's median time rose about 40 to 55 times, to 220-300 us, because the extra-digit path triggers wherever max(F N, K N) > 4 c. That covers out-of-the-money options generally (a cancellation factor of about 6 is typical at x = -1), not only near the money. LBR gets comparable accuracy in 7 us by evaluating a normalized Black function designed not to cancel.

## 7. Limitations and threats to validity
- Both grids use F = 100 or 37.5 and the same option types and grid shape. Neither covers extreme F/K scaling such as F = 1e-6 or 1e6.
- The study's timings come from 5 repeats with other work running on the machine. A separate quiet re-measurement on grid A's 1954 informative points (median of 3 solves per point) confirms them: SN median 5.5 us before against 223 us after, mean 7.4 against 568 us. One machine only.
- H2's follow-up was designed after seeing H2 fail. Only its registered confirmation (section 9) counts as evidence.

## 8. Next questions
- Replace the extra-digit path with Jaeckel's (2015) normalized Black function and its expansions, once their equations are checked against the paper (SPEC 17), to recover LBR-like speed.
- Confirm on grids with extreme F/K scaling.

## 9. Follow-up: time-value cutoff 0 (registered before the change or any run)
Section 6 concluded, after seeing H2 fail, that the in-the-money cutoff should be 0: solve whenever c exceeds the intrinsic value. This section registers a test of that change on a third grid that no code has yet been run on.

**Change.** In `pricing.implied_vol.solve_black`, an in-the-money price is at the lower bound (s = 0) only when its time value c - iota is at most 0. Prices up to 8 eps max(F, K) below iota stay at the lower bound, and prices further below remain infeasible (SPEC 6.2). Everything else is unchanged.

**Hypotheses.** These are judged on the confirmation grid only. "Informative" and e_norm are as above.
- F1 (H2 again): at every informative point where LBR converges, SN converges too.
- F2 (no accuracy lost): over informative points, SN's e_norm has median <= 2 and 99th percentile <= 20.
- F3 (near money still fixed): on informative points with |x| <= 0.5 and s <= 1e-2, SN's median e_norm is at most 4.
- F4 (robustness): at informative points, neither SN nor Brent returns MAX_ITERATIONS, BRACKET_FAILURE or INFEASIBLE.

**Design.**
- Confirmation grid: F = 2, T = 2.5, x from -5.95 to 5.95 in steps of 0.35 (35 values), s log-spaced from 1.7e-4 to 3.1 (53 values), calls and puts: 3710 problems. Because the Black price scales with F, only (x, s) and the rounding at this F differ from the earlier grids; none of these (x, s) pairs occurs on grid A or the new grid.
- Grids A and new are rerun with the change too. Those numbers are post hoc, since the change was designed from them, and are reported as such.
- Same code path, solvers, truth and metrics as sections 1 to 3. Script `experiments/study_a2_iv.py` (extended to evaluate each grid separately); config `experiments/configs/study_a2_followup.toml`.
- Also reported, not tested: what SN now returns at uninformative in-the-money points that previously gave s = 0.

## 10. Follow-up results
- Code: commit `28544ae`, clean tree. Command: `python -m experiments.study_a2_iv --config experiments/configs/study_a2_followup.toml` (about 1.5 minutes).
- Environment: as section 4.
- Outputs:
  - tables: `experiments/results/tables/study_a2_followup_{grid_a,new_grid,confirm_grid}.csv` and `study_a2_followup_summary.json`;
  - figures: `experiments/results/figures/study_a2_followup_*`.

| Statistic (informative points) | Confirmation grid (registered) | Grid A (post hoc) | New grid (post hoc) |
|---|---|---|---|
| Informative points | 1246 of 3710 | 1954 of 5880 | 1807 of 5472 |
| F1: points LBR solves that SN does not | 0 of 1238 | 0 of 1940 | 0 of 1797 |
| F2: SN e_norm median / p99 | 0.0 / 2.20 | 0.018 / 2.63 | 0.0 / 2.13 |
| F3: SN median e_norm near the money | 0.0 (n = 44) | 0.0 (n = 56) | 0.0 (n = 12) |
| F4: bad outcomes (Brent, SN) | 0, 0 | 0, 0 | 0, 0 |
| SN median iterations | 9 | 8 | 9 |
| SN median time per solve (reported, not tested) | 340 us | 244 us | 302 us |

At uninformative points priced above the intrinsic value, which previously gave s = 0, SN now converges at all of them:
- confirmation grid: 5 of 5, median relative error 9.6%;
- grid A: 8 of 8, 0.8%;
- new grid: 7 of 7, 1.7%.

LBR also converges at 19 of these 20 points.

### Follow-up verdicts
- F1: SUPPORTED. SN converges at every informative point where LBR does.
- F2: SUPPORTED. Median 0.0, p99 2.20, within the thresholds LBR met in Study A.
- F3: SUPPORTED.
- F4: SUPPORTED.

### Conclusion
With the follow-up, our safeguarded Newton matches Let's Be Rational: it solves the same prices, with p99 normalized error 2.1 to 2.6 against LBR's 2.0 to 2.7. It is still about 35 to 50 times slower per solve, the cost of the extra-digit evaluation (section 6). The README's two implied-volatility limitations from Study A are resolved; the speed cost replaces them.
