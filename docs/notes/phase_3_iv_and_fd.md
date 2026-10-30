# Phase 3: implied-volatility solvers, Let's Be Rational oracle, finite-difference bumps

Status: COMPLETE (tag `phase-3`). This is a phase-gate record, not a study. The bump study, which was registered, has its own note: `phase_3_bump_study.md`.

## 1. Question
Do the implied-volatility solvers recover volatility as accurately as the price's own rounding allows? Do they report every failure instead of returning an unconverged number? Is the Let's Be Rational oracle wired and pinned? Does the bump study explain the observed optimal bump?

## 2. Claims (exit gate, SPEC 14)
- G1: the SPEC 6.4 tests pass: round trips, infeasible prices, AT_LOWER_BOUND, the iteration cap, and deep out-of-the-money low-vega cases.
- G2: the bump-study note explains observed versus predicted optimal eta.

## 3. Design
- Code:
  - `src/pricing/implied_vol.py`: feasibility; Brent, naive Newton and safeguarded Newton solvers; `implied_volatility` (raises `ConvergenceError`); `implied_volatility_result` and `solve_black` (full diagnostics);
  - `src/pricing/finite_diff.py`: one-sided differences;
  - `src/pricing/oracles.py`: `lbr_implied_volatility`, `vollib_price`;
  - `experiments/greek_bumps.py`.
- Tests: 348 in total, 36 new:
  - `test_implied_vol.py`: 20;
  - `test_finite_diff.py`: +8;
  - `test_oracles.py`: +4 (Let's Be Rational convention, infeasible prices, round-trip agreement, vollib);
  - `test_greek_bumps_helpers.py`: 4.

## 4. Results
- Round trips over the SPEC 6.4 grid of 520 cases:
  - safeguarded Newton and Brent: 306 converged, 214 at the lower bound, none failed;
  - safeguarded Newton: median 8 iterations (9 evaluations), at most 58; Brent: median 11 (15 evaluations);
  - accuracy: worst error 3.3 x eps x scale / (dc/ds), the price's own rounding propagated through the slope, plus Brent's stopping tolerance for Brent;
  - naive Newton: 104 converged, 182 zero derivative, 16 diverged, 4 hit the cap. Every failure is reported, none returns a value.
- Lower-bound cases: all 214 are prices whose time value is below rounding or that underflow to 0; the test checks each one.
- Let's Be Rational: `lets_be_rational` and `py_lets_be_rational` gave identical results on 326 points; the first is kept. Signature `(price, F, K, T, q)` with the undiscounted price, pinned by a test (the discounted price gives 0.1726 instead of 0.20 on R1). It meets the same accuracy bound on every well-posed round-trip point.
- vollib: within the doubled Phase 1 rounding budget of our BSM on 360 cases (worst relative difference 5e-12, at a 1e-29 price).
- Bump study: all 8 registered checks supported. Observed optima match refined predictions within factors 0.51 to 2.11 (see its note).

Diagnoses and decisions:
1. SPEC 6.2 amended. The lower-bound tolerance 8 eps max(F, K) now applies only when the intrinsic value is positive. Out of the money the intrinsic value is exactly 0, and the original rule turned prices below about 2e-13 (on F = 100), whose volatility is well conditioned, into sigma = 0.
2. SPEC 6.3 amended. Safeguarded Newton bisects when a Newton step is not less than half the previous one (Numerical Recipes' rtsafe). Without that, 30 of the 520 round trips hit the cap. Deep out of the money from the Manaster-Koehler start, which is the inflection point of c(s), Newton crawls: the step c / c' ~ s^3 / x^2.
3. Added termination reasons DIVERGED and ZERO_DERIVATIVE for naive Newton; sigma is None unless converged.
4. The round-trip tolerance comes from the price's rounding, not from SPEC 10's condition number kappa alone. At the money with small s, kappa = 1, yet the price is a difference of two terms of about 50 that nearly cancel to 0.055, so the target is accurate only to about 900 eps relative.
5. Removing `py_lets_be_rational` from a working environment deleted `lets_be_rational`'s files, apparently because the two wheels share files; a forced reinstall fixed it. Fresh installs, as in CI, are unaffected.

## 5. Verdicts
- G1: SUPPORTED.
- G2: SUPPORTED.

## 6. Interpretation
- An implied-volatility solver's accuracy is bounded by the price's accuracy divided by dc/ds. Two things shape that bound: conditioning (kappa) and how accurately the price itself was computed. Study A measures the first; this phase showed the second matters as much near the money at small s.
- Robustness is a property of the safeguards, not of Newton. The same Newton step that converges in 8 iterations near the money crawls for hundreds deep out of the money. A bracket plus a progress test turns it into a method that always converges, and Brent shows the price paid: about 1.7 times the evaluations.
- Bump sizes should come from the problem's scale. The universal eps^(1/3) rule is within a factor of 4 here; the refined rule is within one grid step.

## 7. Limitations and threats to validity
- The solvers are scalar and pure Python. Speed is Study A's subject, not this phase's.
- The round-trip grid follows SPEC 6.4. Study A's grid is wider (|x| up to 6, s down to 1e-4) and uses mpmath targets.
- Citations whose numbering is unverified are listed in SPEC 17.

## 8. Next questions
- Study A (Phase 4): error, conditioning and cost across the (x, s) plane, against Let's Be Rational, with 50-digit targets.
