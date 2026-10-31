# Phase 3: finite-difference bump study

Status: COMPLETE. Sections 1 to 3 registered at commit `64b9131`, before the experiment code existed. Results from commit `214cd6c`.

## 1. Question
How does the error of a finite-difference Greek depend on the bump size? Does the optimal bump match the theoretical prediction of SPEC 7: eta about eps^(1/3) = 6.1e-6 for first derivatives and eps^(1/4) = 1.2e-4 for second derivatives? A desk that computes Greeks by bumping a pricer has to choose this bump. Too large a bump gives truncation error, too small a bump amplifies rounding. This study measures where the trade-off sits.

## 2. Hypotheses
Error model for a central difference with bump h on a function computed with relative rounding error about eps:
- first derivative: E(h) ~ (h^2 / 6) |f'''| + eps |f| / h;
- second derivative: E(h) ~ (h^2 / 12) |f''''| + 4 eps |f| / h^2.

In log-log axes this traces a V. The right branch (truncation) has slope +2. The left branch (rounding) has slope -1 for first derivatives and -2 for second derivatives. The minimum lies near h ~ eps^(1/3) and eps^(1/4) on the variable's own scale.

- D1, V-shape slopes, from least-squares fits of log10(relative error) on log10(eta):
  - right branch over eta in {1e-3, 10^-2.5, 1e-2}: slope in [1.7, 2.3] for delta, gamma, vega, theta and rho;
  - left branch over eta in {1e-12, 10^-11.5, 1e-11, 10^-10.5, 1e-10}: slope in [-1.5, -0.5] for the first derivatives and [-2.5, -1.5] for gamma.
- D2, optimum location: the eta with the smallest error lies within a factor of 10 of the SPEC prediction, i.e. in [6.1e-7, 6.1e-5] for delta, vega, theta and rho, and in [1.2e-5, 1.2e-3] for gamma.
- D3, best attainable error: the minimum relative error is at most 1e-8 for each first derivative and at most 1e-6 for gamma. Theory gives about eps^(2/3) = 3.7e-11 and eps^(1/2) = 1.5e-8, times problem-dependent constants.
- D4, one-sided differences near T = 0: for theta at T = 1 day, where a central difference is unavailable once h >= T, the right-branch slope over eta in {1e-5, 10^-4.5, 1e-4, 10^-3.5} lies in [0.7, 1.3] for the forward first-order difference and [1.7, 2.3] for the three-point forward difference (truncation orders 1 and 2).

## 3. Design
- Contract: European call, K = 100, T = 0.5, r = 0.05, q = 0.02, sigma = 0.25; main case S = 100.
- Greeks and bumped variable:
  - delta: central first difference in S;
  - gamma: central second difference in S;
  - vega: central first difference in sigma;
  - theta: minus the central first difference in T;
  - rho: central first difference in r.
- Bumps: h = eta x max(|x|, 1), with x the bumped variable's value (SPEC 7, x_scale = 1). eta runs over 23 values, 10^-1, 10^-1.5, ..., 10^-12.
- Reference: the analytic Greeks of `pricing.black_scholes`, verified in Phase 1 against mpmath, finite differences and QuantLib to about 1e-13 relative or better here.
- Metric: relative error |FD - analytic| / |analytic|. The optimum is the grid eta with the smallest error.
- D4 case: T = 1/365, theta by `forward_first` and `forward_first_order2` (labelled one-sided), and by the central difference where T - h > 0.
- Figures (reporting only, not hypotheses):
  - the V-shape for all five Greeks, with predicted optima marked;
  - for delta and gamma, a 3D surface of log10(relative error) over (log10 eta, S) for S = 70 to 130 in steps of 5, with the same data as a 2D heatmap;
  - the one-sided comparison at T = 1 day.
- No randomness. Config: `experiments/configs/greek_bumps.toml`. Outputs: tables and figures under `experiments/results/`, with commit, dirty flag, library versions and platform.

## 4. Results
- Code: commit `214cd6c`, clean tree. Command: `python -m experiments.greek_bumps --config experiments/configs/greek_bumps.toml` (under 1 s).
- Environment: macOS 26.6 arm64, Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1, matplotlib 3.11.2.
- Outputs: `experiments/results/tables/greek_bumps.csv`, `greek_bumps_summary.json`. Figures in `experiments/results/figures/`: `greek_bumps_vshape.png`, `greek_bumps_surface_3d.png`, `greek_bumps_surface_heatmap.png`, `greek_bumps_one_sided.png`.
- History: the script was committed and run at `a15a0ba`. Reviewing the figures found overlapping labels, which were fixed in `214cd6c`. The numbers are identical between the two runs.

| Greek | Right slope | Left slope | Optimal eta (observed) | SPEC prediction | Best relative error |
|---|---|---|---|---|---|
| delta | +2.00 | -0.77 | 3.2e-6 | 6.1e-6 | 9.2e-13 |
| gamma | +2.00 | -2.13 | 3.2e-5 | 1.2e-4 | 1.5e-8 |
| vega | +2.00 | -0.79 | 1.0e-5 | 6.1e-6 | 2.9e-13 |
| theta | +2.00 | -1.27 | 3.2e-6 | 6.1e-6 | 7.3e-11 |
| rho | +2.00 | -1.20 | 1.0e-5 | 6.1e-6 | 1.8e-11 |

One-sided theta at T = 1 day: right-branch slope +0.98 for the forward first-order difference, +1.95 for the three-point forward difference.

Post-hoc analysis of the optimum (not pre-registered). The rule of thumb eta ~ eps^(1/3) assumes |f| and |f'''| are of order 1 on the variable's scale. The optimum of the error model itself is h* = (3 eps |V| / |V'''|)^(1/3) for first derivatives and (48 eps |V| / |V''''|)^(1/4) for the second. Evaluating V's higher derivatives with mpmath at 50 digits (V = 7.683):

| Greek | Refined eta* | Observed | Observed / refined |
|---|---|---|---|
| delta | 2.3e-6 | 3.2e-6 | 1.39 |
| gamma | 6.1e-5 | 3.2e-5 | 0.51 |
| vega | 7.3e-6 | 1.0e-5 | 1.36 |
| theta | 6.2e-6 | 3.2e-6 | 0.51 |
| rho | 4.7e-6 | 1.0e-5 | 2.11 |

## 5. Verdicts
- D1: SUPPORTED. Right slopes +2.00 for all five Greeks. Left slopes -0.77 to -1.27 for first derivatives (band [-1.5, -0.5]) and -2.13 for gamma (band [-2.5, -1.5]).
- D2: SUPPORTED. Observed optima within factors 0.26 to 1.65 of SPEC's prediction.
- D3: SUPPORTED. Best errors 2.9e-13 to 7.3e-11 for first derivatives, 1.5e-8 for gamma.
- D4: SUPPORTED. Forward first order +0.98, three-point +1.95.

## 6. Interpretation
**Observed versus predicted optimal eta.** The SPEC rule of thumb is right to within a factor of 4 and the refined prediction to within a factor of 2.1. Both residual gaps are within one grid step (a factor of 3.16), the resolution of the measurement. The rule of thumb drifts because the optimum depends on the ratio |V| / |V'''|, which is not 1. For gamma (refined 6.1e-5 vs 1.2e-4), the fourth derivative in S is large relative to the price at the money, so a smaller bump is better than eps^(1/4) suggests. The practical lesson for a desk: scale the bump to the problem, not to a universal eta, and a factor of 2 either way costs little because the V is flat-bottomed in log terms.

**The rounding branch is shallower and lower than the model predicts.** The left slopes run -0.77 to -1.27, and delta's best error, 9.2e-13, beats the model's eps^(2/3) scale. The model assumes the rounding errors in V(x + h) and V(x - h) are independent, each of size eps |V|. For a smooth function evaluated at nearby points they are partly correlated and partly cancel in the difference. The left branch is therefore noisy, and only its trend (about -1, about -2 for gamma) is meaningful.

**Gamma is the expensive one.** A second difference divides rounding noise by h^2, so its best error, 1.5e-8, is about four orders of magnitude worse than delta's. This is the same effect that made Phase 1 check gamma by differencing delta instead (`docs/notes/phase_1_bsm.md`).

**Near expiry, one-sided differences pay their order.** The first-order forward difference's error falls only linearly with h, and its best error is about 1.6e-7 at eta = 1e-9. The three-point form recovers second order, with a best error of about 1.2e-9 at eta = 1e-7. Once h > T no central difference exists at all, which is the case SPEC 7's one-sided rule is for.

The 3D surfaces show that the valley's position barely moves across S from 70 to 130. Its floor behaves differently for the two Greeks, from the table rather than read off the 3D view:
- Delta's best error rises away from the money, from 9e-13 at S = 100 to 2e-10 at S = 70 and 5e-11 at S = 130.
- Gamma's best error varies irregularly between 5e-10 and 7e-8 with no clear trend in moneyness, because its floor sits in the noisy rounding regime.

## 7. Limitations and threats to validity
- One contract (a call at T = 0.5, sigma = 0.25); bumps relative with x_scale = 1. Other scalings move the optimum.
- The optimum is located on a half-decade grid, so its resolution is a factor of 3.16.
- The left-branch fits sit in a noisy regime; their bands were wide by design.
- Analytic Greeks serve as truth; Phase 1 verified them to about 1e-13, far below the errors measured here except at delta's and vega's minima (2.9e-13 to 9.2e-13), where the reference error is a non-negligible share.

## 8. Next questions
- A complex-step derivative, (Im f(x + ih)) / h, has no subtractive cancellation and would remove the rounding branch for first derivatives. It needs a complex-valued pricer, which is outside the current scope.
- An adaptive bump using estimated |V'''| (the refined rule above) in production Greek code.
