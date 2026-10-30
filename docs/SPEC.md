# Options Pricing Engine: Technical Specification

Version 1.0, 30 October 2026. This document is the standard every phase is judged against. Read the section for the current phase before planning. Section numbers are stable; refer to them in commits and research notes.

---

## 0. Mandate

Build a small, transparent derivatives-pricing library whose every number can be defended: model assumptions stated, numerical error quantified, results validated against published values and independent implementations, experiments reproducible.

The pricing kernels serve two pre-registered studies, which are the core deliverables:

- **Study A (§10):** accuracy, robustness, and cost of implied-volatility solvers (Brent, naive Newton, safeguarded Newton) against Jäckel's "Let's Be Rational", measured relative to the problem's condition number.
- **Study B (§11):** convergence order and error-versus-runtime of CRR, Leisen-Reimer, BBS, and BBSR trees for European and American options, scored against published benchmarks and a high-precision oracle.

Principles: correctness before speed; explicit assumptions before abstraction; independent validation before claims; measured performance before optimization.

## 1. Scope

**In scope:** European calls and puts (BSM closed form, Monte Carlo); American calls and puts (binomial trees); analytic and finite-difference Greeks; implied volatility for European options; Studies A and B; benchmarks.

**Out of scope:** live or historical market data, volatility-surface calibration to market quotes, stochastic or local volatility, jumps, discrete cash dividends, transaction costs, funding and collateral, trading connectivity.

## 2. Model contract and conventions

### 2.1 Dynamics
Under the risk-neutral measure, with constant r, q, sigma:

```
dS(t) = (r - q) S(t) dt + sigma S(t) dW(t)
```

A model price is the discounted risk-neutral expectation of the payoff, conditional on these assumptions. It is not a forecast, a recommendation, or evidence of mispricing.

### 2.2 Conventions

| Quantity | Convention |
|---|---|
| S, K | Strictly positive, currency units |
| T | Years, T >= 0. Date handling (if ever added) is isolated from kernels; the QuantLib oracle uses Actual365Fixed and T = days / 365 exactly |
| r, q | Continuously compounded, annualized decimals. Any finite real value accepted by BSM. Trees validate their own constraints (§4.1) |
| sigma | Annualized decimal, sigma >= 0. 0.20 means 20%. sigma > 5 raises `PricingInputError` unless `allow_extreme_vol=True` (guards against passing 20 for 20%) |
| cp | `OptionType.CALL` or `OptionType.PUT`. Internally theta_cp = +1 for calls, -1 for puts |
| Price | Same currency units as S and K, discounted to today |
| Delta, Gamma | Per 1.00 change in S |
| Vega | Per 1.00 change in sigma. Helper `per_vol_point(vega) = vega / 100` |
| Theta | -dV/dT, per year (calendar decay; negative for a typical long option). Helper `per_calendar_day(theta) = theta / 365` |
| Rho | Per 1.00 change in r. Helper `per_percentage_point(rho) = rho / 100` |
| Phi (dividend rho) | Per 1.00 change in q |

### 2.3 API contract
- BSM price and Greek functions accept Python floats or NumPy arrays and follow NumPy broadcasting. All-scalar inputs return `float`; any array input returns `np.ndarray` (float64) of the broadcast shape. Incompatible shapes raise NumPy's broadcasting `ValueError`, untouched.
- Trees, Monte Carlo, and IV solvers take scalars and return frozen dataclasses (`TreeResult`, `MCResult`, `IVResult`) carrying the value plus diagnostics.
- Type hints everywhere (`numpy.typing.NDArray[np.float64]`). `mypy --strict` on `src/`.
- Pure functions. Randomness only through an explicit, required seed. Phase 2 decision: Monte Carlo takes `seed` as an int or `numpy.random.SeedSequence` rather than a `Generator`, because §5.1 requires the result to record the seed that reproduces it and a Generator cannot report that. Child streams (main sample, pilot) are derived without mutating the caller's SeedSequence.

### 2.4 Exceptions (`src/pricing/errors.py`)
- `PricingInputError(ValueError)`: NaN or inf, S <= 0, K <= 0, T < 0, sigma < 0, unsupported contract.
- `UndefinedGreekError(ValueError)`: Greek requested where it does not exist (e.g. delta at T = 0 and S = K).
- `TreeParameterError(ValueError)`: invalid tree probability or grid (§4.1).
- `InfeasiblePriceError(ValueError)`: IV requested for a price outside no-arbitrage bounds (§6.2).
- `ConvergenceError(RuntimeError)`: raised by public IV convenience wrappers when the underlying solver returns `converged=False`.

## 3. Black-Scholes-Merton

### 3.1 Kernel: forward and total-volatility form

```
F  = S * exp((r - q) * T)        # forward
D  = exp(-r * T)                 # discount factor
s  = sigma * sqrt(T)             # total volatility
x  = ln(F / K)                   # log-moneyness
d1 = x / s + s / 2
d2 = d1 - s
```

Undiscounted Black value: `b(F, K, s, cp) = theta_cp * (F * N(theta_cp * d1) - K * N(theta_cp * d2))`. Discounted price `V = D * b`. This is the parametrization used by Jäckel, which makes Study A comparisons exact.

### 3.2 Prices (equivalent spot form, for documentation and tests)

```
C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
```

### 3.3 Analytic Greeks (phi = standard normal density)

| Greek | Call | Put |
|---|---|---|
| Delta | e^{-qT} N(d1) | -e^{-qT} N(-d1) |
| Gamma | e^{-qT} phi(d1) / (S sigma sqrt(T)) | same |
| Vega | S e^{-qT} phi(d1) sqrt(T) | same |
| Theta | -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1) | -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1) |
| Rho | K T e^{-rT} N(d2) | -K T e^{-rT} N(-d2) |
| Phi | -S T e^{-qT} N(d1) | S T e^{-qT} N(-d1) |

Every Greek is cross-checked against finite differences (§7) and QuantLib (§8). Sanity anchor: the Hull case (§9.1) gives call theta about -4.559 per year and vega about 8.81.

### 3.4 Boundary policy (part of the API)
- **T = 0:** price = intrinsic `max(theta_cp (S - K), 0)`. Delta = theta_cp * 1{theta_cp (S - K) > 0}; at S = K raise `UndefinedGreekError`. Gamma, vega, theta raise `UndefinedGreekError` at T = 0.
- **sigma = 0, T > 0:** deterministic forward. Price = `D * max(theta_cp (F - K), 0)`. Greeks follow the deterministic limit; where it is discontinuous (F = K) raise `UndefinedGreekError`.
- **Small total volatility:** no magic threshold. Tests verify continuity: for s in {1e-4, ..., 1e-12}, the price approaches the s = 0 limit without NaN or sign errors.
- **Extreme moneyness:** use `scipy.special.ndtr` and evaluate tails via `N(-x)`; never `1 - N(x)`. Deep OTM prices may underflow to 0 legitimately; that is documented, not hidden.
- **Invalid inputs:** raise `PricingInputError` naming the offending parameter.

### 3.5 Numerical-form rules
- Compute in forward form; convert to spot only for presentation.
- Avoid subtracting nearly equal quantities when a stable equivalent exists (compute puts directly, not via parity).
- Vectorize with NumPy; no Python loops over options in BSM code.

## 4. Binomial trees

All trees: N steps, dt = T / N, growth `g = exp((r - q) dt)`, per-step discount `exp(-r dt)`. Backward induction is vectorized over each time layer: O(N^2) time, O(N) memory.

### 4.1 Cox-Ross-Rubinstein (1979)

```
u = exp(sigma * sqrt(dt)),  d = 1 / u
p = (g - d) / (u - d)
```

**Validity:** 0 < p < 1 iff d < g < u, i.e. `|r - q| * sqrt(dt) < sigma`. If violated, raise `TreeParameterError` stating the minimum N: `N > (r - q)^2 T / sigma^2`. Never clip p. Write the derivation from d < g < u in the module docstring.

### 4.2 Leisen-Reimer (1996)
N must be odd (raise otherwise). With d1, d2 from §3.1 using the full T, and the Peizer-Pratt method-2 inversion

```
h(z)  = 0.5 + sign(z) * 0.5 * sqrt(1 - exp(-(z / (N + 1/3 + 0.1/(N + 1)))^2 * (N + 1/6)))
p     = h(d2),  p_bar = h(d1)
u     = g * p_bar / p
d     = (g - p * u) / (1 - p)
```

Numerically checked during spec review: on R1 the LR error falls about 4x per doubling of N (3.5e-6 at N = 101 to 5.7e-8 at N = 801), consistent with O(1/N^2). Still confirm against Leisen and Reimer (1996), Applied Mathematical Finance 3(4), and cite equation numbers in the docstring.

### 4.3 BBS and BBSR (Broadie and Detemple 1996)
- **BBS:** CRR tree in which the values at step N - 1 are replaced by the Black-Scholes European value over the final interval dt (for American options `max(exercise, BS_european(dt))`), then backward induction as usual.
- **BBSR:** two-point Richardson extrapolation `V = 2 * V_BBS(N) - V_BBS(N / 2)`, N even.

### 4.4 American exercise
At every node: `V = max(continuation, exercise)`, exercise = `theta_cp (S_node - K)`. Record the early-exercise boundary (critical S per time layer) as an optional diagnostic.

**American call, q = 0:** assert equality with the European call only when q <= 0 and q <= r (Healy 2021, arXiv:2109.15157, Proposition 1). For r < 0 early exercise can be optimal; tests must not assert equality there.

### 4.5 Richardson extrapolation (general)
For error `c N^{-p}`: `V_RE = (2^p V(2N) - V(N)) / (2^p - 1)`. Apply only where the error sequence is monotone in N (LR with odd N, BBS). Applying it to raw CRR's odd/even oscillation is a documented failure case demonstrated in Study B.

## 5. Monte Carlo (European)

### 5.1 Exact terminal simulation

```
S_T = S * exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z),   Z ~ N(0, 1)
Y   = e^{-rT} * payoff(S_T)
```

Return `MCResult`: estimate, n_paths, n_payoff_evals, sample sd, standard error, 95% normal-approximation CI, seed, method, config. Document that the normal-approximation CI can undercover for small n and skewed payoffs (deep OTM).

### 5.2 Antithetic variates
Pairs `A_j = (Y(Z_j) + Y(-Z_j)) / 2`, j = 1..n/2. Standard error from the sample sd of the n/2 pair averages. Never treat pair members as independent observations.

### 5.3 Control variate
Control `X = e^{-rT} S_T`, known mean `E[X] = S e^{-qT}`. Estimator `Y_cv = mean(Y) - beta (mean(X) - E[X])`, `beta* = Cov(Y, X) / Var(X)`. Default: estimate beta on an independent pilot of size m from a separately spawned seed; pilot cost counts in the budget. Same-sample beta is an option, documented as having O(1/n) bias.

### 5.4 Cost and variance reduction
Cost unit = payoff evaluations. `VRF = (Var_plain * cost_plain) / (Var_method * cost_method)` at matched budgets.

### 5.5 Coverage experiment (statistical, not a unit test)
R = 2000 replications via `np.random.SeedSequence(seed).spawn(R)`, n in {1e2, 1e3, 1e4, 1e5}, three moneyness levels. Report empirical coverage with binomial standard error `sqrt(0.95 * 0.05 / R) ≈ 0.0049`, empirical bias, RMSE. Verify SE scaling ≈ n^{-1/2} by regression on log n.

## 6. Implied volatility

### 6.1 Problem
Given discounted price V, solve for sigma >= 0 with `BSM(sigma) = V`. Work in Black form: `c = V / D`, solve for total volatility s, return `sigma = s / sqrt(T)`. Vega in s: `dc/ds = F phi(d1) = K phi(d2)`.

### 6.2 Feasibility (checked before any solver runs)
Intrinsic `iota = max(theta_cp (F - K), 0)`; upper bound `F` (call) or `K` (put). With `tol_b = 8 * eps * max(F, K)` (Phase 3 amendment: at the lower bound `tol_b` applies only when `iota > 0`; out of the money `iota` is exactly 0 and the lower-bound tolerance is 0, because the tolerance exists to absorb rounding in `F - K`, and an out-of-the-money price below `8 eps max(F, K)` such as 1e-20 has a well-conditioned volatility that the original rule would have returned as 0; see `tests/test_implied_vol.py`):
- `c < iota - tol_b`: raise `InfeasiblePriceError` (below intrinsic).
- `|c - iota| <= tol_b`: return s = 0, reason `AT_LOWER_BOUND`.
- `c >= upper - tol_b`: raise `InfeasiblePriceError` (sigma would be infinite).

### 6.3 Solvers (all return `IVResult`)
`IVResult` fields: sigma, total_vol, residual, iterations, converged, reason (enum: CONVERGED, AT_LOWER_BOUND, MAX_ITERATIONS, BRACKET_FAILURE, plus DIVERGED and ZERO_DERIVATIVE for naive Newton, added in Phase 3), method, optional path of iterates. Phase 3 also records function_evaluations, and sigma and total_vol are None unless converged.

1. **Brent (baseline):** bracket [0, s_hi], s_hi doubling from 1.0 until `c(s_hi) > c_target`, capped at s_max = 20 (else BRACKET_FAILURE). `scipy.optimize.brentq` with `xtol=1e-15`, `rtol=4*eps`, `maxiter=200`, `full_output=True`.
2. **Naive Newton:** start s0 = 0.2 sqrt(T), no safeguards, cap 100 iterations. Exists to be measured, not used in production paths.
3. **Safeguarded Newton:** Manaster-Koehler start `s0 = sqrt(2 |x|)`; if `|x| < 1e-8` use `s0 = 0.2 sqrt(T)`. Maintain a sign bracket. Take the Newton step only if it stays inside the bracket, `dc/ds > vega_min` (named constant, justified), and (Phase 3 amendment) it is less than half the previous step, the progress test of Numerical Recipes' rtsafe; otherwise bisect. Without the progress test, 30 of the 520 §6.4 round-trip cases hit the iteration cap: deep out of the money, Newton crawls inside the bracket because c / c' ~ s^3 / x^2. Converge when `|c(s) - c_target| <= 4 eps c_target` or bracket width `<= 4 eps s`.
4. **Oracle:** Jäckel's Let's Be Rational via `lets_be_rational` (MIT; Phase 3 chose it over the identical `py_lets_be_rational`). Expected call: `implied_volatility_from_a_transformed_rational_guess(price, F, K, T, q)` with undiscounted price and q = +1 call, -1 put. Confirm the signature and conventions in a test before relying on it.

### 6.4 Tests
Round trip price to IV to price over x in [-3, 3], T in {1/365, 7/365, 0.25, 1, 5}, sigma in {0.01, 0.2, 1.0, 3.0}; infeasible prices; AT_LOWER_BOUND; iteration-cap behavior; deep OTM low-vega cases.

## 7. Finite-difference Greeks
- Central differences on smooth interior points: `(V(x + h) - V(x - h)) / (2h)`; second derivative `(V(x + h) - 2V(x) + V(x - h)) / h^2`.
- Relative bumps: `h = eta * max(|x|, x_scale)`.
- Sweep eta in {1e-1, ..., 1e-12}. Plot |error vs analytic| against eta on log-log axes and explain the V-shape: truncation error O(h^2) on the right; rounding error O(eps / h) for first derivatives, O(eps / h^2) for second. Predicted optimal eta ≈ eps^{1/3} ≈ 6e-6 (first) and eps^{1/4} ≈ 1.2e-4 (second). Compare prediction with observation.
- One-sided differences near sigma = 0 or T = 0, labeled as such.
- Theta by FD in T: `theta_fd = -(V(T + h) - V(T - h)) / (2h)`, consistent with §2.2.

## 8. Independent oracles

### 8.1 QuantLib (`pip install QuantLib`)
- Fixed evaluation date; maturity = evaluation date + n days; `Actual365Fixed`; our T = n / 365 exactly.
- `FlatForward` curves for r and q with continuous compounding; `BlackConstantVol`; `BlackScholesMertonProcess`.
- European: `AnalyticEuropeanEngine`. Reconcile prices and all Greeks over a grid of at least 500 cases. Start at relative tolerance 1e-11 (both are double-precision closed forms); justify the final value from the observed error distribution and investigate outliers. Check QuantLib's Greek units before comparing.
- American: `QdFpAmericanEngine` (Andersen-Lake-Offengenden) high-precision scheme as the Study B oracle. Available in QuantLib 1.44 as `QdFpAmericanEngine(process, QdFpAmericanEngine.highPrecisionScheme())`, for puts and calls (Phase 2): it reproduces Ju-Zhong R4 to R6 within 6.4e-4 and agrees with BBSR at N = 16384 within 1.5e-6 (`tests/test_oracles.py`). Fallback if it ever disappears: BBSR at N = 2^15 cross-checked with LR at N = 32769, agreement < 1e-6.
- `BinomialVanillaEngine` ("crr", N) cross-checks our CRR at identical N. Phase 2 finding: the expectation of agreement near machine precision is refuted as written. QuantLib's CRR up-probability is the log-drift approximation 1/2 + (1/2)(r - q - sigma^2/2) sqrt(dt) / sigma, not §4.1's martingale probability, so the prices differ by O(1/N) (at least 2e-9 relative on the test grid). With QuantLib's probability substituted, our backward induction reproduces it within 8.1e-12 relative. `BinomialVanillaEngine` ("lr", N) uses §4.2's formulas and matches our LR within 8.4e-12.

### 8.2 vollib and lets_be_rational
Second implementation for BSM prices and the IV oracle (§6.3). Pin versions. `vollib` is the current name of `py_vollib`: since 1.0.12, `py_vollib` is a shim whose import raises `DeprecationWarning` (Phase 0 finding, 30 October 2026).

### 8.3 mpmath
Study A computes target prices at 50 significant digits before rounding to float64, separating pricing error from solver error.

## 9. Validation

### 9.1 Reference values (keep status current)

| ID | Case | Expected | Source | Status |
|---|---|---|---|---|
| R1 | European, S=42, K=40, r=0.10, q=0, sigma=0.20, T=0.5 | call 4.76, put 0.81 (2 dp); d1 = 0.7693, d2 = 0.6278 | Hull, Options, Futures, and Other Derivatives, Example 15.6 (record edition and page) | Printed 2 dp verified. Unrounded 4.759422 / 0.808599 confirmed by QuantLib 1.44 in Phase 1 (Actual360, 180 days, so T = 0.5 exactly; `tests/test_oracles.py`); our price equals QuantLib's to the last bit |
| R2 | European put, S=100, K=95, T=0.5, r=0.10, q=0.05, sigma=0.20 | 2.4648 | Haug (2007), p. 4, as quoted in the py_vollib doctest | Secondary source; reproduced to 4 dp by the §3.1 kernel and by QuantLib 1.44 (Phase 1) |
| R3 | European call, S=60, K=65, T=0.25, r=0.08, q=0, sigma=0.30 | 2.1334 | Haug (2007), pp. 2 to 3 | Reproduced to 4 dp by the §3.1 kernel and by QuantLib 1.44 (Phase 1); printed value not checked in Haug |
| R4 | American puts, S=40, r=0.0488, q=0. (K, sigma, T) to value: (35, 0.2, 0.0833) 0.006; (40, 0.2, 0.3333) 1.580; (40, 0.2, 0.5833) 1.990; (45, 0.2, 0.5833) 5.267; (45, 0.3, 0.3333) 5.706; (40, 0.4, 0.5833) 4.353; (45, 0.4, 0.5833) 7.383 | as listed | Ju and Zhong (1999), Exhibit 3 (TRUE = binomial N = 10,000, accurate to 3rd decimal) | Verified from primary PDF. Reproduced within 2e-3 by LR (N = 2001) and BBSR (N = 2048) in Phase 2 |
| R4b | American put, S=40, K=40, sigma=0.2, T=0.0833, r=0.0488, q=0 | 0.852 | Ju and Zhong (1999), Exhibit 3 | Reproduced numerically during spec review (LR tree, N = 2001: 0.85218) and in Phase 2 (LR 0.85218, BBSR N = 2048 0.85217); printed value not checked in the paper |
| R5 | American calls, K=100, T=0.5. (S, sigma, r, q) to value: (100, 0.2, 0.03, 0.07) 4.783; (110, 0.2, 0.03, 0.07) 11.098; (100, 0.4, 0.03, 0.07) 10.239; (100, 0.3, 0.00, 0.07) 7.035; (100, 0.3, 0.07, 0.03) 9.250 | as listed | Ju and Zhong (1999), Exhibit 4 | Verified from primary PDF. Reproduced within 2e-3 by LR and BBSR in Phase 2 |
| R6 | American puts, K=100, T=3, sigma=0.20, r=0.08, q=0.12; S = 80, 90, 100, 110, 120 | 25.658, 20.083, 15.498, 11.803, 8.886 | Ju and Zhong (1999), Exhibit 5 | Verified from primary PDF. Reproduced within 2e-3 by LR and BBSR in Phase 2 |

Tolerance for R4 to R6: `abs(err) < 2e-3` (published to 3 decimals and themselves binomial approximations). Tighter accuracy claims come from the QuantLib oracle, not these tables.

### 9.2 Required test categories

| Category | Content |
|---|---|
| Reference | §9.1 values, per status |
| Identities | Put-call parity over a grid spanning r, q (including negative), T, sigma, moneyness |
| Bounds | max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}; max(K e^{-rT} - S e^{-qT}, 0) <= P <= K e^{-rT} |
| Monotonicity | C increasing in S and sigma, decreasing in K; P mirror; maturity monotonicity only under stated assumptions |
| Property-based | Hypothesis strategies over valid domains for parity, bounds, monotonicity, Greek signs |
| Greeks | Analytic vs FD in interior; signs and units; boundary behavior |
| Cross-method | Trees and MC vs BSM; American >= European; American call = European call only under §4.4 conditions |
| Solver | §6.4 |
| Input validation | NaN, inf, S <= 0, K <= 0, T < 0, sigma < 0, bad shapes |
| Oracle | QuantLib, lets_be_rational and vollib reconciliation (`@pytest.mark.oracle`) |
| Statistical | MC coverage and SE scaling (`@pytest.mark.statistical`, fixed seeds) |

### 9.3 Tolerance policy
Every tolerance is a named constant with a justification. Use scale-aware bounds; e.g. parity `|C - P - (S e^{-qT} - K e^{-rT})| <= 64 * eps * max(S, K)` as a starting point, tightened or justified from observed error distributions. Report absolute and relative error. A failing test is diagnosed (formula, units, method, expected value) before any tolerance changes.

## 10. Study A: implied-volatility solver accuracy

**Question.** Across the (log-moneyness, total-volatility) plane, how do Brent, naive Newton, and safeguarded Newton compare with Let's Be Rational in accuracy, robustness, and cost, when accuracy is measured against what the problem's conditioning allows?

**Design.**
- Grid: x = ln(F/K) from -6 to 6 in steps of 0.25 (49 points); s = sigma sqrt(T) on 60 log-spaced points from 1e-4 to 4. Calls and puts.
- Ground truth: choose s_true; compute the target price with mpmath at 50 digits; round once to float64.
- Metrics per solver and grid point:
  - `e_ulp = |s_hat - s_true| / np.spacing(s_true)`
  - Relative condition number `kappa = |c / (s * dc/ds)|`; attainable relative error ≈ kappa * eps. Normalized error `e_norm = (|s_hat - s_true| / s_true) / (kappa * eps)`. e_norm = O(1) means the solver is as accurate as the problem permits.
  - Iterations; median wall time per solve over 50 repeats after warm-up; failure class.
- Points where c underflows or c < eps * max(F, K) are classified ill-posed and reported separately.

**Hypotheses (write in `docs/notes/study_a.md` and commit before running).**
- A1: Naive Newton fails (diverges, cycles, or hits the cap) on a substantial region at small s and large |x|.
- A2: Brent converges wherever a bracket exists but needs several times more iterations than LBR, and its accuracy is limited by its stopping tolerance rather than by kappa.
- A3: LBR achieves e_norm = O(1) across the well-posed region.
- A4: In the ill-posed region, error is set by kappa and representability, not by method.

**Deliverables.** Heatmaps of log10(e_ulp) and log10(e_norm) per solver; failure maps; iteration histograms; timing table; verdict on each hypothesis.

## 11. Study B: American-option convergence

**Question.** What convergence order and error-versus-runtime trade-off do CRR, LR, BBS, and BBSR achieve for European and American options, and do they reproduce published benchmarks?

**Design.**
- Cases: R1 (European); R4, R5, R6 (American).
- Steps: CRR every N from 10 to 2000 (exposes odd/even oscillation); LR odd N from 11 to 2001; BBS and BBSR N = 2^k, k = 4..13.
- References: BSM closed form (European); QuantLib QdFp high precision (American) or the §8.1 fallback. First confirm the reference agrees with Ju-Zhong within 2e-3.
- Observed order p: least-squares fit of log|e_N| on log N for N >= N_min, with standard error. For CRR, fit odd and even N separately and fit the envelope.
- Runtime: median of repeated timings per (method, N).

**Hypotheses (write in `docs/notes/study_b.md` and commit before running).**
- B1: CRR European error is O(1/N) with odd/even oscillation; naive Richardson on CRR does not reliably improve it.
- B2: LR European error is O(1/N^2) for odd N.
- B3: For American puts, LR's observed order falls toward 1 because the early-exercise boundary is not aligned with nodes.
- B4: BBSR gives the lowest error at fixed runtime among tested methods for American puts.

**Deliverables.** Log-log error plots; order table with standard errors; error-versus-runtime Pareto plot; Ju-Zhong reproduction table; verdict on each hypothesis.

## 12. Benchmarks
- Options per second and latency per option and per batch (1, 1e3, 1e6) for vectorized BSM; runtime per N for trees; cost per path for MC.
- Warm up; repeat; report median and interquartile range, never the fastest run.
- Record hardware, OS, Python, NumPy, SciPy, BLAS and thread settings, commit hash.
- C++ (pybind11) only after profiling shows a bottleneck, with output parity against the Python reference at stated tolerances.

## 13. Engineering

### 13.1 Repository layout

```
options-pricing-engine/
  README.md  pyproject.toml  .gitignore
  .github/workflows/ci.yml
  src/pricing/   __init__.py contracts.py errors.py black_scholes.py binomial.py
                 monte_carlo.py implied_vol.py finite_diff.py oracles.py
  tests/         one module per source module, plus test_properties.py,
                 test_reference_values.py, test_oracles.py, conftest.py
  experiments/   study_a_iv.py study_b_trees.py mc_coverage.py greek_bumps.py
                 benchmarks.py configs/*.toml results/{tables,figures}/
  docs/          SPEC.md notes/TEMPLATE.md notes/study_a.md notes/study_b.md
```

### 13.2 Tooling (all free)
Python 3.11+; NumPy, SciPy, Matplotlib, mpmath; pytest, Hypothesis, pytest-benchmark; Ruff; mypy (strict on `src/`); QuantLib, lets_be_rational, vollib for oracles. Constrain versions in `pyproject.toml` (e.g. `numpy>=2.0,<3`); record exact versions in every experiment output.

### 13.3 CI
GitHub Actions on a public repository (free for standard hosted runners). Push and pull request: Ruff, mypy, fast tests on Python 3.11 and 3.12. Manual `workflow_dispatch`: full suite including slow, statistical, and oracle tests.

### 13.4 Local post-edit check
In Phase 0, set up a local editor check that runs `ruff check` on each changed Python file and the fast test subset after every edit, and reports any failure at once. It is personal tooling and stays outside the repository.

## 14. Phases and exit gates

| Phase | Weeks | Read | Required deliverables | Exit gate |
|---|---|---|---|---|
| 0 Environment | 1 | §2.3, §9.2, §13 | Public repo, venv, pyproject, Ruff, mypy, pytest markers, CI, hook, README skeleton | Fresh clone reproduces all checks from README commands; hook demonstrated; no machine-specific paths |
| 1 BSM reference engine | 2 to 4 | §2, §3, §7 (as test tool), §8.1, §9 | §3 complete; errors.py; R1 to R3 per status; identities, bounds, monotonicity, property tests; QuantLib European reconciliation | All required tests pass; reconciliation report committed |
| 2 Numerical methods | 5 to 8 | §4, §5, §8.1, §9.1 R4 to R6 | CRR, LR, BBS, BBSR, American exercise; MC with antithetic and control variates; coverage experiment; Ju-Zhong reproduction | Cross-method tests pass; coverage note committed; R4 to R6 reproduced within tolerance |
| 3 IV and FD Greeks | 9 to 10 | §6, §7, §8.2 | Solvers with diagnostics; LBR oracle wired; bump study | §6.4 tests pass; bump-study note explains observed vs predicted optimal eta |
| 4 Study A | 11 to 12 | §8.3, §10 | §10 complete | Hypotheses committed before results; note with figures and verdicts |
| 5 Study B | 13 to 14 | §11 | §11 complete | Same standard as Study A |
| 6 Performance and publication | 15 to 16 | §12, §15 | Benchmarks; README results section; v1.0 tag | Benchmarks reproducible from one command; README states scope, conventions, results, limitations |

**Stretch (only after v1.0):** Sobol randomized QMC with replicate error bars (`scipy.stats.qmc.Sobol`, power-of-2 sample sizes); pathwise and likelihood-ratio Monte Carlo Greeks; autodiff Greeks (JAX) compared with analytic; Tian tree with truncation and smoothing (Joshi 2009) added to Study B; pybind11 C++ BSM kernel.

## 15. Non-claims
- Constant-volatility BSM is a baseline model, not a description of listed-options markets.
- Synthetic IV round trips validate solvers; they say nothing about market calibration.
- Monte Carlo intervals quantify sampling error under the model, not model risk.
- One machine's benchmarks are not evidence of production performance.
- Nothing in this project is a trading strategy, alpha, or investment recommendation.

## 16. References
- Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy 81(3).
- Merton, R. C. (1973). Theory of Rational Option Pricing. Bell Journal of Economics and Management Science 4(1).
- Cox, J. C., Ross, S. A. and Rubinstein, M. (1979). Option Pricing: A Simplified Approach. Journal of Financial Economics 7(3).
- Leisen, D. and Reimer, M. (1996). Binomial Models for Option Valuation: Examining and Improving Convergence. Applied Mathematical Finance 3(4).
- Broadie, M. and Detemple, J. (1996). American Option Valuation: New Bounds, Approximations, and a Comparison of Existing Methods. Review of Financial Studies 9(4).
- Ju, N. and Zhong, R. (1999). An Approximate Formula for Pricing American Options. Journal of Derivatives 7(2).
- Andersen, L., Lake, M. and Offengenden, D. (2016). High-Performance American Option Pricing. Journal of Computational Finance 20(1). SSRN 2547027.
- Jäckel, P. (2015). Let's Be Rational. Wilmott 2015(75), 40 to 53. DOI 10.1002/wilm.10395.
- Manaster, S. and Koehler, G. (1982). The Calculation of Implied Variances from the Black-Scholes Model: A Note. Journal of Finance 37(1).
- Joshi, M. S. (2009). The Convergence of Binomial Trees for Pricing the American Put. Journal of Risk 11.
- Healy, J. (2021). Pricing American Options under Negative Rates. arXiv:2109.15157.
- Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering. Springer.
- Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms, 2nd ed. SIAM.
- Hull, J. C. Options, Futures, and Other Derivatives (record the edition used).
- Haug, E. G. (2007). The Complete Guide to Option Pricing Formulas, 2nd ed. McGraw-Hill.

Record exact page or equation numbers in docstrings and notes for every formula taken from a source. Verify bibliographic details against the publisher before citing them in the README.

## 17. Open verification items
- R2, R3 (Haug) and R4b (Ju-Zhong 0.852): reproduced numerically during spec review; confirm printed values in the primary sources if copies are available.
- Hull edition and page for R1.
- Resolved in Phase 2: `QdFpAmericanEngine` scheme API (`highPrecisionScheme()`), checked against Ju-Zhong and BBSR (see §8.1).
- Resolved in Phase 3, corrected in Phase 4: Let's Be Rational package and convention. The `lets-be-rational` 1.1.2 wheel (the one vollib depends on) ships the `py_lets_be_rational` implementation and a `lets_be_rational` package that re-exports it, so there is one implementation, not two diverged copies as Phase 0 recorded. It is installed through `lets-be-rational`; the separate `py_lets_be_rational` distribution is not needed. It makes 2 Householder iterations (`implied_volatility_maximum_iterations`). Signature `(price, F, K, T, q)` with the undiscounted price and q = +1 call, -1 put is pinned by `tests/test_oracles.py`.
- Leisen-Reimer formulas (§4.2): numerically consistent with O(1/N^2); confirm equation numbers in the original paper.
- Jäckel (2015) test cases, if obtainable, as extra Study A anchors.
- Citations added in Phase 1, numbering not checked against the books: Press et al., Numerical Recipes, 3rd ed., §5.7 (exact finite-difference steps) and Higham (2002), Theorem 2.5 (Sterbenz's lemma) in `pricing/finite_diff.py`; Hull's chapter on the Greek letters (edition, equation numbers) in `pricing/black_scholes.py`. Added in Phase 2: Glasserman (2004) sections 1.1, 4.1 and 4.2 in `pricing/monte_carlo.py`. Added in Phase 3: Press et al., Numerical Recipes, 3rd ed., section 9.4 (rtsafe progress test) and Manaster and Koehler (1982) (starting value) in `pricing/implied_vol.py`.
