"""Implied volatility of European options (SPEC 6).

Problem (SPEC 6.1). Given a discounted price V, find sigma >= 0 with BSM(sigma) = V. In Black form
the undiscounted price c = V / D depends on sigma and T only through the total volatility
s = sigma sqrt(T):
    c(s) = theta (F N(theta d1) - K N(theta d2)),  d1 = x / s + s / 2,  d2 = d1 - s,  x = ln(F / K).
c increases strictly in s, from the intrinsic value iota = max(theta (F - K), 0) at s = 0 to the
upper bound F (call) or K (put) as s grows without limit. So a solution exists exactly when
iota < c < upper, and it is unique. Its derivative is dc/ds = F phi(d1) = K phi(d2) (SPEC 6.1).
The solvers work in s and return sigma = s / sqrt(T).

Feasibility (SPEC 6.2, as amended in Phase 3 and Study A2). With tol = 8 eps max(F, K):
- c below the intrinsic value (by more than tol when iota > 0): InfeasiblePriceError.
- c >= upper - tol: InfeasiblePriceError (s would be infinite).
- c at or below iota (within tol): s = 0, reason AT_LOWER_BOUND. Any positive time value
  c - iota is solved (Study A2 follow-up): even a time value of only a few ulps of c can pin s,
  because deep in the money it is extremely sensitive to s; how accurately is set by the problem's
  conditioning, not by the price's rounding. A 2 eps max(F, K) cutoff still dropped 20 such
  prices that Let's Be Rational solved to within 4% (Study A2, H2). Out of the money iota is
  exactly 0, so a deep out-of-the-money price such as 1e-20 is solved too.

In the money, the solvers work on the out-of-the-money counterpart (Study A2): by put-call parity
the in-the-money price minus iota is exactly the other option's price, with the same volatility.
Solving the in-the-money problem directly compares c(s) with c where both are close to iota, so
the time value that determines s is lost in the rounding of iota.

Price evaluation. Near the money at small s, c(s) = theta (F N(theta d1) - K N(theta d2)) subtracts
two nearly equal numbers. black_price detects this (CANCELLATION_LIMIT) and recomputes c(s) with
mpmath at enough digits, so every solver sees c(s) to a few eps relative (Study A found the
safeguarded Newton's median normalized error near the money at small s to be 683 without this).

Solvers (SPEC 6.3). Each returns an IVResult; sigma is None unless the solver converged.
1. Brent: bracket [0, s_hi], s_hi = 1, 2, 4, ... until c(s_hi) > c, capped at 20 (else
   BRACKET_FAILURE); then scipy.optimize.brentq with xtol = 1e-15, rtol = 4 eps, maxiter = 200.
   Its accuracy is set by those stopping tolerances, not by the problem's conditioning.
2. Naive Newton: s0 = 0.2 sqrt(T), s <- s - (c(s) - c) / c'(s), at most 100 iterations, no
   safeguards. It exists to be measured. It stops, rather than continuing, if an iterate leaves
   s > 0 or becomes non-finite (DIVERGED) or if c'(s) evaluates to 0 (ZERO_DERIVATIVE); these
   reasons are a Phase 3 addition to SPEC 6.3's list.
3. Safeguarded Newton: Manaster-Koehler start s0 = sqrt(2 |x|) (Manaster and Koehler 1982),
   0.2 sqrt(T) if |x| < 1e-8. It keeps a bracket [lo, hi] with c(lo) < c < c(hi), starting from
   [0, infinity). A Newton step is taken only if c'(s) > VEGA_MIN, the step lands strictly inside
   the bracket, and it is less than half the previous step; otherwise it bisects (or doubles s
   while hi is still infinite). The last condition is the progress test of Numerical Recipes'
   rtsafe (Press et al., 3rd ed., section 9.4), added in Phase 3. Without it, Newton crawls deep
   out of the money: there c(s) ~ exp(-x^2 / (2 s^2)), so the step c / c' ~ s^3 / x^2 is tiny next
   to the distance left, every step stays inside the bracket, and 30 of SPEC 6.4's 520 round-trip
   cases hit the iteration cap. Bisection halves the bracket, so the method always converges once
   the bracket is finite.
Convergence for both Newton methods: |c(s) - c| <= 4 eps c, or the step or bracket width is at
most 4 eps s.

The Let's Be Rational oracle (Jaeckel 2015) lives in pricing.oracles.
"""

from __future__ import annotations

import enum
import math
import sys
from dataclasses import dataclass

import mpmath
from scipy.optimize import brentq
from scipy.special import ndtr

from pricing.contracts import OptionType, option_sign, validate_scalar_inputs
from pricing.errors import ConvergenceError, InfeasiblePriceError, PricingInputError

EPS = sys.float_info.epsilon
BOUND_TOL_FACTOR = 8.0  # SPEC 6.2: tol = 8 eps max(F, K), for infeasibility
# c(s) = theta (a - b) with a = F N(theta d1), b = K N(theta d2) (SPEC 6.1). When max(a, b) is
# more than this many times c, the subtraction has lost log10(max(a, b) / c) digits; c is then
# recomputed with mpmath with that many extra digits (plus a guard) on the exact inputs. A factor
# of 4 bounds the double path's relative error at about 4 eps x the rounding of a and b.
CANCELLATION_LIMIT = 4.0
CANCELLATION_GUARD_DIGITS = 10  # beyond the 17 a double needs and the digits cancelled
MP_MAX_DIGITS = 400  # far past any cancellation a double can express (c >= 5e-324)
CONVERGENCE_FACTOR = 4.0  # SPEC 6.3: 4 eps relative residual, step or bracket width
S_MAX = 20.0  # SPEC 6.3: largest total volatility Brent's bracket search tries
BRENT_XTOL = 1e-15  # SPEC 6.3
BRENT_MAXITER = 200  # SPEC 6.3
NEWTON_MAXITER = 100  # SPEC 6.3 (naive); same cap for the safeguarded method
SMALL_LOG_MONEYNESS = 1e-8  # SPEC 6.3: below this the Manaster-Koehler start sqrt(2|x|) is ~0
DEFAULT_TOTAL_VOL_FACTOR = 0.2  # SPEC 6.3: start at sigma = 0.2, i.e. s0 = 0.2 sqrt(T)
# Newton divides by c'(s). Below the smallest normal double, c'(s) is subnormal or zero and has
# lost relative precision, so the quotient is meaningless; bisect instead.
VEGA_MIN = sys.float_info.min


class IVMethod(enum.Enum):
    BRENT = "brent"
    NAIVE_NEWTON = "naive_newton"
    SAFEGUARDED_NEWTON = "safeguarded_newton"


class IVReason(enum.Enum):
    CONVERGED = "converged"
    AT_LOWER_BOUND = "at_lower_bound"
    MAX_ITERATIONS = "max_iterations"
    BRACKET_FAILURE = "bracket_failure"
    DIVERGED = "diverged"  # naive Newton: iterate left s > 0 or became non-finite
    ZERO_DERIVATIVE = "zero_derivative"  # naive Newton: c'(s) evaluated to 0


@dataclass(frozen=True)
class IVResult:
    """An implied-volatility solve with its diagnostics (SPEC 6.3). sigma and total_vol are None
    unless `converged`."""

    sigma: float | None
    total_vol: float | None
    residual: float  # c(s) - c at the last evaluated iterate, undiscounted price units
    iterations: int
    function_evaluations: int
    converged: bool
    reason: IVReason
    method: IVMethod
    path: tuple[float, ...] | None = None  # total-volatility iterates, when requested


@dataclass(frozen=True)
class _Problem:
    """Black-form problem: find s with c(s) = target (SPEC 6.1)."""

    theta: int
    F: float
    K: float
    x: float  # ln(F / K)
    T: float
    target: float  # undiscounted price c
    lower: float  # intrinsic value, c(0)


def implied_volatility(
    option_type: OptionType,
    price: float,
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    method: IVMethod = IVMethod.SAFEGUARDED_NEWTON,
) -> float:
    """Implied volatility of a discounted European price. Raises ConvergenceError if the solver
    does not converge and InfeasiblePriceError if no volatility reproduces the price."""
    result = implied_volatility_result(option_type, price, S=S, K=K, T=T, r=r, q=q, method=method)
    if result.sigma is None:
        raise ConvergenceError(
            f"{result.method.value} did not converge: {result.reason.value} after "
            f"{result.iterations} iterations (residual {result.residual!r})"
        )
    return result.sigma


def implied_volatility_result(
    option_type: OptionType,
    price: float,
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    method: IVMethod = IVMethod.SAFEGUARDED_NEWTON,
    max_iterations: int | None = None,
    record_path: bool = False,
) -> IVResult:
    """Solve for the implied volatility of a discounted price V; full diagnostics (SPEC 6.3)."""
    m = validate_scalar_inputs(S=S, K=K, T=T, r=r, q=q, sigma=0.0, allow_extreme_vol=False)
    _check_price(price)
    forward = m.S * math.exp((m.r - m.q) * m.T)
    undiscounted = price / math.exp(-m.r * m.T)
    return solve_black(
        option_type,
        undiscounted,
        F=forward,
        K=m.K,
        T=m.T,
        method=method,
        max_iterations=max_iterations,
        record_path=record_path,
    )


def solve_black(
    option_type: OptionType,
    c: float,
    *,
    F: float,
    K: float,
    T: float,
    method: IVMethod = IVMethod.SAFEGUARDED_NEWTON,
    max_iterations: int | None = None,
    record_path: bool = False,
) -> IVResult:
    """Implied volatility of an undiscounted Black price c on forward F (SPEC 6.1, 6.2, 6.3)."""
    theta = option_sign(option_type)
    for name, value in (("F", F), ("K", K), ("T", T)):
        if not (math.isfinite(value) and value > 0.0):
            raise PricingInputError(f"{name} must be finite and > 0; got {value!r}")
    _check_price(c)
    if not isinstance(method, IVMethod):
        raise PricingInputError(f"method must be an IVMethod; got {method!r}")
    if max_iterations is not None and (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise PricingInputError(f"max_iterations must be an int >= 1; got {max_iterations!r}")

    lower = max(theta * (F - K), 0.0)
    upper = F if theta == 1 else K
    rounding = BOUND_TOL_FACTOR * EPS * max(F, K)
    lower_tol = rounding if lower > 0.0 else 0.0  # out of the money, iota = 0 exactly
    if c < lower - lower_tol:
        raise InfeasiblePriceError(f"price {c!r} is below the intrinsic value {lower!r}")
    if c >= upper - rounding:
        raise InfeasiblePriceError(
            f"price {c!r} is at or above the upper bound {upper!r}: volatility would be infinite"
        )
    if lower > 0.0:  # in the money: solve the out-of-the-money counterpart (put-call parity)
        time_value = c - lower
        if time_value <= 0.0:
            return _at_lower_bound(method, lower - c, record_path)
        theta, c, lower = -theta, time_value, 0.0
    if c == 0.0:  # out of the money (possibly after the parity step): c(0) = 0
        return _at_lower_bound(method, -c, record_path)

    problem = _Problem(theta, F, K, math.log(F / K), T, c, lower)
    if method is IVMethod.BRENT:
        return _brent(problem, max_iterations or BRENT_MAXITER, record_path)
    if method is IVMethod.NAIVE_NEWTON:
        return _naive_newton(problem, max_iterations or NEWTON_MAXITER, record_path)
    return _safeguarded_newton(problem, max_iterations or NEWTON_MAXITER, record_path)


def _at_lower_bound(method: IVMethod, residual: float, record_path: bool) -> IVResult:
    """s = 0: the price carries no information about volatility (SPEC 6.2)."""
    return IVResult(
        0.0, 0.0, residual, 0, 0, True, IVReason.AT_LOWER_BOUND, method, () if record_path else None
    )


def black_price(p: _Problem, s: float) -> float:
    """Undiscounted Black price c(s) (SPEC 3.1 in Black form), to a few eps relative: when the
    two terms nearly cancel, c is recomputed with extra digits (module docstring)."""
    if s == 0.0:
        return p.lower
    d1 = p.x / s + 0.5 * s
    th = p.theta
    a = p.F * float(ndtr(th * d1))
    b = p.K * float(ndtr(th * (d1 - s)))
    c = th * (a - b)
    largest = max(a, b)
    if largest == 0.0 or (c > 0.0 and largest <= CANCELLATION_LIMIT * c):
        return c  # no cancellation (or both terms underflowed, so c = 0 exactly)
    return _black_price_extra_digits(p, s, largest, c)


def _black_price_extra_digits(p: _Problem, s: float, largest: float, c: float) -> float:
    """c(s) evaluated with mpmath on the exact double inputs, with enough digits that the
    cancellation leaves at least 17 + CANCELLATION_GUARD_DIGITS, then rounded once."""
    lost = math.log10(largest / c) if c > 0.0 else -math.log10(EPS)  # c <= 0: all digits lost
    digits = 17 + math.ceil(lost) + CANCELLATION_GUARD_DIGITS
    while True:
        with mpmath.workdps(digits):
            F, K, s_mp = mpmath.mpf(p.F), mpmath.mpf(p.K), mpmath.mpf(s)
            d1 = mpmath.log(F / K) / s_mp + s_mp / 2
            value = p.theta * (
                F * mpmath.ncdf(p.theta * d1) - K * mpmath.ncdf(p.theta * (d1 - s_mp))
            )
            if value > 0:
                cancelled = math.ceil(float(mpmath.log10(mpmath.mpf(largest) / value)))
                needed = 17 + CANCELLATION_GUARD_DIGITS + cancelled
            else:  # every digit cancelled even here: c(s) > 0 for s > 0, so use more digits
                needed = digits + 1
        if needed <= digits or digits >= MP_MAX_DIGITS:
            return float(value)
        digits = min(2 * digits, MP_MAX_DIGITS)


def black_vega(p: _Problem, s: float) -> float:
    """dc/ds = F phi(d1) (SPEC 6.1)."""
    d1 = p.x / s + 0.5 * s
    return p.F * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)


class _Tracker:
    """Counts function evaluations and records iterates."""

    def __init__(self, problem: _Problem, record: bool) -> None:
        self.problem, self.record = problem, record
        self.evaluations = 0
        self.path: list[float] = []

    def residual(self, s: float) -> float:
        self.evaluations += 1
        if self.record:
            self.path.append(s)
        return black_price(self.problem, s) - self.problem.target

    def result(
        self, method: IVMethod, s: float | None, residual: float, iterations: int, reason: IVReason
    ) -> IVResult:
        converged = reason is IVReason.CONVERGED
        sigma = s / math.sqrt(self.problem.T) if converged and s is not None else None
        return IVResult(
            sigma=sigma,
            total_vol=s if converged else None,
            residual=residual,
            iterations=iterations,
            function_evaluations=self.evaluations,
            converged=converged,
            reason=reason,
            method=method,
            path=tuple(self.path) if self.record else None,
        )


def _brent(p: _Problem, max_iterations: int, record: bool) -> IVResult:
    t = _Tracker(p, record)
    hi = 1.0
    f_hi = t.residual(hi)
    while f_hi <= 0.0:
        if hi >= S_MAX:
            return t.result(IVMethod.BRENT, None, f_hi, 0, IVReason.BRACKET_FAILURE)
        hi = min(2.0 * hi, S_MAX)
        f_hi = t.residual(hi)
    root, info = brentq(
        t.residual,
        0.0,
        hi,
        xtol=BRENT_XTOL,
        rtol=CONVERGENCE_FACTOR * EPS,
        maxiter=max_iterations,
        full_output=True,
        disp=False,
    )
    s = float(root)
    reason = IVReason.CONVERGED if info.converged else IVReason.MAX_ITERATIONS
    return t.result(IVMethod.BRENT, s, t.residual(s), int(info.iterations), reason)


def _converged(residual: float, target: float) -> bool:
    return abs(residual) <= CONVERGENCE_FACTOR * EPS * target


def _naive_newton(p: _Problem, max_iterations: int, record: bool) -> IVResult:
    t = _Tracker(p, record)
    method = IVMethod.NAIVE_NEWTON
    s = DEFAULT_TOTAL_VOL_FACTOR * math.sqrt(p.T)
    residual = t.residual(s)
    for iteration in range(1, max_iterations + 1):
        if _converged(residual, p.target):
            return t.result(method, s, residual, iteration - 1, IVReason.CONVERGED)
        vega = black_vega(p, s)
        if vega == 0.0:
            return t.result(method, None, residual, iteration - 1, IVReason.ZERO_DERIVATIVE)
        s_next = s - residual / vega
        if not (math.isfinite(s_next) and s_next > 0.0):
            return t.result(method, None, residual, iteration, IVReason.DIVERGED)
        small_step = abs(s_next - s) <= CONVERGENCE_FACTOR * EPS * s_next
        s, residual = s_next, t.residual(s_next)
        if small_step:
            return t.result(method, s, residual, iteration, IVReason.CONVERGED)
    if _converged(residual, p.target):
        return t.result(method, s, residual, max_iterations, IVReason.CONVERGED)
    return t.result(method, None, residual, max_iterations, IVReason.MAX_ITERATIONS)


def _safeguarded_newton(p: _Problem, max_iterations: int, record: bool) -> IVResult:
    t = _Tracker(p, record)
    method = IVMethod.SAFEGUARDED_NEWTON
    if abs(p.x) >= SMALL_LOG_MONEYNESS:
        s = math.sqrt(2.0 * abs(p.x))  # Manaster and Koehler (1982)
    else:
        s = DEFAULT_TOTAL_VOL_FACTOR * math.sqrt(p.T)
    lo, hi = 0.0, math.inf  # c(0) = iota < target, established by the feasibility check
    previous_step = math.inf
    residual = t.residual(s)
    for iteration in range(1, max_iterations + 1):
        if _converged(residual, p.target):
            return t.result(method, s, residual, iteration - 1, IVReason.CONVERGED)
        if residual < 0.0:
            lo = s
        else:
            hi = s
        if hi - lo <= CONVERGENCE_FACTOR * EPS * s:
            return t.result(method, s, residual, iteration - 1, IVReason.CONVERGED)
        vega = black_vega(p, s)
        newton = s - residual / vega if vega > VEGA_MIN else math.nan
        progressing = abs(newton - s) < 0.5 * previous_step
        if lo < newton < hi and progressing:  # False for nan
            s_next = newton
        elif math.isinf(hi):
            s_next = 2.0 * s
        else:
            s_next = 0.5 * (lo + hi)
        previous_step = abs(s_next - s)
        small_step = previous_step <= CONVERGENCE_FACTOR * EPS * s_next
        s, residual = s_next, t.residual(s_next)
        if small_step:
            return t.result(method, s, residual, iteration, IVReason.CONVERGED)
    if _converged(residual, p.target):
        return t.result(method, s, residual, max_iterations, IVReason.CONVERGED)
    return t.result(method, None, residual, max_iterations, IVReason.MAX_ITERATIONS)


def _check_price(price: float) -> None:
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price):
        raise PricingInputError(f"price must be a finite real number; got {price!r}")
