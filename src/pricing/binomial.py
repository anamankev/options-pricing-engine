"""Binomial trees for European and American options (SPEC 4).

All trees take N steps of length dt = T / N. Over one step the stock moves from S to S u (with
probability p) or S d, the growth factor of the forward is g = exp((r - q) dt), and one step of
discounting is exp(-r dt). Backward induction runs one time layer at a time, vectorized over the
layer: O(N^2) time, O(N) memory.

Cox-Ross-Rubinstein (Cox, Ross and Rubinstein 1979; SPEC 4.1):
    u = exp(sigma sqrt(dt)),  d = 1 / u,  p = (g - d) / (u - d)
p makes the discounted stock a martingale on the tree: p u + (1 - p) d = g.

Validity (SPEC 4.1). p is a probability, 0 < p < 1, exactly when d < g < u. Taking logs, with
ln u = sigma sqrt(dt), ln d = -sigma sqrt(dt) and ln g = (r - q) dt:
    -sigma sqrt(dt) < (r - q) dt < sigma sqrt(dt)   <=>   |r - q| sqrt(dt) < sigma.
With dt = T / N this is N > (r - q)^2 T / sigma^2. If it fails the tree is not arbitrage-free and
TreeParameterError says how many steps are needed. p is never clipped into (0, 1).

Numerical form. For small dt both g - d and u - d are differences of nearly equal numbers. Here
u - d = 2 sinh(sigma sqrt(dt)) and g - d = d (exp((r - q) dt + sigma sqrt(dt)) - 1), computed with
sinh and expm1, so p keeps full relative precision. Node prices are S exp(j ln u + (n - j) ln d)
for n steps and j up-moves, evaluated directly at every layer rather than by repeated division,
which would accumulate rounding over N layers.

Leisen-Reimer (Leisen and Reimer 1996; SPEC 4.2). N odd. With d1, d2 from SPEC 3.1 for the full
maturity and the Peizer-Pratt method-2 inversion
    h(z) = 1/2 + sign(z) (1/2) sqrt(1 - exp(-(z / (N + 1/3 + 0.1 / (N + 1)))^2 (N + 1/6))),
    p = h(d2),  p_bar = h(d1),  u = g p_bar / p,  d = g (1 - p_bar) / (1 - p).
The last form equals SPEC's d = (g - p u) / (1 - p) but avoids that subtraction; it also gives
p u + (1 - p) d = g exactly. h and 1 - h are both computed without cancellation: 1 - h(z) for
z > 0 is (1/2) exp(-x) / (1 + sqrt(1 - exp(-x))), which stays accurate when exp(-x) is tiny, the
same issue as N(-x) versus 1 - N(x). The tree is centred on the strike, so its European error falls
like 1/N^2 (SPEC 4.2; measured in Study B). Equation numbers in the paper: SPEC 17.

BBS and BBSR (Broadie and Detemple 1996; SPEC 4.3). BBS is a CRR tree whose layer N - 1 holds the
Black-Scholes European value over the last step dt (for American options, the larger of that and
immediate exercise); induction proceeds as usual from there. This removes the payoff kink from the
last step, which is where CRR's oscillation comes from. BBSR is two-point Richardson
extrapolation, 2 V_BBS(N) - V_BBS(N / 2), for even N.

Richardson extrapolation (SPEC 4.5). If V(N) = V + c N^(-p) + ..., then
(2^p V(2N) - V(N)) / (2^p - 1) cancels the c term. Use it only where the error is monotone in N
(LR with odd N, BBS); on raw CRR's odd/even oscillation it is unreliable.

American exercise (SPEC 4.4): at every node V = max(continuation, theta (S_node - K)). The
optional early-exercise boundary records, per layer, the critical stock price: the highest
exercised node for a put, the lowest for a call, or None where no node is exercised.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from pricing import black_scholes as bs
from pricing.contracts import (
    ExerciseStyle,
    FloatArray,
    OptionType,
    ScalarInputs,
    option_sign,
    validate_scalar_inputs,
)
from pricing.errors import PricingInputError, TreeParameterError


@dataclass(frozen=True)
class TreeResult:
    """A tree price with the parameters that produced it (SPEC 2.3)."""

    value: float
    method: str  # "CRR", "LR", "BBS" or "BBSR"
    exercise: ExerciseStyle
    n_steps: int
    dt: float
    up: float  # u
    down: float  # d
    p_up: float  # risk-neutral probability of an up move
    # Critical stock price per layer 0 .. N-1 (American, when requested); None: no exercise.
    exercise_boundary: tuple[float | None, ...] | None = None
    # BBSR only: (BBS with N steps, BBS with N / 2 steps).
    components: tuple[float, float] | None = None


@dataclass(frozen=True)
class _Lattice:
    """Step parameters shared by every tree built on one market."""

    theta: int
    m: ScalarInputs
    n_steps: int
    dt: float
    ln_u: float
    ln_d: float
    p: float
    discount: float  # exp(-r dt)


def crr(
    option_type: OptionType,
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int,
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    record_boundary: bool = False,
    allow_extreme_vol: bool = False,
) -> TreeResult:
    """Cox-Ross-Rubinstein tree price (SPEC 4.1, 4.4)."""
    _check_exercise(exercise)
    lattice = _crr_lattice(option_type, S, K, T, r, q, sigma, n_steps, allow_extreme_vol)
    values = _payoff(lattice, _spots(lattice, lattice.n_steps))
    value, boundary = _roll_back(lattice, values, lattice.n_steps, exercise, record_boundary)
    return _result(value, "CRR", exercise, lattice, boundary)


def leisen_reimer(
    option_type: OptionType,
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int,
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    record_boundary: bool = False,
    allow_extreme_vol: bool = False,
) -> TreeResult:
    """Leisen-Reimer tree price, n_steps odd (SPEC 4.2, 4.4)."""
    theta = option_sign(option_type)
    m = _tree_inputs(S, K, T, r, q, sigma, allow_extreme_vol)
    _check_exercise(exercise)
    _check_steps(n_steps, odd=True)
    dt = m.T / n_steps
    s = m.sigma * math.sqrt(m.T)
    d1 = (math.log(m.S / m.K) + (m.r - m.q) * m.T) / s + 0.5 * s
    p, one_minus_p = _peizer_pratt_2(d1 - s, n_steps)
    p_bar, one_minus_p_bar = _peizer_pratt_2(d1, n_steps)
    carry = (m.r - m.q) * dt
    ln_u = carry + _ln_probability(p_bar, one_minus_p_bar) - _ln_probability(p, one_minus_p)
    ln_d = carry + _ln_probability(one_minus_p_bar, p_bar) - _ln_probability(one_minus_p, p)
    _check_probability(p, "LR")
    if not ln_d < carry < ln_u:
        raise TreeParameterError(f"LR tree needs d < g < u; got ln d={ln_d!r}, ln u={ln_u!r}")
    lattice = _Lattice(theta, m, n_steps, dt, ln_u, ln_d, p, math.exp(-m.r * dt))
    values = _payoff(lattice, _spots(lattice, n_steps))
    value, boundary = _roll_back(lattice, values, n_steps, exercise, record_boundary)
    return _result(value, "LR", exercise, lattice, boundary)


def _peizer_pratt_2(z: float, n: int) -> tuple[float, float]:
    """(h(z), 1 - h(z)) for SPEC 4.2's Peizer-Pratt method-2 inversion, both without
    cancellation."""
    t = z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))
    x = t * t * (n + 1.0 / 6.0)
    root = math.sqrt(-math.expm1(-x))  # sqrt(1 - exp(-x))
    small = 0.5 * math.exp(-x) / (1.0 + root)  # (1 - root) / 2, rewritten
    large = 0.5 + 0.5 * root
    return (large, small) if z > 0 else (small, large)


def _ln_probability(p: float, one_minus_p: float) -> float:
    """ln p, via log1p(-(1 - p)) when p is near 1 so no digits are lost."""
    return math.log1p(-one_minus_p) if p > 0.5 else math.log(p)


def bbs(
    option_type: OptionType,
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int,
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    record_boundary: bool = False,
    allow_extreme_vol: bool = False,
) -> TreeResult:
    """Broadie-Detemple BBS: CRR with Black-Scholes values one step before expiry (SPEC 4.3)."""
    _check_exercise(exercise)
    lattice = _crr_lattice(option_type, S, K, T, r, q, sigma, n_steps, allow_extreme_vol)
    m, last = lattice.m, lattice.n_steps - 1
    spots = _spots(lattice, last)
    values = np.asarray(
        bs.price(
            option_type,
            S=spots,
            K=m.K,
            T=lattice.dt,
            r=m.r,
            q=m.q,
            sigma=m.sigma,
            allow_extreme_vol=allow_extreme_vol,
        )
    )
    last_boundary: list[float | None] = []
    if exercise is ExerciseStyle.AMERICAN:
        intrinsic = lattice.theta * (spots - m.K)
        exercised = intrinsic > values
        values = np.where(exercised, intrinsic, values)
        last_boundary = [_critical_spot(spots, exercised, lattice.theta)]
    value, boundary = _roll_back(lattice, values, last, exercise, record_boundary)
    if boundary is not None:
        boundary = boundary + tuple(last_boundary)
    return _result(value, "BBS", exercise, lattice, boundary)


def bbsr(
    option_type: OptionType,
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int,
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
    record_boundary: bool = False,
    allow_extreme_vol: bool = False,
) -> TreeResult:
    """BBSR = 2 BBS(N) - BBS(N / 2), n_steps even (SPEC 4.3). Parameters are those of BBS(N)."""
    _check_exercise(exercise)
    _check_steps(n_steps, even=True)
    full = bbs(
        option_type,
        S=S,
        K=K,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        n_steps=n_steps,
        exercise=exercise,
        record_boundary=record_boundary,
        allow_extreme_vol=allow_extreme_vol,
    )
    half = bbs(
        option_type,
        S=S,
        K=K,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        n_steps=n_steps // 2,
        exercise=exercise,
        allow_extreme_vol=allow_extreme_vol,
    )
    return dataclasses.replace(
        full,
        value=richardson(half.value, full.value, order=1.0),
        method="BBSR",
        components=(full.value, half.value),
    )


def richardson(coarse: float, fine: float, *, order: float) -> float:
    """Extrapolate V(N) (coarse) and V(2N) (fine) for an error c N^-order (SPEC 4.5)."""
    if not order > 0.0:
        raise ValueError(f"order must be > 0; got {order!r}")
    weight = math.pow(2.0, order)
    return (weight * fine - coarse) / (weight - 1.0)


def _crr_lattice(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int,
    allow_extreme_vol: bool,
) -> _Lattice:
    theta = option_sign(option_type)
    m = _tree_inputs(S, K, T, r, q, sigma, allow_extreme_vol)
    _check_steps(n_steps)
    dt = m.T / n_steps
    ln_u = m.sigma * math.sqrt(dt)
    carry = (m.r - m.q) * dt  # ln g
    if not abs(carry) < ln_u:
        minimum = (m.r - m.q) ** 2 * m.T / m.sigma**2
        raise TreeParameterError(
            f"CRR probability outside (0, 1): needs |r - q| sqrt(dt) < sigma (SPEC 4.1); "
            f"use n_steps > {minimum:.6g} (got {n_steps})"
        )
    p = math.exp(-ln_u) * math.expm1(carry + ln_u) / (2.0 * math.sinh(ln_u))
    _check_probability(p, "CRR")
    return _Lattice(theta, m, n_steps, dt, ln_u, -ln_u, p, math.exp(-m.r * dt))


def _tree_inputs(
    S: float, K: float, T: float, r: float, q: float, sigma: float, allow_extreme_vol: bool
) -> ScalarInputs:
    m = validate_scalar_inputs(
        S=S, K=K, T=T, r=r, q=q, sigma=sigma, allow_extreme_vol=allow_extreme_vol
    )
    if m.T == 0.0 or m.sigma == 0.0:
        raise TreeParameterError(
            "a tree needs T > 0 and sigma > 0 (no steps or no spread); "
            "use black_scholes.price for those boundary cases"
        )
    return m


def _check_exercise(exercise: object) -> None:
    """Only an ExerciseStyle is accepted. A string such as "american" is not an ExerciseStyle, so
    `exercise is ExerciseStyle.AMERICAN` would be False and the tree would silently price a
    European option while reporting exercise="american"."""
    if not isinstance(exercise, ExerciseStyle):
        raise PricingInputError(
            f"exercise must be ExerciseStyle.EUROPEAN or ExerciseStyle.AMERICAN; got {exercise!r}"
        )


def _check_steps(n_steps: int, *, odd: bool = False, even: bool = False) -> None:
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise TreeParameterError(f"n_steps must be a positive int; got {n_steps!r}")
    if odd and n_steps % 2 == 0:
        raise TreeParameterError(f"n_steps must be odd for this tree; got {n_steps}")
    if even and n_steps % 2 == 1:
        raise TreeParameterError(f"n_steps must be even for this tree; got {n_steps}")


def _check_probability(p: float, method: str) -> None:
    if not 0.0 < p < 1.0:
        raise TreeParameterError(f"{method} probability p = {p!r} is outside (0, 1)")


def _spots(lattice: _Lattice, step: int) -> FloatArray:
    """Stock prices at layer `step`, node j = number of up moves, j = 0 .. step."""
    j = np.arange(step + 1, dtype=np.float64)
    return lattice.m.S * np.exp(j * lattice.ln_u + (step - j) * lattice.ln_d)


def _payoff(lattice: _Lattice, spots: FloatArray) -> FloatArray:
    return np.maximum(lattice.theta * (spots - lattice.m.K), 0.0)


def _roll_back(
    lattice: _Lattice,
    values: FloatArray,
    from_step: int,
    exercise: ExerciseStyle,
    record_boundary: bool,
) -> tuple[float, tuple[float | None, ...] | None]:
    """Backward induction from the values at layer `from_step` down to layer 0."""
    american = exercise is ExerciseStyle.AMERICAN
    boundary: list[float | None] = []
    p, discount = lattice.p, lattice.discount
    for step in range(from_step - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            spots = _spots(lattice, step)
            intrinsic = lattice.theta * (spots - lattice.m.K)
            exercised = intrinsic > values
            values = np.where(exercised, intrinsic, values)
            if record_boundary:
                boundary.append(_critical_spot(spots, exercised, lattice.theta))
    recorded = tuple(reversed(boundary)) if american and record_boundary else None
    return float(values[0]), recorded


def _critical_spot(spots: FloatArray, exercised: np.ndarray, theta: int) -> float | None:
    """Highest exercised spot for a put, lowest for a call; None when nothing is exercised."""
    if not exercised.any():
        return None
    hits = spots[exercised]
    return float(hits.max() if theta == -1 else hits.min())


def _result(
    value: float,
    method: str,
    exercise: ExerciseStyle,
    lattice: _Lattice,
    boundary: tuple[float | None, ...] | None,
    components: tuple[float, float] | None = None,
) -> TreeResult:
    return TreeResult(
        value=value,
        method=method,
        exercise=exercise,
        n_steps=lattice.n_steps,
        dt=lattice.dt,
        up=math.exp(lattice.ln_u),
        down=math.exp(lattice.ln_d),
        p_up=lattice.p,
        exercise_boundary=boundary,
        components=components,
    )
