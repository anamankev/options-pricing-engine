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

American exercise (SPEC 4.4): at every node V = max(continuation, theta (S_node - K)). The
optional early-exercise boundary records, per layer, the critical stock price: the highest
exercised node for a put, the lowest for a call, or None where no node is exercised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricing.contracts import (
    ExerciseStyle,
    FloatArray,
    OptionType,
    ScalarInputs,
    option_sign,
    validate_scalar_inputs,
)
from pricing.errors import TreeParameterError


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
    lattice = _crr_lattice(option_type, S, K, T, r, q, sigma, n_steps, allow_extreme_vol)
    values = _payoff(lattice, _spots(lattice, lattice.n_steps))
    value, boundary = _roll_back(lattice, values, lattice.n_steps, exercise, record_boundary)
    return _result(value, "CRR", exercise, lattice, boundary)


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
