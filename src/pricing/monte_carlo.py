"""Monte Carlo pricing of European options (SPEC 5).

Exact terminal simulation (SPEC 5.1). Under SPEC 2.1, ln S_T is normal, so S_T is sampled exactly
in one step, with no time-discretization error:
    S_T = S exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z),   Z ~ N(0, 1)
    Y   = exp(-r T) max(theta (S_T - K), 0),   E[Y] = BSM price.
The estimator is the sample mean of n independent Y. Its standard error is the sample standard
deviation over sqrt(n), and the 95% interval is estimate +/- z SE with z = N^-1(0.975)
(Glasserman 2004, section 1.1). The interval rests on the central limit theorem, so it can
undercover for small n or skewed payoffs such as deep out-of-the-money options; the coverage
experiment measures this (SPEC 5.5).

Antithetic variates (SPEC 5.2; Glasserman 2004, section 4.2). Each Z is paired with -Z. The pair
average A = (Y(Z) + Y(-Z)) / 2 is the unit of independence: the n / 2 pair averages are i.i.d.,
the two members of a pair are not, so the standard error is sd(A) / sqrt(n / 2). For a payoff that
is monotone in Z, Y(Z) and Y(-Z) are negatively correlated and Var(A) < Var(Y) / 2.

Control variate (SPEC 5.3; Glasserman 2004, section 4.1). X = exp(-r T) S_T has the known mean
E[X] = S exp(-q T). The estimator mean(Y) - beta (mean(X) - E[X]) is unbiased for any fixed beta,
and beta* = Cov(Y, X) / Var(X) minimizes its variance. By default beta is estimated on an
independent pilot sample, so it is independent of the main sample and the estimator stays
unbiased; the pilot's payoff evaluations count in the cost. Estimating beta on the main sample
(same_sample_beta=True) costs nothing extra but adds an O(1/n) bias.

Cost and variance reduction (SPEC 5.4). Cost is counted in payoff evaluations. A method's variance
reduction factor against plain Monte Carlo is VRF = (Var_plain cost_plain) / (Var_method
cost_method), the ratio of work needed for equal error. With SE^2 = Var / n this is
(SE_plain^2 n_plain) / (SE_method^2 n_method), counting each method's own cost.

Randomness (SPEC 2.3). A required seed, an int or a numpy SeedSequence, is the only source of
randomness. The main sample and the pilot use child sequences whose spawn keys append (0,) and
(1,). They are built directly rather than with SeedSequence.spawn, which would mutate the caller's
object, so the same seed always gives the same result.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.special import ndtri

from pricing.contracts import (
    FloatArray,
    OptionType,
    ScalarInputs,
    option_sign,
    validate_scalar_inputs,
)
from pricing.errors import PricingInputError

Z_95 = float(ndtri(0.975))  # 1.959963984540054, the two-sided 95% normal quantile


class MCMethod(enum.Enum):
    PLAIN = "plain"
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"


@dataclass(frozen=True)
class MCResult:
    """A Monte Carlo estimate with its sampling diagnostics (SPEC 5.1)."""

    estimate: float
    std_error: float
    ci_low: float  # normal-approximation 95% interval
    ci_high: float
    sample_sd: (
        float  # sd of the i.i.d. unit: paths, antithetic pair averages, or controlled samples
    )
    n_paths: int  # main-sample paths; antithetic counts Z and -Z separately
    n_payoff_evals: int  # cost, including any pilot sample
    method: MCMethod
    seed_entropy: int | tuple[int, ...]
    seed_spawn_key: tuple[int, ...]
    inputs: Mapping[str, float | str]  # option type and market inputs
    beta: float | None = None  # control-variate coefficient
    pilot_paths: int | None = None


def monte_carlo(
    option_type: OptionType,
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    seed: int | np.random.SeedSequence,
    method: MCMethod = MCMethod.PLAIN,
    pilot_paths: int | None = None,
    same_sample_beta: bool = False,
    allow_extreme_vol: bool = False,
) -> MCResult:
    """Monte Carlo price of a European option by exact terminal simulation (SPEC 5)."""
    theta = option_sign(option_type)
    m = validate_scalar_inputs(
        S=S, K=K, T=T, r=r, q=q, sigma=sigma, allow_extreme_vol=allow_extreme_vol
    )
    if m.T == 0.0 or m.sigma == 0.0:
        raise PricingInputError(
            "Monte Carlo needs T > 0 and sigma > 0 (otherwise S_T is certain); "
            "use black_scholes.price for those cases"
        )
    if not isinstance(method, MCMethod):
        raise PricingInputError(f"method must be an MCMethod; got {method!r}")
    _check_count("n_paths", n_paths, minimum=2)
    root = _seed_sequence(seed)
    rng = np.random.default_rng(_child(root, 0))
    market: dict[str, float | str] = {
        "option_type": option_type.name,
        "S": m.S,
        "K": m.K,
        "T": m.T,
        "r": m.r,
        "q": m.q,
        "sigma": m.sigma,
    }
    inputs = MappingProxyType(market)

    def result(
        samples: FloatArray, cost: int, beta: float | None = None, pilot: int | None = None
    ) -> MCResult:
        mean, sd, se = summarize(samples)
        return MCResult(
            estimate=mean,
            std_error=se,
            ci_low=mean - Z_95 * se,
            ci_high=mean + Z_95 * se,
            sample_sd=sd,
            n_paths=n_paths,
            n_payoff_evals=cost,
            method=method,
            seed_entropy=_entropy(root),
            seed_spawn_key=tuple(root.spawn_key),
            inputs=inputs,
            beta=beta,
            pilot_paths=pilot,
        )

    if method is MCMethod.PLAIN:
        payoffs, _ = _simulate(m, theta, rng.standard_normal(n_paths))
        return result(payoffs, n_paths)

    if method is MCMethod.ANTITHETIC:
        if n_paths % 2 or n_paths < 4:
            raise PricingInputError(f"antithetic n_paths must be even and >= 4; got {n_paths}")
        z = rng.standard_normal(n_paths // 2)
        pair_averages = 0.5 * (_simulate(m, theta, z)[0] + _simulate(m, theta, -z)[0])
        return result(pair_averages, n_paths)

    payoffs, discounted_spot = _simulate(m, theta, rng.standard_normal(n_paths))
    if same_sample_beta:
        beta, cost, pilot = _beta(payoffs, discounted_spot), n_paths, None
    else:
        if pilot_paths is None:
            raise PricingInputError(
                "control variate needs pilot_paths, or same_sample_beta=True (biased, SPEC 5.3)"
            )
        _check_count("pilot_paths", pilot_paths, minimum=2)
        pilot_rng = np.random.default_rng(_child(root, 1))
        beta = _beta(*_simulate(m, theta, pilot_rng.standard_normal(pilot_paths)))
        cost, pilot = n_paths + pilot_paths, pilot_paths
    expected_control = m.S * math.exp(-m.q * m.T)  # E[exp(-r T) S_T]
    controlled = payoffs - beta * (discounted_spot - expected_control)
    return result(controlled, cost, beta=beta, pilot=pilot)


def variance_reduction_factor(plain: MCResult, other: MCResult) -> float:
    """(SE_plain^2 cost_plain) / (SE_other^2 cost_other): above 1 means `other` is cheaper for the
    same error (SPEC 5.4)."""
    if plain.std_error <= 0.0 or other.std_error <= 0.0:
        raise ValueError("variance reduction factor needs positive standard errors")
    work_plain = plain.std_error**2 * plain.n_payoff_evals
    return work_plain / (other.std_error**2 * other.n_payoff_evals)


def summarize(samples: FloatArray) -> tuple[float, float, float]:
    """Mean, sample sd (ddof = 1) and standard error sd / sqrt(n) of i.i.d. samples."""
    sd = float(np.std(samples, ddof=1))
    return float(np.mean(samples)), sd, sd / math.sqrt(samples.size)


def _simulate(m: ScalarInputs, theta: int, z: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Discounted payoffs Y and discounted terminal prices X for normal draws z (SPEC 5.1)."""
    terminal = m.S * np.exp((m.r - m.q - 0.5 * m.sigma**2) * m.T + m.sigma * math.sqrt(m.T) * z)
    discount = math.exp(-m.r * m.T)
    return discount * np.maximum(theta * (terminal - m.K), 0.0), discount * terminal


def _beta(payoffs: FloatArray, controls: FloatArray) -> float:
    """Sample Cov(Y, X) / Var(X), the variance-minimizing control coefficient (SPEC 5.3)."""
    centred = controls - controls.mean()
    variance = float(centred @ centred)
    if variance <= 0.0:
        raise PricingInputError("control variate has zero sample variance; beta is undefined")
    return float(centred @ (payoffs - payoffs.mean())) / variance


def _check_count(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PricingInputError(f"{name} must be an int >= {minimum}; got {value!r}")


def _seed_sequence(seed: int | np.random.SeedSequence) -> np.random.SeedSequence:
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PricingInputError(f"seed must be a non-negative int or a SeedSequence; got {seed!r}")
    return np.random.SeedSequence(seed)


def _child(root: np.random.SeedSequence, index: int) -> np.random.SeedSequence:
    """The child SeedSequence.spawn would create, built without mutating `root`."""
    return np.random.SeedSequence(
        root.entropy, spawn_key=(*root.spawn_key, index), pool_size=root.pool_size
    )


def _entropy(root: np.random.SeedSequence) -> int | tuple[int, ...]:
    entropy = root.entropy
    if isinstance(entropy, int):
        return int(entropy)
    if entropy is None:  # numpy always sets entropy at construction; kept for the type checker
        raise PricingInputError("seed sequence has no entropy")
    return tuple(int(e) for e in entropy)
