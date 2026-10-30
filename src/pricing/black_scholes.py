"""Black-Scholes-Merton prices for European options (SPEC 3).

Model (SPEC 2.1). Under the risk-neutral measure dS = (r - q) S dt + sigma S dW, with r, q and
sigma constant. A price is the discounted risk-neutral expected payoff under those assumptions
(Black and Scholes 1973; Merton 1973 for the dividend yield q). It is not a forecast.

Kernel (SPEC 3.1), forward and total-volatility form:
    F = S exp((r - q) T)     forward price of the underlying for delivery at T
    D = exp(-r T)            discount factor
    s = sigma sqrt(T)        total volatility: the standard deviation of ln S_T
    x = ln(F / K)            log-moneyness against the forward
    d1 = x / s + s / 2,      d2 = d1 - s
    V = D theta (F N(theta d1) - K N(theta d2)),   theta = +1 for a call, -1 for a put

Derivation. Under the model, ln S_T ~ Normal(ln F - s^2/2, s^2), so E[S_T] = F. Then
E[K 1{S_T > K}] = K P(ln S_T > ln K) = K N(d2). Changing to the measure under which S_T / F is
the numeraire shifts the mean of ln S_T by s^2, so E[S_T 1{S_T > K}] = F N(d1). Their difference
is the undiscounted call, F N(d1) - K N(d2); the put follows from the same steps on 1{S_T < K}.
This is Black's (1976) formula applied to the forward, the parametrization Jaeckel (2015) uses,
which keeps Study A comparisons exact. Substituting F gives the spot form of SPEC 3.2.

Numerical form (SPEC 3.4, 3.5):
- Puts use their own formula, never call minus forward, which would cancel for deep OTM puts.
- Tails use N(theta d) via scipy.special.ndtr, never 1 - N(d): for d = 9, 1 - N(d) is computed
  as 0 while N(-d) is 1.1e-19 to full relative precision.
- Deep out-of-the-money prices may underflow to exactly 0. That is the correct float result.

Boundary policy (SPEC 3.4):
- T = 0: the price is the intrinsic value max(theta (S - K), 0).
- sigma = 0 with T > 0: the forward is certain, so the price is D max(theta (F - K), 0).
- No threshold switches to these formulas for small s; continuity is tested instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from pricing.contracts import (
    BoolArray,
    FloatArray,
    OptionType,
    option_sign,
    validate_market_inputs,
)


@dataclass(frozen=True)
class _Kernel:
    """Quantities shared by the price and Greek formulas (SPEC 3.1), computed once per call.

    s, d1 and d2 hold placeholder values (T = sigma = 1) wherever `regular` is False, so the
    vectorized closed form never evaluates 0/0. Those entries are always replaced by the boundary
    formulas before a result is returned.
    """

    theta: int  # +1 call, -1 put
    S: FloatArray
    K: FloatArray
    T: FloatArray
    r: FloatArray
    q: FloatArray
    sigma: FloatArray
    F: FloatArray  # forward
    D: FloatArray  # discount factor exp(-r T)
    disc_q: FloatArray  # dividend discount exp(-q T)
    sqrt_T: FloatArray  # sqrt(T), placeholder 1 where not regular
    s: FloatArray  # total volatility, placeholder where not regular
    d1: FloatArray
    d2: FloatArray
    regular: BoolArray  # T > 0 and sigma > 0: the closed form applies
    expired: BoolArray  # T == 0
    in_the_money: BoolArray  # theta (F - K) > 0
    at_forward: BoolArray  # F == K: where the boundary payoff has its kink
    all_scalar: bool


def _kernel(
    option_type: OptionType,
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    q: npt.ArrayLike,
    sigma: npt.ArrayLike,
    allow_extreme_vol: bool,
) -> _Kernel:
    theta = option_sign(option_type)
    m = validate_market_inputs(
        S=S, K=K, T=T, r=r, q=q, sigma=sigma, allow_extreme_vol=allow_extreme_vol
    )
    F = m.S * np.exp((m.r - m.q) * m.T)
    expired = m.T == 0.0
    regular = ~expired & (m.sigma > 0.0)
    sqrt_T = np.sqrt(np.where(regular, m.T, 1.0))
    s = np.where(regular, m.sigma, 1.0) * sqrt_T
    d1 = np.log(F / m.K) / s + 0.5 * s
    return _Kernel(
        theta=theta,
        S=m.S,
        K=m.K,
        T=m.T,
        r=m.r,
        q=m.q,
        sigma=m.sigma,
        F=F,
        D=np.exp(-m.r * m.T),
        disc_q=np.exp(-m.q * m.T),
        sqrt_T=sqrt_T,
        s=s,
        d1=d1,
        d2=d1 - s,
        regular=regular,
        expired=expired,
        in_the_money=theta * (F - m.K) > 0.0,
        at_forward=F == m.K,
        all_scalar=m.all_scalar,
    )


def _result(values: FloatArray, all_scalar: bool) -> float | FloatArray:
    """SPEC 2.3: all-scalar inputs return a Python float; any array input returns an ndarray."""
    return float(values) if all_scalar else values


def price(
    option_type: OptionType,
    *,
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    q: npt.ArrayLike,
    sigma: npt.ArrayLike,
    allow_extreme_vol: bool = False,
) -> float | FloatArray:
    """Discounted BSM price of a European option (SPEC 3.1), boundaries per SPEC 3.4.

    Market inputs are keyword-only so that r and q, or S and K, cannot be swapped by position.
    """
    k = _kernel(option_type, S, K, T, r, q, sigma, allow_extreme_vol)
    th = k.theta
    closed_form = k.D * th * (k.F * ndtr(th * k.d1) - k.K * ndtr(th * k.d2))
    # The payoff at a certain forward, not a clip. It also covers T = 0, where D = 1 and F = S
    # exactly, giving the intrinsic value max(theta (S - K), 0).
    certain_forward = k.D * np.maximum(th * (k.F - k.K), 0.0)
    return _result(np.where(k.regular, closed_form, certain_forward), k.all_scalar)
