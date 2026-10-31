"""Black-Scholes-Merton prices and analytic Greeks for European options (SPEC 3).

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
- Accuracy is absolute: |error| <= 64 eps x scale, where scale is about the size of S and K
  (tests/test_black_scholes.py). Near the money at extremely small total volatility the price
  itself is tiny and F N(d1) - K N(d2) cancels, so its relative error grows (about 7e-5 at
  sigma sqrt(T) = 1e-12), while the absolute error stays a fraction of eps S. Where relative
  accuracy of such a price matters, as in implied volatility, pricing.implied_vol.black_price
  evaluates it with extra digits (Study A2).

Boundary policy (SPEC 3.4):
- T = 0: the price is the intrinsic value max(theta (S - K), 0).
- sigma = 0 with T > 0: the forward is certain, so the price is D max(theta (F - K), 0).
- No threshold switches to these formulas for small s; continuity is tested instead.
- Greeks at those boundaries are the derivatives of the boundary price: the deterministic limit.
  Where that price has a kink (F = K, which at T = 0 means S = K), delta, gamma, vega, theta,
  rho and phi do not exist and raise UndefinedGreekError. Gamma, vega and theta also raise at
  T = 0 for every S (SPEC 3.4). Rho and phi at T = 0 are 0: the expiry payoff does not depend on
  r or q (SPEC is silent here; this is the exact derivative).

Greeks (SPEC 3.3), per 1.00 change in the input (SPEC 2.2); theta = -dV/dT per year. With
phi(.) the standard normal density, Sq = S exp(-q T) and Kr = K exp(-r T):
    delta = theta exp(-q T) N(theta d1)
    gamma = exp(-q T) phi(d1) / (S s)
    vega  = Sq phi(d1) sqrt(T)
    theta = -Sq phi(d1) sigma / (2 sqrt(T)) - theta r Kr N(theta d2) + theta q Sq N(theta d1)
    rho   = theta T Kr N(theta d2)
    phi   = -theta T Sq N(theta d1)        (dividend rho, dV/dq)
Each row of SPEC 3.3's call and put columns is the theta = +1 and theta = -1 case of these lines.
They follow from differentiating V with the identity Sq phi(d1) = Kr phi(d2), which removes every
term in dN/dd (Hull, Options, Futures, and Other Derivatives, chapter on the Greek letters).
"""

from __future__ import annotations

import math
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
from pricing.errors import UndefinedGreekError

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_AT_KINK = (
    "where the boundary payoff has its kink at the forward "
    "(T = 0 with S = K, or sigma = 0 with F = K; SPEC 3.4)"
)
_AT_EXPIRY = "at T = 0 (SPEC 3.4)"


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


def delta(
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
    """dV/dS per 1.00 of S. Raises UndefinedGreekError at the boundary kink (SPEC 3.4)."""
    k = _kernel(option_type, S, K, T, r, q, sigma, allow_extreme_vol)
    _raise_if_undefined("delta", ~k.regular & k.at_forward, _AT_KINK)
    th = k.theta
    closed_form = th * k.disc_q * ndtr(th * k.d1)
    certain_forward = th * k.disc_q * k.in_the_money.astype(np.float64)
    return _result(np.where(k.regular, closed_form, certain_forward), k.all_scalar)


def gamma(
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
    """d2V/dS2 per 1.00 of S. Raises UndefinedGreekError at T = 0 and at the kink (SPEC 3.4)."""
    k = _kernel(option_type, S, K, T, r, q, sigma, allow_extreme_vol)
    _raise_if_undefined("gamma", k.expired, _AT_EXPIRY)
    _raise_if_undefined("gamma", ~k.regular & k.at_forward, _AT_KINK)
    closed_form = k.disc_q * _normal_pdf(k.d1) / (k.S * k.s)
    return _result(np.where(k.regular, closed_form, 0.0), k.all_scalar)


def vega(
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
    """dV/dsigma per 1.00 of sigma (per_vol_point divides by 100). Raises like gamma."""
    k = _kernel(option_type, S, K, T, r, q, sigma, allow_extreme_vol)
    _raise_if_undefined("vega", k.expired, _AT_EXPIRY)
    _raise_if_undefined("vega", ~k.regular & k.at_forward, _AT_KINK)
    closed_form = k.S * k.disc_q * _normal_pdf(k.d1) * k.sqrt_T
    return _result(np.where(k.regular, closed_form, 0.0), k.all_scalar)


def theta(
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
    """-dV/dT per year: calendar decay, negative for a typical long option (SPEC 2.2).

    Raises like gamma. At sigma = 0 away from the kink it is theta (q Sq - r Kr) in the money and
    0 out of it: minus the T-derivative of D max(theta (F - K), 0).
    """
    k = _kernel(option_type, S, K, T, r, q, sigma, allow_extreme_vol)
    _raise_if_undefined("theta", k.expired, _AT_EXPIRY)
    _raise_if_undefined("theta", ~k.regular & k.at_forward, _AT_KINK)
    th = k.theta
    Sq, Kr = k.S * k.disc_q, k.K * k.D
    closed_form = (
        -Sq * _normal_pdf(k.d1) * k.sigma / (2.0 * k.sqrt_T)
        - th * k.r * Kr * ndtr(th * k.d2)
        + th * k.q * Sq * ndtr(th * k.d1)
    )
    certain_forward = th * (k.q * Sq - k.r * Kr) * k.in_the_money.astype(np.float64)
    return _result(np.where(k.regular, closed_form, certain_forward), k.all_scalar)


def rho(
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
    """dV/dr per 1.00 of r. 0 at T = 0; raises at the sigma = 0 kink (SPEC 3.4)."""
    k = _kernel(option_type, S, K, T, r, q, sigma, allow_extreme_vol)
    _raise_if_undefined("rho", ~k.regular & ~k.expired & k.at_forward, _AT_KINK)
    th = k.theta
    Kr = k.K * k.D
    closed_form = th * k.T * Kr * ndtr(th * k.d2)
    certain_forward = th * k.T * Kr * k.in_the_money.astype(np.float64)
    return _result(np.where(k.regular, closed_form, certain_forward), k.all_scalar)


def phi(
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
    """Dividend rho, dV/dq per 1.00 of q. 0 at T = 0; raises at the sigma = 0 kink (SPEC 3.4)."""
    k = _kernel(option_type, S, K, T, r, q, sigma, allow_extreme_vol)
    _raise_if_undefined("phi", ~k.regular & ~k.expired & k.at_forward, _AT_KINK)
    th = k.theta
    Sq = k.S * k.disc_q
    closed_form = -th * k.T * Sq * ndtr(th * k.d1)
    certain_forward = -th * k.T * Sq * k.in_the_money.astype(np.float64)
    return _result(np.where(k.regular, closed_form, certain_forward), k.all_scalar)


def _normal_pdf(x: FloatArray) -> FloatArray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _raise_if_undefined(greek: str, undefined: BoolArray, reason: str) -> None:
    """Raise for the whole call if any element is undefined; arrays name the first bad index."""
    if not np.any(undefined):
        return
    where = ""
    if undefined.ndim > 0:
        first = tuple(int(i) for i in np.argwhere(undefined)[0])
        where = f" ({int(np.count_nonzero(undefined))} inputs; first at index {first})"
    raise UndefinedGreekError(f"{greek} is undefined {reason}{where}")
