"""High-precision BSM reference used as ground truth in tests.

Evaluates the SPEC 3.1 formula with mpmath at 50 significant digits on the exact binary values of
the float64 inputs, then rounds once to float64. A comparison against it therefore measures only
the float64 implementation's error. Every expected value from here is "computed by mpmath at 50
significant digits" (the installed mpmath version is printed by `pip list` in CI logs).
"""

import mpmath

DIGITS = 50


def mp_price_and_error_scale(
    theta: int, *, S: float, K: float, T: float, r: float, q: float, sigma: float
) -> tuple[float, float]:
    """Exact-input BSM price, and the scale of its float64 rounding error.

    scale = D F N(theta d1) + D K N(theta d2) + D F phi(d1) (1 + |d1| + |d2|); see the rounding
    budget in tests/test_black_scholes.py for the derivation.
    """
    with mpmath.workdps(DIGITS):
        S_, K_, T_, r_, q_, sigma_ = (mpmath.mpf(v) for v in (S, K, T, r, q, sigma))
        F = S_ * mpmath.exp((r_ - q_) * T_)
        D = mpmath.exp(-r_ * T_)
        s = sigma_ * mpmath.sqrt(T_)
        d1 = mpmath.log(F / K_) / s + s / 2
        d2 = d1 - s
        forward_term = D * F * mpmath.ncdf(theta * d1)
        strike_term = D * K_ * mpmath.ncdf(theta * d2)
        vega_s = D * F * mpmath.npdf(d1)  # dV/ds
        scale = forward_term + strike_term + vega_s * (1 + abs(d1) + abs(d2))
        return float(theta * (forward_term - strike_term)), float(scale)
