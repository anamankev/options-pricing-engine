"""Central finite differences. In Phase 1 they are a test tool that checks analytic Greeks (SPEC 7).

Formulas (SPEC 7), for f smooth near an interior point x. Expanding f(x + h) and f(x - h) in
Taylor series about x, the even terms cancel in their difference and the odd terms in their sum:
    f'(x)  ~ (f(x + h) - f(x - h)) / (2h)             truncation error  h^2 f'''(x) / 6
    f''(x) ~ (f(x + h) - 2 f(x) + f(x - h)) / h^2      truncation error  h^2 f''''(x) / 12
Rounding error grows as h shrinks: about eps |f| / h for f' and eps |f| / h^2 for f''. Balancing
the two gives h ~ eps^(1/3) and h ~ eps^(1/4) relative to the scale of x (SPEC 7; measured in
the Phase 3 bump study).

Steps are relative, h = eta * max(|x|, x_scale) (SPEC 7), so one eta works for S = 100 and for
r = 0.05.

Exact steps: x + h is rounded to a float, so the step actually taken is not exactly h. These
functions divide by the steps actually taken, (x + h) - x and x - (x - h), which are computed
exactly whenever h <= |x| (Sterbenz's lemma; Higham 2002, Theorem 2.5). See Press et al.,
Numerical Recipes, 3rd ed., section 5.7. The second derivative therefore uses the three-point
formula for unequal steps, which equals SPEC 7's formula when the two steps are equal.

One-sided differences (SPEC 7), for points where x - h would leave the domain, such as T near 0
or sigma near 0. Callers label results computed this way. With the steps actually taken:
    forward_first:         (f(x + h) - f(x)) / h                    truncation error (h / 2) f''
    forward_first_order2:  three points x, x + h, x + 2h            truncation error -(h^2 / 3) f'''
The second is the derivative at x of the quadratic through the three points (Lagrange form); with
equal steps it is (-3 f(x) + 4 f(x + h) - f(x + 2h)) / (2h).
"""

from __future__ import annotations

from collections.abc import Callable


def relative_step(x: float, *, eta: float, x_scale: float) -> float:
    """Bump size h = eta * max(|x|, x_scale) (SPEC 7). x_scale sets h when x is near zero."""
    if not (eta > 0.0 and x_scale > 0.0):
        raise ValueError(f"eta and x_scale must be > 0; got eta={eta!r}, x_scale={x_scale!r}")
    return eta * max(abs(x), x_scale)


def central_first(f: Callable[[float], float], x: float, h: float) -> float:
    """Central-difference estimate of f'(x) with step h > 0, using the steps actually taken."""
    x_up, x_down = _bracket(x, h)
    return (f(x_up) - f(x_down)) / (x_up - x_down)


def central_second(f: Callable[[float], float], x: float, h: float) -> float:
    """Central-difference estimate of f''(x) with step h > 0, using the steps actually taken."""
    x_up, x_down = _bracket(x, h)
    h_up, h_down = x_up - x, x - x_down
    numerator = h_down * f(x_up) - (h_up + h_down) * f(x) + h_up * f(x_down)
    return 2.0 * numerator / (h_up * h_down * (h_up + h_down))


def forward_first(f: Callable[[float], float], x: float, h: float) -> float:
    """One-sided (forward) estimate of f'(x), first order, using the step actually taken."""
    x_up = _forward_point(x, h, 1.0)
    return (f(x_up) - f(x)) / (x_up - x)


def forward_first_order2(f: Callable[[float], float], x: float, h: float) -> float:
    """One-sided (forward) estimate of f'(x) from x, x + h, x + 2h, second order."""
    a = _forward_point(x, h, 1.0) - x
    b = _forward_point(x, h, 2.0) - x
    return -(a + b) / (a * b) * f(x) + b / (a * (b - a)) * f(x + a) - a / (b * (b - a)) * f(x + b)


def _forward_point(x: float, h: float, multiple: float) -> float:
    if not h > 0.0:
        raise ValueError(f"step h must be > 0; got {h!r}")
    point = x + multiple * h
    if point == x:
        raise ValueError(f"step h={h!r} is below the floating-point resolution of x={x!r}")
    return point


def _bracket(x: float, h: float) -> tuple[float, float]:
    if not h > 0.0:
        raise ValueError(f"step h must be > 0; got {h!r}")
    x_up, x_down = x + h, x - h
    if x_up == x or x_down == x:
        raise ValueError(f"step h={h!r} is below the floating-point resolution of x={x!r}")
    return x_up, x_down
