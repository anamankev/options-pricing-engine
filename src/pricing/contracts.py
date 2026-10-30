"""Option types, market-input validation and unit helpers shared by every pricer.

Conventions (SPEC 2.2): S and K are positive currency amounts; T is in years, T >= 0; r and q are
continuously compounded annualized decimals and may be any finite value, including negative;
sigma is an annualized decimal, sigma >= 0, where 0.20 means 20%. Greeks are per 1.00 change in
their input; the helpers at the end of this module convert to desk units.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from pricing.errors import PricingInputError

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# SPEC 2.2: sigma above 5 (500%) almost always means 20 was passed for 0.20.
MAX_SIGMA_WITHOUT_OVERRIDE = 5.0

DAYS_PER_YEAR = 365.0  # calendar days, matching the Actual365Fixed QuantLib oracle (SPEC 2.2)
VOL_POINTS_PER_UNIT = 100.0  # sigma = 1.00 is 100 volatility points
PERCENTAGE_POINTS_PER_UNIT = 100.0  # r = 1.00 is 100 percentage points

_REAL_DTYPE_KINDS = "iuf"  # signed int, unsigned int, float; rejects bool, complex, str, object


class OptionType(enum.Enum):
    """Call or put. The value is theta_cp from SPEC 2.2: +1 for calls, -1 for puts."""

    CALL = 1
    PUT = -1


@dataclass(frozen=True)
class MarketInputs:
    """Validated market inputs as float64 arrays broadcast to one shape (SPEC 2.3)."""

    S: FloatArray
    K: FloatArray
    T: FloatArray
    r: FloatArray
    q: FloatArray
    sigma: FloatArray
    all_scalar: bool  # every input was a scalar, so pricers return a Python float


def option_sign(option_type: object) -> int:
    """theta_cp for an option type: +1 for a call, -1 for a put. Raises PricingInputError."""
    if not isinstance(option_type, OptionType):
        raise PricingInputError(
            f"option_type must be OptionType.CALL or OptionType.PUT; got {option_type!r}"
        )
    sign: int = option_type.value
    return sign


def validate_market_inputs(
    *,
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    q: npt.ArrayLike,
    sigma: npt.ArrayLike,
    allow_extreme_vol: bool,
) -> MarketInputs:
    """Check every input against SPEC 2.2 and 2.4, then broadcast them to one shape.

    Raises PricingInputError naming the parameter, the offending value and, for arrays, its
    index. Incompatible shapes raise NumPy's own broadcasting ValueError, untouched (SPEC 2.3).
    """
    raw = {"S": S, "K": K, "T": T, "r": r, "q": q, "sigma": sigma}
    arrays = {name: _as_float_array(name, value) for name, value in raw.items()}

    for name, values in arrays.items():
        _require(name, values, np.isfinite(values), "must be finite")
    _require("S", arrays["S"], arrays["S"] > 0.0, "must be > 0")
    _require("K", arrays["K"], arrays["K"] > 0.0, "must be > 0")
    _require("T", arrays["T"], arrays["T"] >= 0.0, "must be >= 0 (years)")
    _require("sigma", arrays["sigma"], arrays["sigma"] >= 0.0, "must be >= 0")
    if not allow_extreme_vol:
        _require(
            "sigma",
            arrays["sigma"],
            arrays["sigma"] <= MAX_SIGMA_WITHOUT_OVERRIDE,
            "must be <= 5 (0.20 means 20%); pass allow_extreme_vol=True if intended",
        )

    S_b, K_b, T_b, r_b, q_b, sigma_b = np.broadcast_arrays(*arrays.values())
    all_scalar = all(np.ndim(v) == 0 and not isinstance(v, np.ndarray) for v in raw.values())
    return MarketInputs(S=S_b, K=K_b, T=T_b, r=r_b, q=q_b, sigma=sigma_b, all_scalar=all_scalar)


def _as_float_array(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value)
    if array.dtype.kind not in _REAL_DTYPE_KINDS:
        raise PricingInputError(
            f"{name} must be a real number or an array of real numbers; got dtype {array.dtype}"
        )
    return array.astype(np.float64, copy=False)


def _require(name: str, values: FloatArray, ok: BoolArray, rule: str) -> None:
    if np.all(ok):
        return
    index = tuple(int(i) for i in np.argwhere(~ok)[0])
    where = f" at index {index}" if values.ndim > 0 else ""
    raise PricingInputError(f"{name} {rule}; got {float(values[index])!r}{where}")


_Greek = TypeVar("_Greek", float, FloatArray)


def per_vol_point(vega: _Greek) -> _Greek:
    """Vega per volatility point (sigma + 0.01) from vega per 1.00 of sigma (SPEC 2.2)."""
    return vega / VOL_POINTS_PER_UNIT


def per_calendar_day(theta: _Greek) -> _Greek:
    """Theta per calendar day from theta per year (SPEC 2.2)."""
    return theta / DAYS_PER_YEAR


def per_percentage_point(rho: _Greek) -> _Greek:
    """Rho (or phi) per percentage point of rate from rho per 1.00 (SPEC 2.2)."""
    return rho / PERCENTAGE_POINTS_PER_UNIT
