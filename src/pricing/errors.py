"""Exception hierarchy (SPEC 2.4).

Invalid states raise one of these; nothing is clipped into range and nothing returns a silent NaN.
Each subclasses ValueError or RuntimeError, so generic handlers still work.
"""


class PricingInputError(ValueError):
    """Invalid input: NaN or inf, S <= 0, K <= 0, T < 0, sigma < 0, or an unsupported contract."""


class UndefinedGreekError(ValueError):
    """A Greek requested where it does not exist, e.g. delta at T = 0 with S = K (SPEC 3.4)."""


class TreeParameterError(ValueError):
    """Invalid binomial-tree probability or grid (SPEC 4.1)."""


class InfeasiblePriceError(ValueError):
    """Implied volatility requested for a price outside the no-arbitrage bounds (SPEC 6.2)."""


class ConvergenceError(RuntimeError):
    """An implied-volatility wrapper's solver returned converged=False (SPEC 6.3)."""
