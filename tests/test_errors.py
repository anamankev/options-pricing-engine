"""The exception hierarchy matches SPEC 2.4, so callers can catch by kind or by base class."""

import pytest

from pricing.errors import (
    ConvergenceError,
    InfeasiblePriceError,
    PricingInputError,
    TreeParameterError,
    UndefinedGreekError,
)


@pytest.mark.parametrize(
    ("error", "base"),
    [
        (PricingInputError, ValueError),
        (UndefinedGreekError, ValueError),
        (TreeParameterError, ValueError),
        (InfeasiblePriceError, ValueError),
        (ConvergenceError, RuntimeError),
    ],
)
def test_error_base_classes(error: type[Exception], base: type[Exception]) -> None:
    assert issubclass(error, base)


def test_input_and_greek_errors_are_distinct() -> None:
    # A caller handling an undefined Greek must not also swallow invalid inputs.
    assert not issubclass(UndefinedGreekError, PricingInputError)
    assert not issubclass(PricingInputError, UndefinedGreekError)
