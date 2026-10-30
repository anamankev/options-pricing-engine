"""Monte Carlo (SPEC 5). Deterministic tests check reproducibility, validation and the statistics
plumbing by rebuilding the random streams by hand. Tests marked `statistical` check behaviour that
holds with high probability, with fixed seeds (SPEC 9.2)."""

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from pricing import black_scholes as bs
from pricing.contracts import OptionType
from pricing.errors import PricingInputError
from pricing.monte_carlo import (
    Z_95,
    MCMethod,
    monte_carlo,
    summarize,
    variance_reduction_factor,
)

CALL, PUT = OptionType.CALL, OptionType.PUT
EPS = float(np.finfo(np.float64).eps)
ATM: dict[str, Any] = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2}

# Rebuilt-by-hand statistics repeat the library's float operations on the same draws, but the
# order of a few products can differ: 8 ulps covers it.
REBUILD_RTOL = 8 * EPS
# Statistical checks (fixed seeds). |estimate - BSM| <= 4 SE fails with probability 6e-5 per
# check if the estimator is unbiased and its SE correct.
Z_LIMIT = 4.0
# SE ratio between n = 1e3 and 1e5 should be 10. The sample sd at n = 1e3 has relative noise of a
# few percent for these payoffs, so [8.5, 11.5] is a ~5 sigma band.
SE_RATIO_BAND = (8.5, 11.5)


def _child_rng(seed: int, index: int) -> np.random.Generator:
    # SPEC 2.3 decision recorded in monte_carlo.py: main stream spawn key (0,), pilot (1,).
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index,)))


def _discounted_payoffs(case: dict[str, Any], theta: int, z: np.ndarray) -> np.ndarray:
    # SPEC 5.1, written independently of the library.
    S, K, T, r, q, sigma = (case[k] for k in ("S", "K", "T", "r", "q", "sigma"))
    terminal = S * np.exp((r - q - sigma**2 / 2) * T + sigma * math.sqrt(T) * z)
    return math.exp(-r * T) * np.maximum(theta * (terminal - K), 0.0)


# --- Reproducibility and validation (deterministic) ----------------------------------------


def test_same_seed_gives_identical_results() -> None:
    a = monte_carlo(CALL, **ATM, n_paths=1000, seed=7)
    assert a == monte_carlo(CALL, **ATM, n_paths=1000, seed=7)
    assert a != monte_carlo(CALL, **ATM, n_paths=1000, seed=8)


def test_seed_sequence_input_is_not_mutated() -> None:
    root = np.random.SeedSequence(123)
    first = monte_carlo(
        CALL, **ATM, n_paths=1000, seed=root, method=MCMethod.CONTROL_VARIATE, pilot_paths=200
    )
    second = monte_carlo(
        CALL, **ATM, n_paths=1000, seed=root, method=MCMethod.CONTROL_VARIATE, pilot_paths=200
    )
    assert first == second
    assert root.n_children_spawned == 0
    assert first.seed_entropy == 123
    assert first.seed_spawn_key == ()


@pytest.mark.parametrize("bad", [-1, 1.5, True, "7", None])
def test_seed_must_be_explicit_non_negative_int_or_seed_sequence(bad: object) -> None:
    with pytest.raises(PricingInputError, match="seed"):
        monte_carlo(CALL, **ATM, n_paths=1000, seed=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_paths": 1}, "n_paths"),
        ({"n_paths": 1000.0}, "n_paths"),
        ({"n_paths": 1001, "method": MCMethod.ANTITHETIC}, "even"),
        ({"n_paths": 1000, "method": MCMethod.CONTROL_VARIATE}, "pilot_paths"),
        ({"n_paths": 1000, "method": "plain"}, "MCMethod"),
    ],
)
def test_invalid_configurations_raise(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(PricingInputError, match=message):
        monte_carlo(CALL, **ATM, seed=1, **kwargs)


@pytest.mark.parametrize("override", [{"T": 0.0}, {"sigma": 0.0}, {"S": np.array([1.0, 2.0])}])
def test_degenerate_or_array_inputs_raise(override: dict[str, Any]) -> None:
    with pytest.raises(PricingInputError):
        monte_carlo(CALL, **{**ATM, **override}, n_paths=1000, seed=1)


def test_result_is_frozen_and_records_inputs() -> None:
    res = monte_carlo(PUT, **ATM, n_paths=100, seed=1)
    assert res.inputs == {"option_type": "PUT", **ATM}
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.estimate = 0.0  # type: ignore[misc]


# --- Statistics plumbing (deterministic) ---------------------------------------------------


def test_summarize_uses_ddof_one() -> None:
    mean, sd, se = summarize(np.array([1.0, 2.0, 3.0, 4.0]))
    assert mean == 2.5
    assert sd == pytest.approx(math.sqrt(5 / 3), rel=REBUILD_RTOL)  # sum of squares 5, n - 1 = 3
    assert se == pytest.approx(sd / 2, rel=REBUILD_RTOL)


def test_interval_is_the_normal_approximation() -> None:
    # SPEC 5.1: 95% normal-approximation interval; N^-1(0.975) = 1.959963984540054.
    assert pytest.approx(1.959963984540054, rel=REBUILD_RTOL) == Z_95
    res = monte_carlo(CALL, **ATM, n_paths=1000, seed=3)
    assert res.ci_low == res.estimate - Z_95 * res.std_error
    assert res.ci_high == res.estimate + Z_95 * res.std_error


def test_plain_estimate_matches_hand_rebuilt_sample() -> None:
    res = monte_carlo(CALL, **ATM, n_paths=5000, seed=11)
    y = _discounted_payoffs(ATM, 1, _child_rng(11, 0).standard_normal(5000))
    assert res.estimate == pytest.approx(y.mean(), rel=REBUILD_RTOL)
    assert res.std_error == pytest.approx(y.std(ddof=1) / math.sqrt(5000), rel=REBUILD_RTOL)
    assert res.n_payoff_evals == res.n_paths == 5000


def test_antithetic_error_comes_from_pair_averages() -> None:
    # SPEC 5.2: the n / 2 pair averages are the i.i.d. unit; the 2n payoffs are not independent.
    res = monte_carlo(CALL, **ATM, n_paths=5000, seed=11, method=MCMethod.ANTITHETIC)
    z = _child_rng(11, 0).standard_normal(2500)
    pairs = 0.5 * (_discounted_payoffs(ATM, 1, z) + _discounted_payoffs(ATM, 1, -z))
    assert res.std_error == pytest.approx(pairs.std(ddof=1) / math.sqrt(2500), rel=REBUILD_RTOL)
    # Treating the 5000 payoffs as independent would report a different (wrong) error.
    members = np.concatenate([_discounted_payoffs(ATM, 1, z), _discounted_payoffs(ATM, 1, -z)])
    assert abs(members.std(ddof=1) / math.sqrt(5000) / res.std_error - 1) > 0.05
    assert res.n_payoff_evals == 5000


def test_control_variate_beta_comes_from_the_independent_pilot() -> None:
    res = monte_carlo(
        CALL, **ATM, n_paths=5000, seed=11, method=MCMethod.CONTROL_VARIATE, pilot_paths=800
    )
    z = _child_rng(11, 1).standard_normal(800)
    y = _discounted_payoffs(ATM, 1, z)
    x = (
        math.exp(-ATM["r"] * ATM["T"])
        * ATM["S"]
        * np.exp((ATM["r"] - ATM["q"] - ATM["sigma"] ** 2 / 2) * ATM["T"] + ATM["sigma"] * z)
    )
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    assert res.beta == pytest.approx(beta, rel=REBUILD_RTOL)  # observed 2 eps over 50 seeds
    assert res.n_payoff_evals == 5800  # SPEC 5.3: pilot cost counts
    assert res.pilot_paths == 800


def test_same_sample_beta_costs_no_pilot() -> None:
    res = monte_carlo(
        CALL, **ATM, n_paths=5000, seed=11, method=MCMethod.CONTROL_VARIATE, same_sample_beta=True
    )
    assert res.pilot_paths is None
    assert res.n_payoff_evals == 5000


def test_variance_reduction_factor_definition() -> None:
    # SPEC 5.4: VRF = (SE_plain^2 cost_plain) / (SE_other^2 cost_other).
    plain = monte_carlo(CALL, **ATM, n_paths=1000, seed=1)
    other = dataclasses.replace(plain, std_error=plain.std_error / 2, n_payoff_evals=2000)
    assert variance_reduction_factor(plain, other) == pytest.approx(2.0, rel=REBUILD_RTOL)


# --- Statistical behaviour (fixed seeds) ---------------------------------------------------

CASES = [
    {**ATM, "K": 80.0},  # in the money
    ATM,
    {**ATM, "K": 130.0},  # out of the money
]


@pytest.mark.statistical
@pytest.mark.parametrize("method", list(MCMethod))
@pytest.mark.parametrize("option_type", [CALL, PUT])
def test_estimates_agree_with_black_scholes(method: MCMethod, option_type: OptionType) -> None:
    for seed, case in enumerate(CASES):
        res = monte_carlo(
            option_type, **case, n_paths=200_000, seed=100 + seed, method=method, pilot_paths=20_000
        )
        z = (res.estimate - bs.price(option_type, **case)) / res.std_error
        assert abs(z) <= Z_LIMIT, (case, z)


@pytest.mark.statistical
def test_variance_reduction_beats_plain_monte_carlo() -> None:
    # Antithetic: the call payoff is monotone in Z, so the pair is negatively correlated.
    # Control variate: the payoff is strongly correlated with S_T in and at the money.
    for seed, case in enumerate(CASES[:2]):
        plain = monte_carlo(CALL, **case, n_paths=100_000, seed=200 + seed)
        anti = monte_carlo(
            CALL, **case, n_paths=100_000, seed=300 + seed, method=MCMethod.ANTITHETIC
        )
        cv = monte_carlo(
            CALL,
            **case,
            n_paths=100_000,
            seed=400 + seed,
            method=MCMethod.CONTROL_VARIATE,
            pilot_paths=10_000,
        )
        assert variance_reduction_factor(plain, anti) > 1.0
        assert variance_reduction_factor(plain, cv) > 1.0


@pytest.mark.statistical
def test_standard_error_scales_as_one_over_root_n() -> None:
    small = monte_carlo(CALL, **ATM, n_paths=1_000, seed=500)
    large = monte_carlo(CALL, **ATM, n_paths=100_000, seed=501)
    low, high = SE_RATIO_BAND
    assert low <= small.std_error / large.std_error <= high
