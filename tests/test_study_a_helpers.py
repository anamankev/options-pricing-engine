"""Helpers of experiments/study_a_iv.py (definitions registered in docs/notes/study_a.md,
section 3)."""

import math

import pytest
from experiments.study_a_iv import errors, grid_axes, outcome, truth, well_posed

from pricing import black_scholes as bs
from pricing.contracts import OptionType

EPS = 2.220446049250313e-16
# truth() rounds a 50-digit value once: within half an ulp of the float64 BSM price up to that
# price's own rounding (Phase 1 model, a few eps relative for this at-the-money case).
TRUTH_RTOL = 64 * EPS


def test_grid_matches_spec_10() -> None:
    xs, ss = grid_axes(
        {
            "x": {"start": -6.0, "stop": 6.0, "step": 0.25},
            "s": {"low": 1e-4, "high": 4.0, "count": 60},
        }
    )
    assert len(xs) == 49
    assert xs[0] == -6.0
    assert xs[-1] == 6.0
    assert len(ss) == 60
    assert ss[0] == pytest.approx(1e-4, rel=TRUTH_RTOL)
    assert ss[-1] == pytest.approx(4.0, rel=TRUTH_RTOL)


def test_truth_matches_bsm_and_kappa_definition() -> None:
    # T = 1, r = q = 0: the undiscounted Black price equals BSM with S = F.
    t = truth(1, 100.0, 100.0, 0.2, 50)
    assert t["c_target"] == pytest.approx(
        float(bs.price(OptionType.CALL, S=100.0, K=100.0, T=1.0, r=0.0, q=0.0, sigma=0.2)),
        rel=TRUTH_RTOL,
    )
    # At the money kappa = c / (s F phi(s/2)); for small s c ~ F s phi(0), so kappa -> 1.
    assert truth(1, 100.0, 100.0, 1e-4, 50)["kappa"] == pytest.approx(1.0, rel=1e-6)


def test_well_posed_threshold() -> None:
    assert well_posed(EPS * 200.0, 100.0, 200.0)
    assert not well_posed(0.5 * EPS * 200.0, 100.0, 200.0)
    assert not well_posed(0.0, 100.0, 200.0)


def test_error_metrics() -> None:
    s = 0.25
    e = errors(s + 2 * math.ulp(s), s, kappa=1e-6)
    assert e["e_ulp"] == pytest.approx(2.0, rel=TRUTH_RTOL)
    # Registered floor: kappa below 1 is replaced by 1 in e_norm, not in SPEC's raw version.
    assert e["e_norm"] == pytest.approx(e["e_rel"] / EPS, rel=TRUTH_RTOL)
    assert e["e_norm_spec"] == pytest.approx(e["e_rel"] / (1e-6 * EPS), rel=TRUTH_RTOL)
    assert errors(None, s, 1.0) == {
        "e_rel": None,
        "e_ulp": None,
        "e_norm": None,
        "e_norm_spec": None,
    }


def test_outcome_classes() -> None:
    assert outcome("lets_be_rational", lambda: 0.0, 1.0)["outcome"] == "AT_LOWER_BOUND"
    assert outcome("lets_be_rational", lambda: 0.3, 1.0)["s_hat"] == 0.3

    def infeasible() -> None:
        from pricing.errors import InfeasiblePriceError

        raise InfeasiblePriceError("test")

    assert outcome("brent", infeasible, 1.0)["outcome"] == "INFEASIBLE"


def test_informative_region_definition() -> None:
    # Post hoc (docs/notes/study_a.md, section 4): a normal float target and kappa eps < 1.
    from experiments.study_a_iv import informative

    assert informative(1e-20, kappa=1e-6)  # tiny OTM price, well conditioned
    assert not informative(50.0, kappa=1e17)  # ITM price whose time value is lost (kappa eps > 1)
    assert not informative(1e-310, kappa=1e-6)  # subnormal target has lost precision
    assert not informative(0.0, kappa=1e-6)
