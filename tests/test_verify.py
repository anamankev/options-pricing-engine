"""`python -m verify`: how it compares regenerated results, reports, and skips without QuantLib."""

import pytest
from verify import __main__ as cli
from verify import checks
from verify.checks import FAIL, PASS, SKIP, Result


def test_identical_results_have_no_differences() -> None:
    summary = {"environment": {"commit": "a"}, "stats": {"x": 1.5, "n": 3}, "verdicts": {"A": True}}
    assert cli.compare_summaries("validation", summary, summary) == []


def test_a_changed_number_is_reported_with_both_values() -> None:
    old = {"stats": {"worst": 0.76}}
    new = {"stats": {"worst": 0.81}}
    assert cli.compare_summaries("validation", old, new) == [
        "stats.worst: committed 0.76, now 0.81"
    ]


def test_rounding_level_differences_are_not_reported() -> None:
    old = {"stats": {"worst": 0.7595878789861512}}
    new = {"stats": {"worst": 0.7595878789861512 * (1 + 1e-12)}}
    assert cli.compare_summaries("validation", old, new) == []


def test_provenance_and_timings_are_never_compared() -> None:
    old = {"environment": {"commit": "a"}, "time_median_us": 5.0, "timing_ms": {"crr": 1.0}}
    new = {"environment": {"commit": "b"}, "time_median_us": 9.0, "timing_ms": {"crr": 2.0}}
    assert cli.compare_summaries("study_a_iv", old, new) == []


def test_benchmarks_compare_verdicts_only() -> None:
    old = {"bsm_per_option_ratio": {"price": 1095.0}, "verdicts": {"P1": False}}
    new = {"bsm_per_option_ratio": {"price": 900.0}, "verdicts": {"P1": True}}
    assert cli.compare_summaries("benchmarks", old, new) == [
        "verdicts.P1: committed False, now True"
    ]


def test_a_missing_key_is_a_difference() -> None:
    assert cli.compare_summaries("validation", {"a": 1}, {}) == ["a: committed 1, now None"]


def test_oracle_checks_skip_with_the_reason_when_quantlib_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checks, "oracle_available", lambda: False)
    for check in (checks.european_against_quantlib, checks.trees_against_quantlib):
        result = check()
        assert result.status == SKIP
        assert "QuantLib not installed" in result.detail


def test_report_counts_and_flags_failures() -> None:
    results = [
        Result("first", PASS, "ok"),
        Result("second", FAIL, "off by 2", "<= 1", "the detail"),
        Result("third", SKIP, "", "", "why it was skipped"),
    ]
    text = cli.render(results, markdown=False)
    assert "[FAIL] second" in text
    assert "Summary: 1 passed, 1 failed, 1 skipped" in text
    assert "Something failed" in text
    table = cli.render(results, markdown=True)
    assert "| FAIL | second | off by 2 | <= 1 |" in table


def test_quick_rechecks_pass_on_the_committed_code() -> None:
    # The same rechecks `python -m verify` runs (the test suite and ledger have their own tests).
    for check in checks.QUICK_CHECKS:
        result = check()
        assert result.status in (PASS, SKIP), (result.name, result.measured, result.detail)


def test_study_b_runtime_ranking_is_not_compared() -> None:
    # B4 ranks methods by error at a fixed runtime budget, so it can change with machine load.
    old = {"verdicts": {"B1a": True, "B4": True}, "stats": {"B4_bbsr_share": 0.85}}
    new = {"verdicts": {"B1a": True, "B4": False}, "stats": {"B4_bbsr_share": 0.70}}
    assert cli.compare_summaries("study_b_trees", old, new) == []
    assert "B4" in cli.NOT_COMPARED["study_b_trees"]


def test_full_rerun_covers_every_experiment_config() -> None:
    # A new experiment must be added to verify's list, or --full would silently leave it out.
    configs = {p.stem for p in (cli.ROOT / "experiments" / "configs").glob("*.toml")}
    assert {config for _, _, config, _, _ in cli.EXPERIMENTS} == configs
