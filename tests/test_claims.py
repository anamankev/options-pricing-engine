"""The claims ledger (docs/claims.toml): every number in the README's Status and Results sections
matches the result file it comes from, and no number there is unaccounted for."""

import dataclasses

import pytest
from verify.ledger import README, Claim, check, compact, load_claims, uncovered_numbers

CLAIMS = load_claims()


@pytest.mark.parametrize("claim", CLAIMS, ids=[c.id for c in CLAIMS])
def test_readme_number_matches_its_source(claim: Claim) -> None:
    outcome = check(claim, README.read_text())
    assert outcome.passed, f"{claim.id}: {outcome.detail}"


def test_every_number_in_the_results_is_a_claim_or_a_label() -> None:
    assert uncovered_numbers() == []


def test_claim_ids_are_unique() -> None:
    ids = [c.id for c in CLAIMS]
    assert len(ids) == len(set(ids))


def test_a_mistyped_readme_number_is_caught() -> None:
    # The ledger must fail when the README and the source disagree, not only when both agree.
    claim = next(c for c in CLAIMS if c.id == "validation.mpmath_worst")
    edited = README.read_text().replace(claim.readme, claim.readme.replace("0.76", "0.67"))
    assert not check(dataclasses.replace(claim, readme=claim.readme), edited).passed
    assert not check(
        dataclasses.replace(claim, shown="0.67", readme=claim.readme.replace("0.76", "0.67")),
        edited,
    ).passed


def test_an_unlisted_number_is_caught() -> None:
    text = README.read_text().replace("## Usage", "A new result: 42 widgets.\n\n## Usage", 1)
    assert ("42", "A new result: 42 widgets.") in uncovered_numbers(text)


@pytest.mark.parametrize(
    ("python", "readme"),
    [("4.8e+07", "4.8e7"), ("8.4e-12", "8.4e-12"), ("1e+06", "1e6"), ("0.83", "0.83")],
)
def test_scientific_notation_is_written_as_the_readme_does(python: str, readme: str) -> None:
    assert compact(python) == readme
