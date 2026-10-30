"""Smoke tests: the package installs, imports, reports one version, and ships its type marker."""

from importlib.metadata import version
from importlib.resources import files

import pricing


def test_version_matches_installed_metadata() -> None:
    # pyproject.toml reads its version from pricing.__version__ at install time. A mismatch
    # means the installed copy is stale: rerun `pip install -e ".[dev]"`.
    assert pricing.__version__ == version("options-pricing-engine")


def test_package_ships_py_typed_marker() -> None:
    # Without py.typed, mypy treats the package as untyped for anyone who imports it (PEP 561).
    assert files("pricing").joinpath("py.typed").is_file()
