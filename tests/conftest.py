"""Shared pytest configuration.

Hypothesis profiles, selected with the HYPOTHESIS_PROFILE environment variable:

- dev (default): random exploration. Failing examples are saved in .hypothesis/ and replayed
  first on the next run; print_blob prints a string that reproduces the failure exactly.
- ci: derandomize=True, so every CI run tests the same examples (no hidden seed). A CI
  failure reproduces locally with HYPOTHESIS_PROFILE=ci.

deadline=None in both: Hypothesis's default 200 ms per-example deadline makes timing-sensitive
tests (trees at large N) flaky on shared CI runners. Speed is measured by benchmarks, not here.
"""

import os

from hypothesis import settings

settings.register_profile("dev", deadline=None, print_blob=True)
settings.register_profile("ci", deadline=None, print_blob=True, derandomize=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
