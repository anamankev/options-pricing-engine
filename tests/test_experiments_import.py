"""Experiment modules import without the oracle extra (QuantLib, lets_be_rational, vollib).

CI's fast job installs only the dev extra, and pytest imports these modules when it collects the
helper tests. A top-level oracle import once broke collection there (Phase 4); this test blocks the
oracle packages in a fresh interpreter, so the failure shows up locally too.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("quantlib_reconciliation", "mc_coverage", "greek_bumps", "study_a_iv")
BLOCKED = ("QuantLib", "lets_be_rational", "py_lets_be_rational", "vollib")


def test_experiment_modules_import_without_oracle_packages() -> None:
    script = (
        "import sys\n"
        + "".join(f"sys.modules[{name!r}] = None\n" for name in BLOCKED)
        + "".join(f"import experiments.{module}\n" for module in MODULES)
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
