"""Provenance recorded with every experiment result: commit, dirty flag, versions, platform."""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any

RESULTS_DIR = "experiments/results"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def environment(libraries: tuple[str, ...]) -> dict[str, Any]:
    """Capture before writing any output. `dirty` ignores the results folder, which experiments
    write to themselves."""
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--", ".", f":(exclude){RESULTS_DIR}")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libraries": {lib: version(lib) for lib in libraries},
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
