"""The claims ledger: every number the README states, checked against the file it comes from.

docs/claims.toml lists one entry per number. Each entry names
- `readme`: the exact phrase in README.md that contains the number;
- `shown`: the number as the README prints it;
- `source`: where the value comes from (grammar below);
- `format`: how the value is rounded for display (a Python format spec);
- optionally `scale` (multiply before formatting, e.g. 100 for a percentage), `aggregate`
  (min, max, median, len) when the source yields several values, and `kind`:
  "equals" (default: the formatted value must read exactly `shown`), "at_most" (the value must
  not exceed `shown`, for claims such as "within 2e-3"), or "true" (a recorded check passed).

Source grammar:
- `json:PATH#a.b.c` walks keys. A segment `*` takes every value of a mapping; a segment
  `name[k=v,k2=a|b]` keeps the list items whose field k equals v (or any of a|b).
- `toml:PATH#a.b` the same, for TOML files.
- `csv:PATH[k=v,...].column` the column of the matching rows.
- `py:module:NAME` a constant defined in code (for example a tolerance).
- `count:tests` the number of tests in the repository's tracked test files.
- `count:glob:PATTERN` the number of files matching PATTERN.
"""

from __future__ import annotations

import csv
import importlib
import json
import re
import statistics
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "claims.toml"
README = ROOT / "README.md"
_FILTER = re.compile(r"^(?P<name>[^\[]+)\[(?P<conds>[^\]]+)\]$")


@dataclass(frozen=True)
class Claim:
    id: str
    readme: str
    shown: str
    source: str
    format: str = "g"
    scale: float = 1.0
    aggregate: str | None = None
    kind: str = "equals"


@dataclass(frozen=True)
class Outcome:
    claim: Claim
    value: Any  # the value from the source, after aggregation and scaling
    displayed: str  # the value as the README would print it
    in_readme: bool
    passed: bool
    detail: str


def load_claims(path: Path = LEDGER) -> list[Claim]:
    with path.open("rb") as f:
        return [Claim(**entry) for entry in tomllib.load(f)["claim"]]


def compact(text: str) -> str:
    """Python's 4.8e+07 / 8.4e-12 as the README writes them: 4.8e7 / 8.4e-12."""
    mantissa, sep, exponent = text.partition("e")
    if not sep:
        return text
    sign = "-" if exponent.startswith("-") else ""
    return f"{mantissa}e{sign}{exponent.lstrip('+-').lstrip('0') or '0'}"


def _matches(item: dict[str, Any], conds: str) -> bool:
    for cond in conds.split(","):
        key, _, wanted = cond.partition("=")
        options = wanted.split("|")
        actual = item.get(key.strip())
        if not any(_equal(actual, option.strip()) for option in options):
            return False
    return True


def _equal(actual: Any, wanted: str) -> bool:
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        try:
            return float(actual) == float(wanted)
        except ValueError:
            return False
    return str(actual) == wanted


def _walk(data: Any, path: str) -> Any:
    """Follow a dotted path; returns a list once a wildcard or filter has been applied."""
    current: Any = data
    many = False
    for segment in path.split("."):
        if many:
            current = [_step(item, segment) for item in current]
            current = [x for sub in current for x in (sub if isinstance(sub, list) else [sub])]
        else:
            current = _step(current, segment)
            many = isinstance(current, list) and (segment == "*" or "[" in segment)
    return current


def _step(node: Any, segment: str) -> Any:
    if segment == "*":
        return list(node.values())
    match = _FILTER.match(segment)
    if match:
        items = node[match["name"]]
        return [item for item in items if _matches(item, match["conds"])]
    return node[segment]


def _count_tests() -> int:
    tracked = subprocess.run(
        ["git", "ls-files", "tests/test_*.py"], capture_output=True, text=True, cwd=ROOT
    ).stdout.split()
    targets = tracked or [str(p.relative_to(ROOT)) for p in ROOT.glob("tests/test_*.py")]
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *targets],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout
    found = re.search(r"(\d+) tests? collected", out)
    if found is None:
        raise RuntimeError(f"could not count tests:\n{out[-2000:]}")
    return int(found.group(1))


def resolve(source: str) -> Any:
    kind, _, rest = source.partition(":")
    if kind in ("json", "toml"):
        file, _, path = rest.partition("#")
        text = (ROOT / file).read_text()
        data = json.loads(text) if kind == "json" else tomllib.loads(text)
        return _walk(data, path)
    if kind == "csv":
        match = re.match(r"^(?P<file>[^\[]+)\[(?P<conds>[^\]]+)\]\.(?P<column>\w+)$", rest)
        if match is None:
            raise ValueError(f"bad csv source: {source}")
        with (ROOT / match["file"]).open(newline="") as f:
            rows = [row for row in csv.DictReader(f) if _matches(row, match["conds"])]
        return [float(row[match["column"]]) for row in rows]
    if kind == "py":
        module, _, name = rest.partition(":")
        return getattr(importlib.import_module(module), name)
    if kind == "count":
        if rest == "tests":
            return _count_tests()
        if rest.startswith("glob:"):
            return len(list(ROOT.glob(rest.removeprefix("glob:"))))
    raise ValueError(f"unknown source kind in {source!r}")


def _aggregate(value: Any, how: str | None) -> Any:
    if how is None:
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"source gave {len(value)} values and no aggregate was named")
            return value[0]
        return value
    if how == "len":
        return len(value)
    functions = {"min": min, "max": max, "median": statistics.median}
    return functions[how](value)


def check(claim: Claim, readme_text: str) -> Outcome:
    value = _aggregate(resolve(claim.source), claim.aggregate)
    if claim.kind == "true":
        displayed = str(value)
        passed = value is True
    else:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = value * claim.scale if claim.scale != 1.0 else value
        displayed = compact(format(value, claim.format)) if claim.format != "s" else str(value)
        if claim.kind == "at_most":
            passed = float(value) <= float(claim.shown)
        else:
            passed = displayed == claim.shown
    in_readme = claim.readme in readme_text and claim.shown in claim.readme
    detail = "" if passed else f"source gives {displayed}, README says {claim.shown}"
    if not in_readme:
        detail = (detail + "; " if detail else "") + "phrase not found in README.md"
    return Outcome(claim, value, displayed, in_readme, passed and in_readme, detail)


@dataclass(frozen=True)
class Label:
    text: str
    readme: str
    reason: str


# A number as written in prose: 1,000,000 or 0.83 or 8.4e-12 or -1.995. Section signs (§6.4),
# exponents (N^1.4 is two claims) and identifiers (R1, A5) are not numbers.
NUMBER = re.compile(r"(?<![\w§.^])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e-?\d+)?(?![\w.])")
CHECKED_SECTIONS = ("## Status", "## Usage")  # every number between these headings is checked


def load_labels(path: Path = LEDGER) -> list[Label]:
    with path.open("rb") as f:
        return [Label(**entry) for entry in tomllib.load(f).get("label", [])]


def uncovered_numbers(readme_text: str | None = None) -> list[tuple[str, str]]:
    """(number, line) pairs in the checked sections that no claim or label accounts for."""
    text = README.read_text() if readme_text is None else readme_text
    start, end = (text.index(heading) for heading in CHECKED_SECTIONS)
    claims, labels = load_claims(), load_labels()
    missing = []
    for line in text[start:end].splitlines():
        covered = {c.shown for c in claims if c.readme in line}
        covered |= {lb.text for lb in labels if lb.readme in line}
        missing += [(n, line) for n in NUMBER.findall(line) if n not in covered]
    return missing


def check_all(*, skip_slow: bool = False) -> list[Outcome]:
    readme_text = README.read_text()
    claims = load_claims()
    return [check(c, readme_text) for c in claims if not (skip_slow and c.source == "count:tests")]
