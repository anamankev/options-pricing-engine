"""The README's Usage examples run as written, and each value quoted in a trailing comment is what
the code returns."""

import ast
import re

import pytest
from verify.ledger import README

BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)
# A trailing comment that starts with a number: "# 4.759422392871536" or "# 1.99047..." (prefix).
QUOTED = re.compile(r"#\s*(?P<number>-?\d+\.\d+)(?P<prefix>\.\.\.)?")


def _quoted_values(source: str) -> list[tuple[ast.expr, str, bool]]:
    """(expression, quoted number, is a truncated prefix) for each expression statement whose last
    line ends in a numeric comment."""
    lines = source.splitlines()
    found = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Expr) and node.end_lineno is not None:
            match = QUOTED.search(lines[node.end_lineno - 1])
            if match:
                found.append((node.value, match["number"], match["prefix"] is not None))
    return found


def test_readme_has_examples() -> None:
    assert len(BLOCKS) >= 3


@pytest.mark.parametrize("source", BLOCKS, ids=[f"block{i}" for i in range(len(BLOCKS))])
def test_readme_example_runs_and_matches_its_comments(source: str) -> None:
    namespace: dict[str, object] = {}
    exec(compile(source, "README.md", "exec"), namespace)
    for expression, quoted, is_prefix in _quoted_values(source):
        value = eval(compile(ast.Expression(expression), "README.md", "eval"), namespace)
        if is_prefix:  # "1.99047..." quotes the leading digits
            assert repr(float(value)).startswith(quoted), (ast.unparse(expression), value)
        else:  # quoted to as many decimals as written
            decimals = len(quoted.split(".")[1])
            assert f"{float(value):.{decimals}f}" == quoted, (ast.unparse(expression), value)
