# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Every public class and function has a docstring (``contributor-guide.md``,
"Docstrings"), in the folders documented so far: one more with each
documentation PR, until it is all of LION."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTED = ["LION/optimizers", "LION/models", "LION/data_loaders"]


def _undocumented(path: Path) -> list[str]:
    missing = []

    def walk(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not child.name.startswith("_") and ast.get_docstring(child) is None:
                    missing.append(prefix + child.name)
                if isinstance(child, ast.ClassDef):
                    walk(child, child.name + ".")

    walk(ast.parse(path.read_text(encoding="utf-8")))
    return missing


FILES = sorted(
    path.relative_to(ROOT)
    for folder in DOCUMENTED
    for path in (ROOT / folder).rglob("*.py")
)


@pytest.mark.parametrize("path", FILES, ids=str)
def test_public_definitions_have_docstrings(path):
    assert _undocumented(ROOT / path) == []
