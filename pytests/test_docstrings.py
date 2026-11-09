# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Every public class and function in LION has a docstring
(``contributor-guide.md``, "Docstrings")."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def test_public_definitions_have_docstrings():
    missing = {
        str(path.relative_to(ROOT)): names
        for path in sorted((ROOT / "LION").rglob("*.py"))
        if (names := _undocumented(path))
    }
    assert missing == {}
