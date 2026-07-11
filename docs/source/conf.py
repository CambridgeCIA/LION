"""Sphinx configuration for the LION documentation."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import sys

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList


ROOT = Path(__file__).resolve().parents[2]
DOCS_SOURCE = Path(__file__).resolve().parent
BUILD_ROOT = DOCS_SOURCE.parent / "_build"

sys.path.insert(0, str(ROOT))

# LION resolves its data roots while modules are imported.  Documentation
# builds do not access datasets, but autodoc still needs valid, isolated paths.
os.environ.setdefault("LION_DATA_PATH", str(BUILD_ROOT / "data"))
os.environ.setdefault("LION_EXPERIMENTS_PATH", str(BUILD_ROOT / "experiments"))
os.environ.setdefault("MPLCONFIGDIR", str(BUILD_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(BUILD_ROOT / "cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

project = "LION"
copyright = "2026, Cambridge Computational Imaging Group and contributors"
author = "LION contributors"
release = "development"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
# Object entries remain searchable and linkable inside expandable API panels,
# but do not flood Furo's page-level table of contents while panels are closed.
toc_object_entries = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
myst_heading_anchors = 3

# Repository READMEs are included verbatim so that their GitHub-facing guides
# remain the single source of truth.  Some contain links whose targets live
# outside the Sphinx source tree; those links remain useful on GitHub but
# cannot be resolved as Sphinx cross-references.
suppress_warnings = ["myst.xref_missing"]


class SourceApiSummary(Directive):
    """Render public Python symbols without importing the target module.

    This complements autodoc for legacy modules whose import-time behaviour
    requires optional packages, datasets, or a CUDA device. The summary is
    generated from Python's abstract syntax tree, so it stays aligned with the
    source while keeping documentation builds safe and deterministic.
    """

    required_arguments = 1
    final_argument_whitespace = False
    has_content = False

    @staticmethod
    def _signature(item: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
        return f"{prefix}{item.name}({ast.unparse(item.args)})"

    @staticmethod
    def _description(item: ast.AST) -> str:
        """Return an object's docstring as safe display text."""

        return ast.get_docstring(item, clean=True) or "No docstring is available."

    def _symbol_item(self, item: ast.AST, label: str) -> nodes.list_item:
        """Build one documented symbol entry."""

        entry = nodes.list_item()
        signature = nodes.paragraph()
        signature += nodes.literal(text=label)
        entry += signature
        for paragraph_text in self._description(item).split("\n\n"):
            entry += nodes.paragraph(text=" ".join(paragraph_text.splitlines()))
        return entry

    def run(self) -> list[nodes.Node]:
        module = self.arguments[0].strip()
        relative = Path(*module.split("."))
        source = ROOT / relative.with_suffix(".py")
        if not source.is_file():
            source = ROOT / relative / "__init__.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

        result: list[nodes.Node] = [nodes.rubric(text=module.rsplit(".", 1)[-1])]
        module_docstring = ast.get_docstring(tree, clean=True)
        if module_docstring:
            for paragraph_text in module_docstring.split("\n\n"):
                result.append(
                    nodes.paragraph(text=" ".join(paragraph_text.splitlines()))
                )
        entries = nodes.bullet_list()
        for item in tree.body:
            if (
                item.name.startswith("_")
                if isinstance(
                    item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
                else True
            ):
                continue
            if isinstance(item, ast.ClassDef):
                label = f"class {item.name}"
                methods = [
                    child
                    for child in item.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not child.name.startswith("_")
                ]
                entry = self._symbol_item(item, label)
                if methods:
                    method_entries = nodes.bullet_list()
                    for method in methods:
                        method_entries += self._symbol_item(
                            method, self._signature(method)
                        )
                    entry += method_entries
            else:
                label = self._signature(item)
                entry = self._symbol_item(item, label)
            entries += entry
        if entries.children:
            result.append(entries)
        else:
            result.append(nodes.paragraph(text="No public symbols are defined."))
        return result


class ApiModule(Directive):
    """Render one module in a consistent, expandable API panel."""

    required_arguments = 1
    final_argument_whitespace = False
    has_content = False
    option_spec = {
        "source": directives.flag,
        "warning": directives.flag,
        "open": directives.flag,
    }

    def run(self) -> list[nodes.Node]:
        module = self.arguments[0].strip()
        warning = "🚧 " if "warning" in self.options else ""
        lines = [f".. dropdown:: {warning}``{module}``"]
        if "open" in self.options:
            lines.append("   :open:")
        lines.extend(["", "   .. rst-class:: api-module-status", ""])
        if "warning" in self.options:
            lines.extend(
                [
                    "      This module has not yet received a complete narrative and "
                    "docstring audit.",
                    "",
                ]
            )
        if "source" in self.options:
            lines.append(f"   .. sourceautosummary:: {module}")
        else:
            lines.extend(
                [
                    f"   .. automodule:: {module}",
                    "      :members:",
                    "      :show-inheritance:",
                ]
            )

        container = nodes.container(classes=["api-module-wrapper"])
        self.state.nested_parse(StringList(lines), self.content_offset, container)
        return [container]


def setup(app):
    """Register documentation helpers local to LION."""

    app.add_directive("sourceautosummary", SourceApiSummary)
    app.add_directive("apimodule", ApiModule)


html_theme = "furo"
html_title = "LION Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/THartigan/LION/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
