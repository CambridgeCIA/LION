"""Sphinx configuration for the LION documentation."""

from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
import re
import sys

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx.ext.napoleon.docstring import NumpyDocstring


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
        signature = f"{item.name}({SourceApiSummary._safe_arguments(item.args)})"
        if item.returns is not None:
            signature += f" -> {ast.unparse(item.returns)}"
        return signature

    @staticmethod
    def _description(item: ast.AST) -> str:
        """Return an object's docstring as safe display text."""

        description = (
            ast.get_docstring(item, clean=True) or "No docstring is available."
        )
        return re.sub(r"\[([A-Za-z][\w-]*)\]_", r"[\1]", description)

    @staticmethod
    def _safe_arguments(arguments: ast.arguments) -> str:
        """Render arguments while replacing unparseable f-string defaults."""

        safe = copy.deepcopy(arguments)
        safe.defaults = [
            ast.Constant(value=Ellipsis)
            if any(isinstance(node, ast.JoinedStr) for node in ast.walk(default))
            else default
            for default in safe.defaults
        ]
        safe.kw_defaults = [
            ast.Constant(value=Ellipsis)
            if default is not None
            and any(isinstance(node, ast.JoinedStr) for node in ast.walk(default))
            else default
            for default in safe.kw_defaults
        ]
        return ast.unparse(safe)

    def _formatted_description(self, item: ast.AST, *, what: str) -> list[str]:
        """Convert a source docstring with the same Napoleon rules as autodoc."""

        description = self._description(item)
        if not self._use_napoleon:
            lines: list[str] = []
            for paragraph in description.split("\n\n"):
                lines.extend([" ".join(paragraph.splitlines()), ""])
            return lines
        environment = self.state.document.settings.env
        rendered = str(
            NumpyDocstring(
                description,
                config=environment.config,
                app=environment.app,
                what=what,
            )
        )
        return rendered.splitlines()

    def _append_description(
        self,
        lines: list[str],
        item: ast.AST,
        *,
        indent: str = "",
        what: str,
    ) -> None:
        """Append a Napoleon-formatted docstring as nested reStructuredText."""

        lines.extend(
            f"{indent}{line}" if line else ""
            for line in self._formatted_description(item, what=what)
        )
        lines.append("")

    @staticmethod
    def _constructor_signature(item: ast.ClassDef) -> str:
        """Return a class signature derived from its constructor."""

        constructor = next(
            (
                child
                for child in item.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == "__init__"
            ),
            None,
        )
        if constructor is None:
            return item.name
        arguments = SourceApiSummary._safe_arguments(constructor.args)
        arguments = re.sub(r"^(self|cls)(,\s*)?", "", arguments)
        return f"{item.name}({arguments})"

    @staticmethod
    def _public_assignments(tree: ast.Module) -> list[tuple[str, ast.AST]]:
        """Return public module-level assignments."""

        assignments: list[tuple[str, ast.AST]] = []
        for item in tree.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        assignments.append((target.id, item))
            elif (
                isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and not item.target.id.startswith("_")
            ):
                assignments.append((item.target.id, item))
        return assignments

    def run(self) -> list[nodes.Node]:
        module = self.arguments[0].strip()
        relative = Path(*module.split("."))
        source = ROOT / relative.with_suffix(".py")
        if not source.is_file():
            source = ROOT / relative / "__init__.py"
        source_parts = source.relative_to(ROOT).parts
        self._use_napoleon = "PaDIS-Reproduction" in source_parts
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

        # Do not set ``py:currentmodule`` here. Sphinx viewcode treats that as
        # permission to import the referenced file, which is unsafe for CLI,
        # download, training, and test modules with import-time side effects.
        lines: list[str] = []
        module_docstring = ast.get_docstring(tree, clean=True)
        if module_docstring:
            self._append_description(lines, tree, what="module")

        for name, assignment in self._public_assignments(tree):
            lines.extend([f".. py:data:: {name}", "   :no-index:", ""])
            if (
                isinstance(assignment, ast.AnnAssign)
                and assignment.annotation is not None
            ):
                lines.extend(
                    [f"   **Type:** ``{ast.unparse(assignment.annotation)}``", ""]
                )

        symbol_count = 0
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
                symbol_count += 1
                lines.extend(
                    [
                        f".. py:class:: {self._constructor_signature(item)}",
                        "   :no-index:",
                        "",
                    ]
                )
                self._append_description(lines, item, indent="   ", what="class")
                methods = [
                    child
                    for child in item.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not child.name.startswith("_")
                ]
                for method in methods:
                    lines.extend(
                        [
                            f".. py:method:: {item.name}.{self._signature(method)}",
                            "   :no-index:",
                            "",
                        ]
                    )
                    self._append_description(lines, method, indent="   ", what="method")
            else:
                symbol_count += 1
                lines.extend(
                    [f".. py:function:: {self._signature(item)}", "   :no-index:"]
                )
                if isinstance(item, ast.AsyncFunctionDef):
                    lines.append("   :async:")
                lines.append("")
                self._append_description(lines, item, indent="   ", what="function")

        if not symbol_count and not self._public_assignments(tree):
            lines.extend(["No public symbols are defined.", ""])

        container = nodes.container(classes=["source-api-reference"])
        self.state.nested_parse(StringList(lines), self.content_offset, container)
        return [container]


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
