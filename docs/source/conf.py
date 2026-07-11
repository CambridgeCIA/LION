"""Sphinx configuration for the LION documentation."""

from __future__ import annotations

import os
from pathlib import Path
import sys


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

html_theme = "furo"
html_title = "LION Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/CambridgeCIA/LION/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
