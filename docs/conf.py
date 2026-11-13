# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Sphinx configuration of LION's documentation site.

Build it with ``sphinx-build -b html docs docs/_build/html``. The demos
(``demos/``) become the tutorials; they are only run (adding their figures
and printed output) with ``LION_DOCS_RUN_DEMOS=1``, which needs a GPU and the
data (``utils/publish_docs.sh`` on holly). Otherwise the pages show their text
and code.
"""

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "LION"
author = "LION contributors"
copyright = "Cambridge Image Analysis, University of Cambridge"
release = (ROOT / "LION" / "VERSION").read_text().strip()
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "myst_parser",  # Markdown pages: records.md includes metadata.md
]

# API reference: one page per module, from the docstrings
autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
# the class docstring and __init__'s (which lists the arguments), no type
# sections: the docstrings say what each argument is
autoclass_content = "both"
autodoc_typehints = "none"
# backends LION imports only where they are used
autodoc_mock_imports = ["tomosipo", "ts_algorithms", "astra", "spyrit", "mrpro"]
# our docstrings mark arguments as ``name``: no default role is needed
default_role = None
# literal blocks in docstrings (tables, file layouts) as plain text
highlight_language = "none"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Tutorials: the demos, as pages with downloadable .py and notebooks
run_demos = os.environ.get("LION_DOCS_RUN_DEMOS") == "1"
if run_demos:
    # No progress bars in the pages: they would bury the demos' printed
    # output. (The demos still show them when run on their own.)
    import tqdm.std

    _tqdm_init = tqdm.std.tqdm.__init__

    def _quiet_tqdm(self, *args, **kwargs):
        kwargs["disable"] = True
        _tqdm_init(self, *args, **kwargs)

    tqdm.std.tqdm.__init__ = _quiet_tqdm
sphinx_gallery_conf = {
    "examples_dirs": str(ROOT / "demos"),
    "gallery_dirs": "tutorials",
    "filename_pattern": r"/d\d+_",
    "plot_gallery": run_demos,
    "download_all_examples": True,
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
    "show_signature": False,
    # the thumbnail of a demo without figures
    "default_thumb_file": str(ROOT / "docs" / "_static" / "lion-thumbnail.png"),
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]
# metadata.md's links to its own sections (#the-model-file) work on the site
myst_heading_anchors = 3

html_theme = "furo"
html_title = f"LION {release}"
html_static_path = ["_static"]
# The browser-tab icon: the lion head from the logo
html_favicon = "_static/lion-favicon.svg"
html_theme_options = {
    "light_logo": "lion-logo.svg",
    "dark_logo": "lion-logo-dark.svg",
    "source_repository": "https://github.com/CambridgeCIA/LION",
    "source_branch": "main",
    "source_directory": "docs/",
}
