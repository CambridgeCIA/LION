# Configuration file for the Sphinx documentation builder.
import ast
import dataclasses
import inspect
import os
import re
import shutil
import sys
from pathlib import Path
from typing import get_overloads

import nbformat
import sphinx.util.inspect
from sphinx.ext.autodoc import (
    AttributeDocumenter,
    ClassDocumenter,
    MethodDocumenter,
    PropertyDocumenter,
)
from sphinx.util.inspect import (
    isclassmethod,
    isstaticmethod,
    signature,
    stringify_signature,
)
from sphinx_pyproject import SphinxConfig

sys.path.insert(
    0, os.path.abspath("../../src")
)  # Source code dir relative to this file

config = SphinxConfig(
    "../../pyproject.toml",
    globalns=globals(),
)


project = name
copyright = ""  # TODO: Add copyright notice.

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_github_style",
    "sphinx.ext.napoleon",
    "myst_nb",
    "sphinx.ext.mathjax",
    "sphinx-mathjax-offline",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinx_last_updated_by_git",
]


intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "typing-extensions": ("https://typing-extensions.readthedocs.io/en/latest/", None),
}

autosectionlabel_prefix_document = True
napoleon_use_param = True
napoleon_use_rtype = False
typehints_defaults = "comma"
typehints_use_signature = True
typehints_use_signature_return = True
typehints_use_rtype = False
typehints_document_rtype = False
autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False
autodoc_member_order = "groupwise"
autodoc_preserve_defaults = True
autodoc_class_signature = "separated"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
nb_execution_mode = "auto"
nb_output_stderr = "remove"
nb_output_stdout = "remove"
nb_execution_timeout = 120
nb_merge_streams = True
html_favicon = "_static/favicon.svg"
html_theme = "sphinx_rtd_theme"
html_title = name
html_show_sphinx = False
html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo_white.svg"
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}
html_theme_options = {
    "logo_only": True,
    "collapse_navigation": False,
    "navigation_depth": -1,
}
html_context = {
    "display_github": False,
    "github_user": "CambridgeCIA",
    "github_repo": "LION",
    "github_version": "main",
}
linkcode_blob = html_context["github_version"]
linkcode_link_text = "[source]"
default_role = "py:obj"
pygments_style = "default"


def get_lambda_source(obj):
    """Convert lambda to source code."""
    source = inspect.getsource(obj)
    for node in ast.walk(ast.parse(source.strip())):
        if isinstance(node, ast.Lambda):
            return ast.unparse(node.body)


class DefaultValue:
    """Used to store default values of dataclass fields with default factory."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        """This is called by sphinx when rendering the default value."""
        return self.value


def rewrite_dataclass_init_default_factories(app, obj, bound_method) -> None:
    """Replace default fields in dataclass.__init__."""
    if (
        "init" not in str(obj)
        or not getattr(obj, "__defaults__", None)
        or not any(
            isinstance(d, dataclasses._HAS_DEFAULT_FACTORY_CLASS)
            for d in obj.__defaults__
        )
    ):
        # not an dataclass.__init__ method with default factory
        return
    parameters = inspect.signature(obj).parameters
    module = sys.modules[obj.__module__]
    class_ref = getattr(module, obj.__qualname__.split(".")[0])
    defaults = {}
    for field in dataclasses.fields(class_ref):
        if field.default_factory is not dataclasses.MISSING:
            if field.name not in parameters:
                continue
            if (
                hasattr(field.default_factory, "__name__")
                and field.default_factory.__name__ == "<lambda>"
            ):
                defaults[field.name] = DefaultValue(
                    get_lambda_source(field.default_factory)
                )
            elif hasattr(field.default_factory, "__name__"):
                defaults[field.name] = DefaultValue(
                    field.default_factory.__name__ + "()"
                )
            else:
                continue
    new_defaults = tuple(
        defaults.get(name, param.default)
        for name, param in parameters.items()
        if param.default != inspect._empty
    )
    obj.__defaults__ = new_defaults


def autodoc_inherit_overload(app, what, name, obj, options, sig, ret_ann):
    """Create overloaded signatures."""
    if what in ("function", "method") and callable(obj):
        try:
            overloads = get_overloads(obj)
        except:
            return (sig, ret_ann)
        if overloads:
            kwargs = {}
            if app.config.autodoc_typehints in ("none", "description"):
                kwargs["show_annotation"] = False
            if app.config.autodoc_typehints_format == "short":
                kwargs["unqualified_typehints"] = True
            type_aliases = app.config.autodoc_type_aliases
            bound_method = what == "method"
            sigs = []
            for overload in overloads:
                if hasattr(overload, "__func__"):
                    overload = overload.__func__  # classmethod or staticmethod
                overload_sig = signature(
                    overload, bound_method=bound_method, type_aliases=type_aliases
                )
                sigs.append(stringify_signature(overload_sig, **kwargs))
            return "\n".join(sigs), None


class CustomClassDocumenter(ClassDocumenter):
    """Custom Documenter to reorder class members."""

    def sort_members(
        self, documenters: list[tuple["Documenter", bool]], order: str
    ) -> list[tuple["Documenter", bool]]:
        """Sort the given member list with custom logic for `groupwise` ordering."""
        if order == "groupwise":
            if not self.parse_name() or not self.import_object():
                return documenters
            # Split members into groups (non-inherited,inherited)
            static_methods = [], []
            class_methods = [], []
            special_methods = [], []
            instance_methods = [], []
            attributes = [], []
            properties = [], []
            other = [], []
            others_methods = []
            init_method = []
            call_methods = []

            for documenter in documenters:
                doc = documenter[0]
                parsed = doc.parse_name() and doc.import_object()
                inherited = parsed and doc.object_name not in self.object.__dict__
                if isinstance(doc, AttributeDocumenter):
                    attributes[inherited].append(documenter)
                elif isinstance(doc, PropertyDocumenter):
                    properties[inherited].append(documenter)
                elif isinstance(doc, MethodDocumenter):
                    if not parsed:
                        others_methods.append(documenter)
                        continue
                    if doc.object_name == "__init__":
                        init_method.append(documenter)
                    elif doc.object_name in ("__call__", "forward", "adjoint"):
                        call_methods.append(documenter)
                    elif doc.object_name[:2] == "__":
                        special_methods[inherited].append(documenter)
                    elif isclassmethod(doc.object):
                        class_methods[inherited].append(documenter)
                    elif isstaticmethod(doc.object):
                        static_methods[inherited].append(documenter)
                    else:
                        instance_methods[inherited].append(documenter)
                else:
                    other[inherited].append(documenter)
                    continue
            # Combine groups in the desired order
            constructors = init_method + class_methods[0] + class_methods[1]
            call_methods = sorted(call_methods, key=lambda x: x[0].object_name)
            methods = (
                call_methods
                + instance_methods[0]
                + instance_methods[1]
                + others_methods
                + static_methods[0]
                + static_methods[1]
                + special_methods[0]
                + special_methods[1]
            )
            return (
                constructors
                + attributes[0]
                + attributes[1]
                + properties[0]
                + properties[1]
                + methods
                + other[0]
                + other[1]
            )
        else:
            return super().sort_members(documenters, order)


def replace_patterns_in_markdown(app, docname, source):
    """Replace patterns like `module.class` with {py:obj}`module.class` in Markdown cells."""
    if "_notebooks" not in docname:
        return
    notebook = nbformat.reads(source[0], as_version=4)
    for cell in notebook.cells:
        if cell["cell_type"] == "markdown":
            # Replace with `text` with {py:obj}`text`. leave ``text`` as is.
            cell["source"] = re.sub(
                r"(?<!`)`([^`]+)`(?!`)", r"{py:obj}`\1`", cell["source"]
            )

    source[0] = nbformat.writes(notebook)


def sync_notebooks(source_folder, dest_folder):
    """Sync notebooks from source to destination folder.

    Copy only new or updated files.
    Set execution mode to 'force' for all copied files and 'off' for all existing files.
    """
    dest = Path(dest_folder)
    dest.mkdir(parents=True, exist_ok=True)
    for src_file in Path(source_folder).iterdir():
        if src_file.is_file():
            dest_file = dest / src_file.name
            if (
                not dest_file.exists()
                or src_file.stat().st_mtime > dest_file.stat().st_mtime
            ):
                shutil.copy2(src_file, dest_file)
                print(f"Copied {src_file} to {dest_file}. ", end="")
                if os.environ.get("NORUN") == "1":
                    print("NORUN: Skipping execution.")
                    mode = "off"
                else:
                    print('Setting execution mode to "force".')
                    mode = "force"
            else:
                print(f"Existing {dest_file}. Skipping execution.")
                mode = "off"
            content = nbformat.read(dest_file, as_version=nbformat.NO_CONVERT)
            content.metadata["mystnb"] = {"execution_mode": mode}
            nbformat.write(content, dest_file)


object_description_original = sphinx.util.inspect.object_description


def object_description_function_repr_overwrite(
    obj, *, _seen: frozenset[int] = frozenset()
) -> str:
    """Overwrite sphinx default function representation to use functionname instead of <functionname>.

    <> would break interspinx and formatting of the function name."""
    if callable(obj):
        return obj.__name__ + "()"
    return object_description_original(obj, _seen=_seen)


def setup(app):
    app.set_html_assets_policy("always")  # forces mathjax on all pages
    app.connect(
        "autodoc-before-process-signature", rewrite_dataclass_init_default_factories
    )
    app.connect("autodoc-process-signature", autodoc_inherit_overload, 0)
    app.connect("source-read", replace_patterns_in_markdown)
    app.add_autodocumenter(CustomClassDocumenter, True)
    sphinx.util.inspect.object_description = object_description_function_repr_overwrite

    sync_notebooks(
        app.srcdir.parent.parent / "examples" / "notebooks", app.srcdir / "_notebooks"
    )
