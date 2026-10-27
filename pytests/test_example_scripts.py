"""The example scripts in ``scripts/example_scripts`` must at least be importable.

Nothing else in the test suite looks at those files: they are scripts, not
modules, so neither LION nor another test imports them, and an import that
points at a module that has been renamed stays broken until a user runs the
script (``Sparse2Inverse.py`` did, for months). These tests read the imports
statically, so they need neither CUDA nor a dataset, and they do not run the
scripts: running them needs a GPU, and some need a dataset too.

Only module-level imports are checked. Imports inside functions are usually
deliberate (optional or expensive backends).
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((ROOT / "scripts" / "example_scripts").glob("*.py"))
README = ROOT / "scripts" / "example_scripts" / "README.md"

# Import name -> distribution name, as declared in pyproject.toml. A script may
# only import LION, the standard library, another module of this repository, or
# one of these, so that a typo or a module that moved is not silently accepted.
THIRD_PARTY = {
    "deepinv": "deepinv",
    "jaxtyping": "jaxtyping",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "skimage": "scikit-image",
    "torch": "torch",
    "torchmetrics": "torchmetrics",
    "tqdm": "tqdm",
    "ts_algorithms": "ts_algorithms",
}


def _module_level_imports(script: Path) -> list[tuple[int, str]]:
    """The (line, module) of every module imported at the top level of a script."""
    tree = ast.parse(script.read_text(), filename=str(script))
    imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module is None:
                continue  # relative import, not something a script should have
            imports.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
    return imports


def _resolves_in_repo(module: str) -> bool:
    """Whether ``module`` is a file or a package of this repository."""
    path = ROOT.joinpath(*module.split("."))
    return path.with_suffix(".py").exists() or (path / "__init__.py").exists()


def _declared_distributions() -> set[str]:
    """The distribution names pyproject.toml declares, normalised."""
    tomllib = pytest.importorskip("tomllib", reason="needs Python 3.11")
    with open(ROOT / "pyproject.toml", "rb") as f:
        project = tomllib.load(f)["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements += extra
    names = set()
    for requirement in requirements:
        # "name[extra] >= 1.0 ; python_version > '3.10'" -> "name"
        name = requirement.split(";")[0].split("@")[0].split("[")[0]
        for separator in ("==", ">=", "<=", "!=", "~=", ">", "<"):
            name = name.split(separator)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_imports_point_at_existing_modules(script):
    unknown = []
    for line, module in _module_level_imports(script):
        root = module.split(".")[0]
        if root in sys.stdlib_module_names or root == "LION":
            continue
        if root in THIRD_PARTY or _resolves_in_repo(root):
            continue
        unknown.append(f"{script.name}:{line} imports {module}")
    assert not unknown, "unknown modules imported:\n" + "\n".join(unknown)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_lion_imports_point_at_existing_modules(script):
    missing = [
        f"{script.name}:{line} imports {module}"
        for line, module in _module_level_imports(script)
        if module.split(".")[0] == "LION" and not _resolves_in_repo(module)
    ]
    assert not missing, "LION modules that do not exist:\n" + "\n".join(missing)


def test_third_party_modules_are_declared_in_pyproject():
    declared = _declared_distributions()
    missing = sorted(
        f"{module} ({distribution})"
        for module, distribution in THIRD_PARTY.items()
        if distribution.replace("_", "-") not in declared
    )
    assert not missing, "not declared in pyproject.toml: " + ", ".join(missing)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_is_documented(script):
    assert script.name in README.read_text(), f"{script.name} is not in README.md"
