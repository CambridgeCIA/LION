"""Generate the nested Sphinx API tree from the current LION package layout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "LION"
OUTPUT_ROOT = ROOT / "docs" / "source" / "api" / "tree"

# These modules have received the current documentation audit and import safely
# on a CPU-only documentation builder. All other modules remain discoverable
# through source inspection and carry an explicit warning.
AUDITED_MODULES = {
    "LION.CTtools.ct_geometry",
    "LION.CTtools.ct_utils",
    "LION.classical_algorithms.fdk",
    "LION.classical_algorithms.spgl1_torch",
    "LION.classical_algorithms.tv_min",
    "LION.data_loaders.LIDC_IDRI",
    "LION.experiments.ct_experiments",
    "LION.losses.PaDIS",
    "LION.models.LIONmodel",
    "LION.models.diffusion.NCSNpp",
    "LION.optimizers.LIONsolver",
    "LION.optimizers.PaDISSolver",
    "LION.reconstructors.LIONreconstructor",
    "LION.reconstructors.PaDIS",
    "LION.reconstructors.PnP",
    "LION.utils.paths",
}

README_MAP = {
    Path("data_loaders"): PACKAGE_ROOT / "data_loaders" / "README.md",
    Path("data_loaders/LIDC_IDRI"): PACKAGE_ROOT
    / "data_loaders"
    / "LIDC_IDRI"
    / "README.md",
    Path("data_loaders/LUNA16"): PACKAGE_ROOT / "data_loaders" / "LUNA16" / "README.md",
    Path("models"): PACKAGE_ROOT / "models" / "README.md",
}


def heading(title: str, marker: str = "=") -> list[str]:
    """Return a reStructuredText heading."""

    # Docutils treats the warning emoji as a double-width display character.
    display_width = len(title) + title.count("🚧")
    return [title, marker * display_width, ""]


def module_name(path: Path) -> str:
    """Convert a source path into its dotted display/import name."""

    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def relative_include(page: Path, readme: Path) -> str:
    """Return the README path relative to a generated page."""

    return Path(os.path.relpath(readme, page.parent)).as_posix()


def write_module_page(source: Path, destination: Path) -> None:
    """Write one leaf module page."""

    module = module_name(source)
    audited = module in AUDITED_MODULES
    title = module if audited else f"{module} 🚧"
    lines = heading(title)
    lines.append(f".. apimodule:: {module}")
    if not audited:
        lines.extend(["   :source:", "   :warning:"])
    lines.append("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def child_directories(directory: Path) -> list[Path]:
    """Return child directories that contain Python sources."""

    return sorted(
        child
        for child in directory.iterdir()
        if child.is_dir() and any(child.rglob("*.py"))
    )


def write_package_page(directory: Path, destination: Path) -> None:
    """Write a package/directory index and its nested toctree."""

    relative = directory.relative_to(PACKAGE_ROOT)
    dotted = "LION" + (f".{'.'.join(relative.parts)}" if relative.parts else "")
    lines = heading(dotted)

    readme = README_MAP.get(relative)
    if readme is not None:
        lines.extend(
            [
                "Directory README",
                "----------------",
                "",
                f".. include:: {relative_include(destination, readme)}",
                "   :parser: myst_parser.sphinx_",
                "",
            ]
        )

    children = child_directories(directory)
    modules = sorted(
        path for path in directory.glob("*.py") if path.name != "__init__.py"
    )
    if children or modules:
        lines.extend(
            [
                "Directory entries",
                "-----------------",
                "",
            ]
        )
        lines.extend(
            f"- :doc:`{child.name}/ <{child.name}/index>`" for child in children
        )
        lines.extend(f"- :doc:`{module.name} <{module.stem}>`" for module in modules)
        lines.extend(
            [
                "",
                ".. toctree::",
                "   :maxdepth: 20",
                "   :hidden:",
                "",
            ]
        )
        lines.extend(f"   {child.name}/ <{child.name}/index>" for child in children)
        lines.extend(f"   {module.name} <{module.stem}>" for module in modules)
        lines.append("")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def generate() -> None:
    """Regenerate the complete nested API documentation tree."""

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    directories = [PACKAGE_ROOT, *child_directories_recursive(PACKAGE_ROOT)]
    for directory in directories:
        relative = directory.relative_to(PACKAGE_ROOT)
        output_directory = OUTPUT_ROOT / relative
        write_package_page(directory, output_directory / "index.rst")
        for source in sorted(directory.glob("*.py")):
            if source.name != "__init__.py":
                write_module_page(source, output_directory / f"{source.stem}.rst")


def child_directories_recursive(directory: Path) -> list[Path]:
    """Return all descendant directories containing Python files."""

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_dir() and any(path.rglob("*.py"))
    )


def main() -> None:
    """Parse command-line arguments and generate the API tree."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    generate()


if __name__ == "__main__":
    main()
