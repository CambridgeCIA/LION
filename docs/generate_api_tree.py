"""Generate the nested Sphinx API tree from the current LION package layout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs" / "source" / "api" / "tree"
SOURCE_DIRECTORIES = ["LION", "scripts", "utils", "tests", "demos"]

# These audited modules also import safely on a CPU-only documentation builder,
# so Sphinx can use autodoc for them. Audit status and import safety are kept
# separate: executable scripts and tests are source-rendered even when their
# documentation has been audited.
AUTODOC_MODULES = {
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
    "LION.reconstructors.diffusion.data_consistency",
    "LION.reconstructors.diffusion.dps_langevin",
    "LION.reconstructors.diffusion.langevin",
    "LION.reconstructors.diffusion.PaDIS",
    "LION.reconstructors.diffusion.padis.citations",
    "LION.reconstructors.diffusion.padis.generation",
    "LION.reconstructors.diffusion.padis.parameters",
    "LION.reconstructors.diffusion.padis.physics",
    "LION.reconstructors.diffusion.padis.prior",
    "LION.reconstructors.diffusion.padis.sampling",
    "LION.reconstructors.diffusion.predictor_corrector",
    "LION.reconstructors.PnP",
    "LION.utils.paths",
}

AUDITED_SOURCE_ROOTS = {
    Path("scripts/paper_scripts/PaDIS-Reproduction"),
    Path("tests"),
}

README_MAP = {
    Path("."): ROOT / "README.md",
    Path("LION/data_loaders"): ROOT / "LION" / "data_loaders" / "README.md",
    Path("LION/data_loaders/LIDC_IDRI"): ROOT
    / "LION"
    / "data_loaders"
    / "LIDC_IDRI"
    / "README.md",
    Path("LION/data_loaders/LUNA16"): ROOT
    / "LION"
    / "data_loaders"
    / "LUNA16"
    / "README.md",
    Path("LION/models"): ROOT / "LION" / "models" / "README.md",
    Path("demos"): ROOT / "demos" / "README.md",
    Path("scripts"): ROOT / "scripts" / "README.md",
    Path("scripts/example_scripts"): ROOT / "scripts" / "example_scripts" / "README.md",
    Path("scripts/paper_scripts"): ROOT / "scripts" / "paper_scripts" / "README.md",
    Path("scripts/paper_scripts/Continuous_Learned_Primal_Dual"): ROOT
    / "scripts"
    / "paper_scripts"
    / "Continuous_Learned_Primal_Dual"
    / "README.md",
    Path("scripts/paper_scripts/PaDIS-Reproduction"): ROOT
    / "scripts"
    / "paper_scripts"
    / "PaDIS-Reproduction"
    / "README.md",
    Path("scripts/hackathon_scripts/synerby26"): ROOT
    / "scripts"
    / "hackathon_scripts"
    / "synerby26"
    / "README.md",
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


def documentation_audited(source: Path, module: str) -> bool:
    """Return whether a source file has completed the documentation audit."""

    if module in AUTODOC_MODULES:
        return True
    relative = source.relative_to(ROOT)
    return any(relative.is_relative_to(root) for root in AUDITED_SOURCE_ROOTS)


def write_module_page(source: Path, destination: Path) -> None:
    """Write one leaf page with immediately visible API documentation."""

    module = module_name(source)
    audited = documentation_audited(source, module)
    use_autodoc = module in AUTODOC_MODULES
    title = source.name if audited else f"{source.name} 🚧"
    lines = heading(title)
    lines.extend(
        [
            f"**Source:** ``{source.relative_to(ROOT).as_posix()}``",
            "",
        ]
    )
    if use_autodoc:
        lines.extend(
            [
                f".. automodule:: {module}",
                "   :members:",
                "   :show-inheritance:",
            ]
        )
    elif audited:
        lines.append(f".. sourceautosummary:: {module}")
    else:
        lines.extend(
            [
                ".. warning::",
                "",
                "   This file has not yet received a complete narrative and docstring audit.",
                "   Its public source-level API is listed automatically below.",
                "",
                f".. sourceautosummary:: {module}",
            ]
        )
    lines.append("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def child_directories(directory: Path) -> list[Path]:
    """Return child directories that contain Python sources."""

    if directory == ROOT:
        return [ROOT / name for name in SOURCE_DIRECTORIES if (ROOT / name).is_dir()]
    return sorted(
        child
        for child in directory.iterdir()
        if child.is_dir() and any(child.rglob("*.py"))
    )


def write_directory_page(directory: Path, destination: Path) -> None:
    """Write a package/directory index and its nested toctree."""

    relative = directory.relative_to(ROOT)
    title = "Repository Python API" if not relative.parts else relative.as_posix() + "/"
    lines = heading(title)

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
    modules = sorted(directory.glob("*.py"))
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

    directories = [ROOT]
    for name in SOURCE_DIRECTORIES:
        source_root = ROOT / name
        directories.extend([source_root, *child_directories_recursive(source_root)])
    for directory in directories:
        relative = directory.relative_to(ROOT)
        output_directory = OUTPUT_ROOT / relative
        write_directory_page(directory, output_directory / "index.rst")
        for source in sorted(directory.glob("*.py")):
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
