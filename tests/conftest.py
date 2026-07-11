"""Make the reorganised PaDIS reproduction scripts importable in tests."""

from pathlib import Path
import sys


PADIS_SCRIPT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "paper_scripts"
    / "PaDIS-Reproduction"
)

for directory in (
    "core",
    "reconstruction",
    "reporting",
    "training",
    "tuning",
):
    sys.path.insert(0, str(PADIS_SCRIPT_ROOT / directory))
