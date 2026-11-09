# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Every Python file of LION, its tests and its scripts starts with the
licence header (GPL-3, as ASTRA and tomosipo require), naming its authors."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _has_header(path):
    head = path.read_text(encoding="utf-8").splitlines()[:6]
    authors = [line for line in head if line.startswith("# Author")]
    return (
        "# This file is part of LION library" in head
        and "# License : GPL-3" in head
        and bool(authors)
        and bool(authors[0].split(":", 1)[1].strip())
    )


def test_licence_headers():
    missing = [
        str(path.relative_to(ROOT))
        for folder in ("LION", "pytests", "scripts")
        for path in sorted((ROOT / folder).rglob("*.py"))
        if not _has_header(path)
    ]
    assert missing == [], "no licence header, or no authors named"
