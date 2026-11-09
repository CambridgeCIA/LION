# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  :
# =============================================================================
"""Every Python file of LION, its tests and its scripts starts with the
licence header (GPL-3, as ASTRA and tomosipo require)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(
    path.relative_to(ROOT)
    for folder in ("LION", "pytests", "scripts")
    for path in (ROOT / folder).rglob("*.py")
)


@pytest.mark.parametrize("path", FILES, ids=str)
def test_licence_header(path):
    head = (ROOT / path).read_text(encoding="utf-8").splitlines()[:6]
    assert "# This file is part of LION library" in head
    assert "# License : GPL-3" in head
