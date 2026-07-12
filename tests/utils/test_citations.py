"""Tests for component-local academic citation APIs."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from LION.classical_algorithms.fdk import fdk
from LION.classical_algorithms.tv_min import tv_min
from LION.operators import CTProjectionOp
from LION.reconstructors import PaDIS, PnP


@pytest.mark.parametrize(
    ("citation_api", "keys"),
    (
        (
            PaDIS.cite,
            (
                "hu_learning_2024",
                "chung_diffusion_2022",
                "song_generative_2019",
                "song_score-based_2020-1",
                "wang_zero-shot_2022-1",
                "karras_elucidating_2022-1",
                "feldkamp_practical_1984",
            ),
        ),
        (PnP.cite, ("chan_plug-and-play_2017",)),
        (getattr(fdk, "cite"), ("feldkamp_practical_1984",)),
        (getattr(tv_min, "cite"), ("chambolle_first-order_2011",)),
        (
            CTProjectionOp.cite,
            ("hendriksen_tomosipo_2021", "van_aarle_fast_2016"),
        ),
    ),
)
def test_component_bibtex_contains_required_entries(
    citation_api: Callable[[str], None],
    keys: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each implementation should expose its canonical BibTeX entries."""
    citation_api("bib")
    output = capsys.readouterr().out
    for key in keys:
        assert f"{{{key}," in output


@pytest.mark.parametrize(
    "citation_api",
    (
        PaDIS.cite,
        PnP.cite,
        getattr(fdk, "cite"),
        getattr(tv_min, "cite"),
        CTProjectionOp.cite,
    ),
)
def test_component_citations_reject_unknown_formats(
    citation_api: Callable[[str], None],
) -> None:
    """Citation APIs should retain LION's MLA/BibTeX format contract."""
    with pytest.raises(ValueError, match='only "MLA" and "bib" are supported'):
        citation_api("apa")
