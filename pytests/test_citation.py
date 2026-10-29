"""Citations (LION.utils.citation): every LION component's cite() returns its
citation as text (BibTeX or MLA), or None if it has none."""

import importlib
import inspect
import pkgutil
import re

import pytest

import LION
from LION.data_loaders.LIONDataset import LIONDataset
from LION.models.LIONmodel import LIONmodel
from LION.optimizers.LIONsolver import LIONsolver
from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.utils.citation import citation

BASES = (LIONmodel, LIONsolver, LIONReconstructor, LIONDataset)

# Components from a paper or a dataset: they must say where they come from.
FROM_A_PAPER = [
    "LION.data_loaders.LIDC_IDRI.LIDC_IDRI",
    "LION.data_loaders.deteCT.deteCT",
    "LION.models.CNNs.UNets.Unet.UNet",
    "LION.models.CNNs.dncnn.DnCNN",
    "LION.models.iterative_unrolled.LPD.LPD",
    "LION.models.post_processing.FBPConvNetImage.FBPConvNetImage",
    "LION.models.post_processing.FBPMSDNet.FBPMSDNet",
    "LION.optimizers.Noise2InverseSolver.Noise2InverseSolver",
    "LION.optimizers.Proj2ProjSolver.Proj2ProjSolver",
    "LION.optimizers.Sparse2InverseSolver.Sparse2InverseSolver",
    "LION.optimizers.Noisier2InverseSolver.Noisier2InverseSolver",
    "LION.optimizers.Equivariance2InverseSolver.Equivariance2InverseSolver",
    "LION.reconstructors.PnP.PnP",
    "LION.reconstructors.LearnedRegularizerReconstructor.LearnedRegularizerReconstructor",
]


def components():
    found = {}
    for info in pkgutil.walk_packages(LION.__path__, "LION."):
        try:
            module = importlib.import_module(info.name)
        except Exception:  # modules needing a backend or data not installed here
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, BASES) and cls.__module__ == info.name:
                found[f"{cls.__module__}.{cls.__qualname__}"] = cls
    return found


def test_every_component_returns_its_citation_or_none():
    found = components()
    assert len(found) > 20
    for name, cls in found.items():
        for cite_format in ("bib", "MLA"):
            text = cls.cite(cite_format)
            assert text is None or (isinstance(text, str) and text), name
            assert text is None or "\x1b[" not in text, name  # no terminal codes
        bib = cls.cite("bib")
        # accents keep BibTeX's backslash: {\"O}ktem, not {"O}ktem
        assert bib is None or not re.search(r'\{"[A-Za-z]\}', bib), name


def test_components_from_a_paper_cite_it():
    for name in FROM_A_PAPER:
        module, cls = name.rsplit(".", 1)
        cls = getattr(importlib.import_module(module), cls)
        assert cls.cite().startswith("@"), name  # BibTeX by default
        assert cls.cite("MLA") and not cls.cite("MLA").startswith("@"), name


def test_the_citation_helper():
    assert citation("bib", bib="\n    @article{a}\n    ", mla="A.") == "@article{a}"
    assert citation("MLA", bib="", mla="  A. B.  ") == "A. B."
    with pytest.raises(ValueError, match="not understood"):
        citation("APA", bib="", mla="")
