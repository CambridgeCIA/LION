"""Generic plug-and-play reconstructor class."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Literal, TypeVar

from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.utils.parameter import LIONParameter

DenoiserT = TypeVar("DenoiserT", bound=Callable)
SupportedDenoiser = Literal[
    "drunet",
    "drunet_admm",
    "gs_drunet",
    "gs_drunet_deepinv",
    "dwtdnet",
    "dwtdnet_admm",
    "gs_dwtdnet",
]


class BasePnP(LIONReconstructor, Generic[DenoiserT]):
    """Generic plug-and-play reconstructor class."""

    def __init__(self, physics, denoiser: DenoiserT):
        """Initialize the plug-and-play reconstructor.

        Args:
            physics: A CT Geometry or an Operator. If Geometry, CT operator will be inferred.
            denoiser: The denoiser prior used in the PnP algorithm.
        """
        super().__init__(physics)
        self.denoiser = denoiser


@dataclass(kw_only=True)
class DenoiserParameter(LIONParameter):
    """Configuration for a denoiser.

    Attributes:
        name: Name of the denoiser.
        size: Size of the denoiser model (currently only for DWTDNet).
        weights_path: Path to the pre-trained weights for the denoiser.
    """

    name: SupportedDenoiser = "gs_drunet"
    size: str | None = None
    weights_path: Path | str | None = None
    b: float | None = None
    eps: float = 1e-12
