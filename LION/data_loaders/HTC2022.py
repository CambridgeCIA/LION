# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""The Helsinki Tomography Challenge 2022 data (HTC2022): measured fan-beam
sinograms of acrylic phantoms, for limited-angle CT."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import torch

from LION.data_loaders.LIONDataset import LIONDataset
from LION.operators.ct.ct_geometry import Geometry
from LION.utils import paths
from LION.utils.citation import citation
from LION.utils.device import default_device
from LION.utils.parameter import LIONParameter

# The scanner (the files' metadata): fan beam, source to origin and to
# detector in mm, a 560-pixel detector of 0.2 mm (binned 4x), 0.5 degree
# steps over 360 degrees; images of 512 x 512 pixels of the effective pixel
# size (the detector's at the origin), as the challenge's
DSO = 410.66
DSD = 553.74
DETECTOR_PIXELS = 560
DETECTOR_PIXEL_SIZE = 0.2
EFFECTIVE_PIXEL_SIZE = 0.1483223173330444
IMAGE_PIXELS = 512
ANGLE_STEP = 0.5  # degrees
FULL_ANGLES = 720  # 0 to 359.5 degrees (the files' 721st, 360, repeats the first)
# The direction of the angles in LION's operator (tomosipo) for the files' own
# (as CIL's code for the challenge does)
ANGLE_SIGN = -1


def geometry(angles: int = FULL_ANGLES, start: int = 0) -> Geometry:
    """HTC2022's fan-beam geometry with ``angles`` consecutive angles of the
    full scan, from the ``start``-th (0.5 degree steps): all 720 by default;
    e.g. ``geometry(181)``, a 90 degree arc."""
    if not 1 <= angles <= FULL_ANGLES:
        raise ValueError(f"angles must be from 1 to {FULL_ANGLES}, got {angles}")
    degrees = (start + np.arange(angles)) * ANGLE_STEP
    return Geometry(
        mode="fan",
        dso=DSO,
        dsd=DSD,
        detector_shape=[1, DETECTOR_PIXELS],
        detector_size=[DETECTOR_PIXEL_SIZE, DETECTOR_PIXEL_SIZE * DETECTOR_PIXELS],
        image_shape=[1, IMAGE_PIXELS, IMAGE_PIXELS],
        image_size=[
            EFFECTIVE_PIXEL_SIZE,
            EFFECTIVE_PIXEL_SIZE * IMAGE_PIXELS,
            EFFECTIVE_PIXEL_SIZE * IMAGE_PIXELS,
        ],
        angles=ANGLE_SIGN * np.deg2rad(degrees),
    )


def segment(image: np.ndarray) -> np.ndarray:
    """The challenge's segmentation of a reconstruction: negative values set to
    0, Otsu's threshold, then a closing with a disc of radius 3 pixels (1:
    acrylic, 0: holes and background)."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, disk

    image = np.clip(image, 0, None)
    return closing(image > threshold_otsu(image), disk(3)).astype(np.uint8)


class HTC2022(LIONDataset):
    """The Helsinki Tomography Challenge 2022 data (Meaney et al., 2022):
    measured fan-beam sinograms of 26 acrylic phantoms, 7 cm discs with holes.
    The challenge's 5 training phantoms are the "train" split; its 21 test
    phantoms (7 levels of difficulty, 3 each) the "test" split; there is no
    validation split.

    A sample is (sinogram, target): the sinogram (1, angles, 560) with the
    angles of the dataset's geometry (all 720 of the full scan, 0.5 degree
    steps, by default), and the target (1, 512, 512).

    The test phantoms' challenge data are arcs (90 to 30 degrees) starting at
    different angles. ``pre_process_HTC2022.py`` turns each phantom so that
    its arc starts at the first angle (a cyclic shift of its full sinogram, an
    exact rotation), so one geometry fits them all: ``geometry(181)`` gives
    the 90 degree arc of level 1, ``geometry(61)`` the 30 degree one of level
    7. The targets are made from the turned full sinograms, as the challenge
    made its own: the FBP, and its segmentation (``segment``).
    ``validate_HTC2022.py`` compares them with the challenge's.

    ``level`` and ``arc`` of each phantom are in ``sample_dataframe``.
    Samples are returned on ``default_device()``."""

    def __init__(
        self,
        mode,
        operator_params: Geometry = None,
        parameters: LIONParameter = None,
    ):
        """The ``mode`` split ("train" or "test") with ``parameters``
        (``default_parameters()`` if None). ``operator_params``: the geometry,
        replacing ``parameters.operator_params`` if given.

        ``parameters`` (``default_parameters()``):

        - ``path_to_dataset``: the pre-processed data
          (``HTC2022/pre_process_HTC2022.py``);
        - ``target``: "reconstruction" (the FBP of the full scan) or
          "segmentation" (the challenge's ground truth: 1 acrylic, 0 holes and
          background);
        - ``operator_params``: the geometry: ``geometry()``, or one with
          consecutive angles of it from the first, e.g. ``geometry(181)``
          (the measured sinograms are cut to match; others are refused)."""
        if mode not in ("train", "test"):
            raise ValueError(
                f'mode must be "train" or "test" (HTC2022 has no validation), got {mode!r}'
            )
        if parameters is None:
            parameters = self.default_parameters()
        if parameters.target not in ("reconstruction", "segmentation"):
            raise ValueError(
                'target must be "reconstruction" or "segmentation", '
                f"got {parameters.target!r}"
            )
        if operator_params is not None:
            parameters.operator_params = operator_params
        elif not isinstance(parameters.operator_params, Geometry):
            # parameters read from JSON have it as a plain LIONParameter
            parameters.operator_params = Geometry.init_from_parameter(
                parameters.operator_params
            )
        self.parameters = parameters
        self.mode = mode
        self.operator_params = parameters.operator_params
        self.geometry = self.operator_params
        self.angles = self._angles_of(self.geometry)
        self.path_to_dataset = pathlib.Path(parameters.path_to_dataset)
        self.device = default_device()
        table = pd.read_csv(self.path_to_dataset.joinpath("samples.csv"))
        self.sample_dataframe = table[table["split"] == mode].reset_index(drop=True)

    @staticmethod
    def default_parameters():
        """The parameters (see ``__init__``): full scans to their FBP."""
        params = LIONParameter()
        params.path_to_dataset = paths.HTC2022_PROCESSED_DATASET_PATH
        params.target = "reconstruction"
        params.operator_params = geometry()
        return params

    @staticmethod
    def get_default_geometry():
        """The full scan's geometry, ``geometry()``."""
        return geometry()

    @staticmethod
    def _angles_of(geo: Geometry) -> int:
        """How many angles of the full scan ``geo`` has, from the first; an
        error if it isn't HTC2022's geometry with such angles."""
        full = geometry()
        n = len(geo.angles)
        same = (
            geo.mode == full.mode
            and np.isclose(geo.dso, full.dso)
            and np.isclose(geo.dsd, full.dsd)
            and np.array_equal(geo.detector_shape, full.detector_shape)
            and np.allclose(geo.detector_size, full.detector_size)
            and np.array_equal(geo.image_shape, full.image_shape)
            and np.allclose(geo.image_size, full.image_size)
            and n <= FULL_ANGLES
            and np.allclose(geo.angles, full.angles[:n])
        )
        if not same:
            raise ValueError(
                "The data are measured: the geometry must be HTC2022's, "
                "HTC2022.geometry(angles), with consecutive angles from the first"
            )
        return n

    def get_operator(self):
        """The CT operator of the dataset's geometry."""
        return self.geometry.make_operator()

    def sample_ids(self):
        """What each sample is, in order: the phantom's name (e.g. "ta",
        "7c")."""
        return list(self.sample_dataframe["name"])

    def sample_files(self):
        """The files the samples are read from, in order."""
        return [
            self.path_to_dataset.joinpath(name, file)
            for name in self.sample_ids()
            for file in ("sinogram.npy", f"{self.parameters.target}.npy")
        ]

    def __len__(self):
        return len(self.sample_dataframe)

    def __getitem__(self, index):
        folder = self.path_to_dataset.joinpath(self.sample_ids()[int(index)])
        sinogram = np.load(folder.joinpath("sinogram.npy"))[: self.angles]
        target = np.load(folder.joinpath(f"{self.parameters.target}.npy"))
        sinogram = torch.from_numpy(np.ascontiguousarray(sinogram, np.float32))
        target = torch.from_numpy(np.ascontiguousarray(target, np.float32))
        return sinogram[None].to(self.device), target[None].to(self.device)

    @staticmethod
    def cite(cite_format="bib"):
        """The HTC2022 data, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @misc{meaney2023htc2022,
            title={Helsinki Tomography Challenge 2022 ({HTC2022}) open tomographic dataset},
            author={Meaney, Alexander and Silva de Moura, Fernando and Juvonen, Markus and Siltanen, Samuli},
            year={2023},
            publisher={Zenodo},
            version={1.4.0},
            doi={10.5281/zenodo.8041800}
            }
            """,
            mla="""
            Meaney, Alexander, Fernando Silva de Moura, Markus Juvonen, and
            Samuli Siltanen. "Helsinki Tomography Challenge 2022 (HTC2022) open
            tomographic dataset." Zenodo, version 1.4.0 (2023).
            doi:10.5281/zenodo.8041800.
            """,
        )
