"""Parameters for the PCM synthetic PNG data loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np

from LION.utils.parameter import LIONParameter
from LION.utils.paths import PCM_SYNTHETIC_PNG_PATH

PCMDataCategory = Literal["CIGS", "Organic", "Perovskite", "Reference", "Silicone"]
PCMDataType = Literal[
    "image",
    "image_txt_map",
    "hadamard_measurement_vector",
    "original_measurement_data",
    "original_measurement_data_txt",
]
DataType = TypeVar("DataType")


@dataclass(kw_only=True)
class SamplesParameter(LIONParameter):
    """Indicate which samples to load.

    Intended usage: Priortise an explicit list of ids. If such a list is not provided, use an explicit id range.
    If a range is not provided either, use a specified permutation.

    Attributes:
        ids: List of 1-indexed sample IDs to load. Highest priority.
        id_range: Two integers representing start and end IDs (1-indexed, inclusive). Used if `ids` is None.
        total_samples: The total number of samples to consider. Assumed indexed from 1 to `total_samples`.
            Useful for generating a permutation.
        perm_range: Two ints for start and end (1-indexed, inclusive) of permutation.
            Used if both `ids` and `id_range` are None.
        perm: A permutation or path to a file containing a permutation in form of a list of ints, or
            a random seed to generate a permutation. If None, generate a random permutation with non-deterministic seed.
    """

    ids: list[int] | None = None
    id_range: list[int] | None = None
    total_samples: int | None = None
    perm_range: list[int] | None = None
    perm: Path | list[int] | int | None = None

    def __post_init__(self):
        """Make explicit list of sample IDs.

        Priortise an explicit list of ids. If such a list is not provided, use the explicit id range.
        If a range is not provided either, use the specified permutation.
        """
        if self.ids is not None:
            return
        elif self.id_range is not None:
            self.ids = list(range(self.id_range[0], self.id_range[1] + 1))
        elif self.perm_range is not None or self.perm is not None:
            if isinstance(self.perm, str):
                self.perm = Path(self.perm)
            if isinstance(self.perm, Path):
                self.perm = np.loadtxt(self.perm).tolist()
            else:
                assert self.total_samples is not None
                rng = np.random.default_rng(self.perm)
                self.perm = rng.permutation(range(1, self.total_samples + 1)).tolist()
            assert isinstance(self.perm, list)
            if self.perm_range is None:
                self.ids = self.perm
            else:
                self.ids = self.perm[self.perm_range[0] - 1 : self.perm_range[1]]
        else:
            raise ValueError(
                "Must provide either `ids`, `id_range`, `perm_range`, or `perm`."
            )
        print(f"Sample IDs to load: {self.ids}")


@dataclass(kw_only=True)
class PCMDataParameter(LIONParameter):
    """Parameters for the PCM dataset and data loader.

    Attributes:
        root: Root directory containing the synthetic PNG data.
        categories: List of categories to load.
        samples: Indicate the samples to load.
        data_type: Type of the data.
        j_order: Order of the sample. If None, later code needs to infer from data.

    Todo:
        Consider cases where the data is not square or side lengths are not powers of 2.
    """

    root: Path = PCM_SYNTHETIC_PNG_PATH
    categories: list[PCMDataCategory] | None = None
    samples: SamplesParameter | None = None
    data_type: PCMDataType
    j_order: int | None = None

    def __post_init__(self):
        """Post-processing.

        - Convert to correct types in case object was loaded incorrectly.
        """
        self.root = Path(self.root)

    @property
    def n(self) -> int:
        """Return the side length of the sample (2^J).

        Assuming image is square and `j_order` is provided.
        """
        assert self.j_order is not None
        return 1 << self.j_order


@dataclass(kw_only=True)
class PCMScalingParameter(LIONParameter):
    """Parameters for scaling PCM data.

    Attributes:
        tests_scale_ground_truth: Whether to scale the ground truth for testing.
        is_out_of_distribution: Whether the sample is out-of-distribution.
        r_high: High threshold for scaling.
        r_low: Low threshold for scaling.
        fill_value: Value to fill in for missing data.
        scale_eps: Epsilon value for scaling.
        divide_by: Value to divide by for scaling.
        data_sign: Sign correction applied before model scaling.
        requires_scaling: Whether scaling is required.
    """

    tests_scale_ground_truth: bool = False
    is_out_of_distribution: bool = False
    r_high: float | None = None
    r_low: float | None = None
    fill_value: float = 0.0
    scale_eps: float = 1e-12
    divide_by: float | None = None
    data_sign: float = 1.0
    requires_scaling: bool = False

    def scale(self, data: DataType) -> DataType:
        """Scale the data."""
        if self.r_low is None or self.r_high is None:
            return data
        return (data - self.r_low) / max(self.r_high - self.r_low, self.scale_eps)  # type: ignore

    def rescale(self, data: DataType) -> DataType:
        """Re-scale the data back."""
        if self.r_low is None or self.r_high is None:
            return data
        return data * max(self.r_high - self.r_low, self.scale_eps) + self.r_low  # type: ignore


# TODO: Put physics config in `Experiment` class?
@dataclass(kw_only=True)
class PCMSamplingParameter(LIONParameter):
    """Sub-sampling PCM data.

    Attributes:
        sampling_ratios: Ratios for sampling.
        randomising_scheme: Scheme for randomising the sampling.
        coarse_j_values: Coarse j values for sampling.
        coarse_j_offsets_from_j_order: Offsets from j order for coarse j values.
            Used if `coarse_j_values` is None.
    """

    sampling_ratios: list[float] = field(default_factory=lambda: [0.2])
    randomising_scheme: Literal["uniform", "multilevel"] | None = "uniform"
    coarse_j_values: list[int] | None = None
    coarse_j_offsets_from_j_order: list[int] | None = None


@dataclass(kw_only=True)
class PCMSingleSampleParameter(LIONParameter):
    """Input data configuration.

    Attributes:
        data_filename: Name of the file containing the data.
        ground_truth_filename: Name of the file containing the ground truth.
    """

    data_filename: str
    ground_truth_filename: str
