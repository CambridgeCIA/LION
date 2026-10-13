# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : T. Trung (Troy) Vu
# =============================================================================
"""Data classes for PCM reconstruction experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from LION.data_loaders.pcm.data_primitives import (
    PCMDataParameter,
    PCMSamplingParameter,
    PCMScalingParameter,
    PCMSingleSampleParameter,
)
from LION.pcm.noise import PCMNoiseParameter, PCMNoiseSimulation
from LION.pcm_tools.pcm_sampling import make_sampling_cases
from LION.utils.parameter import LIONParameter
from LION.utils.paths import TTV22_PATH
from LION.utils.plot_helper import PlotHelper

DEFAULT_RESULTS_DIR = TTV22_PATH / "pcm/results"
# DEFAULT_RESULTS_DIR = Path("../pcm_results")
# DARK_NOISE_DICT = {}


# @dataclass
# class PCMSyntheticDataExperiment(PCMExperiment):
#     """PCM reconstruction experiment with synthetic data.

#     Fixed values (also see `PCMRecon.default_parameters()`):
#         - ...

#     ## File Structure

#     ```
#     .
#     ├── Input Images
#     │   ├── png
#     │   └── txt
#     └── Synthetic Data
#         ├── CIGS
#         ├── Organic
#         ├── Perovskite
#         ├── Reference
#         └── Silicone
#     ```

#     """


@dataclass(kw_only=True)
class PCMSingleSampleRuntimeParameter(LIONParameter):
    """Runtime and output configuration.

    Attributes:
        root_output_dir: Experiment outputs directory. A subdirectory is created for each trial.
        device: Device to use for computation.
        num_trials: Number of trials to run. Trials are indexed from 0 to num_trials-1.
        num_trials_skip: Number of first trials to skip.
            E.g. set to num_trials_skip to 1 to skip trial 0 and start from trial 1.
            Since each trial generates a random seed deterministically based on its index,
            this is useful for resuming an experiment after a crash or interruption.
        skip_exceptions: Whether to skip test cases that raise exceptions during reconstruction.
    """

    root_output_dir: Path
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    num_trials: int = 1
    num_trials_skip: int = 0
    skip_exceptions: bool = False


@dataclass(kw_only=True)
class PCMLogParameter(LIONParameter):
    """Logging configuration.

    Attributes:
        save_config: Whether to save the experiment configuration.
        save_metrics: Whether to save reconstruction metrics to CSV.
        save_plots: Whether to save images and figures.
        save_recon: Whether to save the reconstructed images.
        save_diff: Whether to save difference image between target and reconstruction.
        save_masks: Whether to save the sampling masks.
        save_zero_filled: Whether to save the zero-filled reconstructions.
    """

    save_config: bool = True
    save_metrics: bool = True
    save_plots: bool = False
    save_recon: bool = False
    save_diff: bool = False
    save_masks: bool = False
    save_zero_filled: bool = False


@dataclass(kw_only=True)
class PCMExperimentParameter(LIONParameter):
    """Top-level configuration for one PCM experiment run for a dataset.

    Attributes:
        name: Name of the experiment.
        data: Parameters for the PCM data (loading, simulation, ...).
        this_sample: Parameters for the sample that is running.
        scaling: Parameters for scaling PCM data.
        sampling: Parameters for sub-sampling PCM data.
        uses_normalized_op: Whether to use a normalized PCM operator.
        noise: Parameters for the PCM noise simulation experiment.
        pnp_admm: Parameters for the PnP-ADMM reconstructor.
        precision_builder: Parameters for the precision builder.
        runtime: Runtime and output configuration.
        plot: Parameters for plotting and visualization.
    """

    id: str
    data: PCMDataParameter
    this_sample: PCMSingleSampleParameter | None = None
    scaling: PCMScalingParameter
    sampling: PCMSamplingParameter
    uses_normalized_op: bool = True
    noise: PCMNoiseSimulation | None = None

    methods: dict[str, LIONParameter] = field(default_factory=dict)

    runtime: PCMSingleSampleRuntimeParameter
    log: PCMLogParameter = field(default_factory=PCMLogParameter)
    plot: PlotHelper
    base_seed: int = 0


# Designs: small-but-many (this) vs. big-but-few (current LION)
#   - Support both?
#   - Turn small-but-many into big-but-few with custom data loader logics:
#     - Pass a list of (sampling_ratio, coarse_J) pairs to the data loader,
#       then iterate over them using the data loader.
#       For logging, data loader can also return the sub-experiment name.
def make_test_cases(config: PCMExperimentParameter) -> list[tuple[float, int]]:
    """Generate sampling/coarse-level test cases."""
    j_order = config.data.j_order
    assert j_order is not None
    return make_sampling_cases(j_order=j_order, sampling=config.sampling)
