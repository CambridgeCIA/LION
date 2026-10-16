"""Select one synthetic PCM dataset image for the standard single-sample runner."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from LION.data_loaders.pcm.data_primitives import (
    PCMSingleSampleParameter,
    SamplesParameter,
)
from LION.experiments.pcm_experiments import PCMExperimentParameter
from LION.experiments.pcm_presets.presets import QUICK_METHODS

_SLIDE_METHOD_IDS = ("pnp_admm_drunet", "pnp_gs_deepinv_drunet", "spgl1")


def is_synthetic_dataset(config: PCMExperimentParameter) -> bool:
    """Identify an image dataset preset, rather than a named single-image preset."""
    return (
        config.data.data_type == "image"
        and config.data.categories is not None
        and config.data.samples is not None
    )


def with_iteration_methods(preset: PCMExperimentParameter) -> PCMExperimentParameter:
    """Make PnP and SPGL1 available without changing the global dataset preset."""
    config = deepcopy(preset)
    if is_synthetic_dataset(config):
        for method_id in _SLIDE_METHOD_IDS:
            config.methods.setdefault(method_id, deepcopy(QUICK_METHODS[method_id]))
    return config


def select_synthetic_sample(
    config: PCMExperimentParameter, category: str, sample_id: int
) -> str:
    """Set this_sample and return the dataset runner's exact seed-bearing sample name.

    A sample ID is a filename suffix (image0288.png), not an index into the
    preset's possibly short, permuted sample list.
    """
    if not is_synthetic_dataset(config):
        raise ValueError(f"Preset {config.id!r} is not a synthetic PNG dataset preset.")
    if category not in config.data.categories:
        raise ValueError(
            f"Category {category!r} is not in preset categories: {config.data.categories}"
        )
    if sample_id < 1 or sample_id > 9999:
        raise ValueError("Sample ID must be between 1 and 9999.")

    filename = f"{category}/image{sample_id:04d}.png"
    image_path = Path(config.data.root).expanduser() / filename
    if not image_path.is_file():
        raise FileNotFoundError(f"Synthetic PCM image not found: {image_path}")

    config.data.categories = [category]
    config.data.samples = SamplesParameter(ids=[sample_id])
    config.this_sample = PCMSingleSampleParameter(
        data_filename=filename, ground_truth_filename=filename
    )
    return f"artifacts/{category}/sample_{sample_id:04d}"
