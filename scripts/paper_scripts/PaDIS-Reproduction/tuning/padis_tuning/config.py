"""Filesystem defaults for PaDIS reconstruction tuning."""

from __future__ import annotations

import pathlib

DEFAULT_EXTERNAL_MODEL_ROOT = pathlib.Path(
    "/home/thomas/DiS/Project/Data/experiments/PaDIS/external_models"
)
DEFAULT_TUNING_ROOT = pathlib.Path(
    "/home/thomas/DiS/Project/Data/experiments/PaDIS/hparam_tuning"
)
DEFAULT_STAGED_TRAINING_ROOT = DEFAULT_TUNING_ROOT / "external_training_root"
DEFAULT_OUTPUT_ROOT = DEFAULT_TUNING_ROOT / "runs"


EXTERNAL_MODEL_LINKS = {
    "patch_lidc_default/padis_lidc_256.pt": "padis_lidc_default.pt",
    "patch_lidc_512/padis_lidc_512.pt": "padis_lidc_512.pt",
    "whole_lidc_default/whole_image_lidc_256_min_val.pt": "whole_lidc_default.pt",
    "pnp_lidc_drunet/pnp_lidc_drunet_min_val.pt": "pnp_lidc_drunet_min_val.pt",
    "pnp_lidc_drunet/pnp_lidc_drunet.pt": "pnp_lidc_drunet_min_val.pt",
}
