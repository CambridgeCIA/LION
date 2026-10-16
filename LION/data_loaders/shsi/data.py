"""Load the NPZ measurement format provided by the sHSI team."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SHSIInput:
    """Observed sHSI data and its sampling mask.

    Arrays use the original (H, W, D) cube shape, or an (N, D) matrix.
    The optional ground truth is used only for evaluation.
    """

    measurement: np.ndarray
    mask: np.ndarray
    sigma: float
    ground_truth: np.ndarray | None = None


def load_shsi_npz(path: str | Path) -> SHSIInput:
    """Load sHSI measurements from a reference NPZ file.

    The file must contain `measurement`, `mask` and `sigma`. It may
    contain `gt`, which is kept separate from the measured input.
    Numeric array dtypes are preserved.
    """
    with np.load(path, allow_pickle=False) as data:
        required = {"measurement", "mask", "sigma"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Missing sHSI NPZ keys: {', '.join(sorted(missing))}")
        measurement = np.asarray(data["measurement"])
        raw_mask = np.asarray(data["mask"])
        sigma_array = np.asarray(data["sigma"])
        ground_truth = np.asarray(data["gt"]) if "gt" in data else None

    if measurement.ndim not in (2, 3):
        raise ValueError("measurement must have shape (N, D) or (H, W, D)")
    if measurement.shape != raw_mask.shape:
        raise ValueError("measurement and mask must have the same shape")
    if not np.issubdtype(measurement.dtype, np.number):
        raise ValueError("measurement must contain numeric values")
    if not np.isfinite(measurement).all():
        raise ValueError("measurement must contain only finite values")
    if not np.all((raw_mask == 0) | (raw_mask == 1)):
        raise ValueError("mask entries must be 0 or 1")
    if sigma_array.size != 1:
        raise ValueError("sigma must be a scalar")

    sigma = float(sigma_array.reshape(-1)[0])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if ground_truth is not None:
        if ground_truth.shape != measurement.shape:
            raise ValueError("ground truth and measurement must have the same shape")
        if not np.isfinite(ground_truth).all():
            raise ValueError("ground truth must contain only finite values")

    return SHSIInput(
        measurement=measurement,
        mask=raw_mask.astype(bool, copy=False),
        sigma=sigma,
        ground_truth=ground_truth,
    )
