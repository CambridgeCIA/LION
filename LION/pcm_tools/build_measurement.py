"""Build measurement vectors from PCM measurement tables."""

from __future__ import annotations

import numpy as np


def build_real_measurement_vector(
    measurements: np.ndarray, n: int | None = None
) -> np.ndarray:
    """Build the real Hadamard measurement vector from a PCM measurement table.

    Args:
        measurements: Array with columns `PatternIndex` and `MeasuredCurrentA`.
        n: Target Hadamard vector length.

    Returns:
        Real measurement vector with length `n`.
    """
    i_pos = measurements[measurements[:, 0] > 0]
    i_neg = measurements[measurements[:, 0] < 0]

    pos_idx = i_pos[:, 0].astype(np.int64)
    neg_idx = (-i_neg[:, 0]).astype(np.int64)

    max_idx = max(pos_idx.max(initial=0), neg_idx.max(initial=0))

    pos_sum = np.bincount(pos_idx, weights=i_pos[:, 1], minlength=max_idx + 1)
    pos_count = np.bincount(pos_idx, minlength=max_idx + 1)

    neg_sum = np.bincount(neg_idx, weights=i_neg[:, 1], minlength=max_idx + 1)
    neg_count = np.bincount(neg_idx, minlength=max_idx + 1)

    pos_mean = np.divide(
        pos_sum, pos_count, out=np.zeros_like(pos_sum), where=pos_count > 0
    )
    neg_mean = np.divide(
        neg_sum, neg_count, out=np.zeros_like(neg_sum), where=neg_count > 0
    )

    i_meas = pos_mean[1:] - neg_mean[1:]
    y_real = np.r_[measurements[0, 1] - measurements[1, 1], i_meas]

    n = y_real.size if n is None else min(n, y_real.size)
    y_real = np.pad(y_real, (0, n - y_real.size))
    return y_real
