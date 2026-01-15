from __future__ import annotations

import torch


def choose_measurement_scale_factor(
    measurements: torch.Tensor,
    *,
    method: str = "rms_pow10",
    target: float = 1.0,
    min_magnitude: float = 1e-12,
) -> float:
    """Choose a scalar factor to rescale measurements to an order-1 range.

    Parameters
    ----------
    measurements
        Measurement vector b (any shape). Flattened internally.
    method
        - "rms": factor = target / rms(b)
        - "maxabs": factor = target / max(abs(b))
        - "rms_pow10": like "rms" but rounded to a power of 10.
    target
        Desired magnitude after scaling (typically 1.0).
    min_magnitude
        Lower bound to avoid division by very small values.

    Returns
    -------
    factor
        Scalar factor so that scaled_measurements = factor * measurements.
    """
    b = measurements.flatten()

    if method == "rms" or method == "rms_pow10":
        rms = torch.linalg.norm(b) / torch.sqrt(torch.tensor(b.numel(), dtype=b.dtype))
        magnitude = max(min_magnitude, float(rms))
        if method == "rms_pow10":
            exponent = -int(torch.floor(torch.log10(torch.tensor(magnitude))))
            return (10.0 ** exponent) * target
        return target / magnitude

    if method == "maxabs":
        max_abs = float(torch.max(torch.abs(b)))
        magnitude = max(min_magnitude, max_abs)
        return target / magnitude

    raise ValueError(f"Unknown method: {method}")
