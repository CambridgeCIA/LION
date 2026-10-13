"""Measure split-Hadamard transform for PCM."""

import torch

from LION.operators.PhotocurrentMapOp import WHTOp
from LION.utils.math import is_power_of_two


def measure_split_hadamard_transform(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute binary split-Hadamard measurements along the last dimension.

    Args:
        x: Signal tensor whose last dimension has power-of-two length.

    Returns:
        Positive-pattern and negative-pattern measurements with the same shape as ``x``.
    """
    n = int(x.shape[-1])

    if not is_power_of_two(n):
        raise ValueError("Input length must be a power of two.")

    hx = WHTOp(dim=-1, sqrt_n=None, device=x.device)(x)
    x_sum = torch.sum(x, dim=-1, keepdim=True)

    y_plus = 0.5 * (x_sum + hx)
    y_minus = 0.5 * (x_sum - hx)

    return y_plus, y_minus
