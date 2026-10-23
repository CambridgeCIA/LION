"""Utility functions for shape manipulation."""

from __future__ import annotations

import torch

from LION.utils.math import next_power_of_two


def pad_to_powers_of_two_torch(
    x: torch.Tensor, *, dim: tuple[int, ...] = (-2, -1), fill_value: float = 0.0
) -> torch.Tensor:
    """Pad selected dimensions to powers of two.

    Args:
        x: Input tensor.
        dim: Dimensions to pad.
        fill_value: Value used for padded elements.

    Returns:
        Padded tensor.
    """
    out_shape = list(x.shape)
    indexing = [slice(None)] * x.ndim
    for d in dim:
        size = x.shape[d]
        out_shape[d] = next_power_of_two(size)
        indexing[d] = slice(size)
    out = torch.full(out_shape, fill_value=fill_value, dtype=x.dtype, device=x.device)
    out[tuple(indexing)] = x
    return out


def torch_flatten_fortran_2d(x: torch.Tensor) -> torch.Tensor:
    """Flatten a 2D tensor like MATLAB/Fortran-order reshape.

    Args:
        x: 2D tensor.

    Returns:
        Flattened tensor.
    """
    return x.T.reshape(-1)


def torch_reshape_fortran_2d(x: torch.Tensor, n_row: int, n_col: int) -> torch.Tensor:
    """Reshape a 1D tensor like MATLAB/Fortran-order reshape.

    Args:
        x: One-dimensional tensor.
        n_row: Number of rows.
        n_col: Number of columns.

    Returns:
        Tensor with shape `(n_row, n_col)`.

    Examples:
        x = torch.tensor([1, 2, 3, 4, 5, 6])
        torch_reshape_fortran_2d(x, n_row=2, n_col=3)
        # Output:
        # tensor([[1, 3, 5],
        #         [2, 4, 6]])
    """
    return x.reshape(n_col, n_row).T
