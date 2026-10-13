"""Reconstruct a PCM image."""

from dataclasses import dataclass

import torch
from spyrit.core.torch import ifwht

from LION.utils.shape import torch_reshape_fortran_2d


@dataclass
class PCMReconstructor:
    """Reconstruct a PCM image from sorted Hadamard measurements.

    Attributes:
        iy: Permutation from ordered indices to Hadamard indices.
        n_row: Padded row count.
        n_col: Padded column count.
        n_rows_orig: Original row count.
        n_cols_orig: Original column count.
    """

    iy: torch.Tensor
    n_row: int
    n_col: int
    n_rows_orig: int
    n_cols_orig: int

    def reconstruct_from_sorted_measurements(
        self, y_sorted: torch.Tensor, transpose_after_reshape: bool
    ) -> torch.Tensor:
        """Reconstruct an image from sorted Hadamard measurements.

        Args:
            y_sorted: Measurements in ordered-pattern order.
            transpose_after_reshape: Whether to use the MATLAB real-measurement reshape convention.

        Returns:
            Reconstructed cropped image.
        """
        iy = self.iy
        n_row = self.n_row
        n_col = self.n_col
        n_rows_orig = self.n_rows_orig
        n_cols_orig = self.n_cols_orig

        n = int(iy.numel())

        y_h = torch.zeros(n, dtype=y_sorted.dtype, device=y_sorted.device)
        y_h[iy] = y_sorted[:n]

        x = ifwht(y_h.reshape(1, -1), order=False, dim=-1)[0]

        if transpose_after_reshape:
            image = torch_reshape_fortran_2d(x, n_col, n_row).T
        else:
            image = torch_reshape_fortran_2d(x, n_row, n_col)

        return image[:n_rows_orig, :n_cols_orig]
