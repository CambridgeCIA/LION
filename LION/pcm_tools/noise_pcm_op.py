"""Reconstruct a PCM image."""

from __future__ import annotations

import numpy as np
import torch
from spyrit.core.torch import fwht, ifwht

from LION.operators.Operator import Operator
from LION.pcm_tools.noise_utils import (
    generate_order_sorting_matrix,
    torch_reshape_fortran_2d,
)


class PCMOp(Operator):
    """Reconstruct a PCM image from sorted Hadamard measurements."""

    def __init__(
        self,
        n_row: int,
        n_col: int,
        n_rows_orig: int,
        n_cols_orig: int,
        iy: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ):
        """Initialize the PCMOp."""
        super().__init__(device=device)
        self.n_row = n_row
        self.n_col = n_col
        self.n_rows_orig = n_rows_orig
        self.n_cols_orig = n_cols_orig
        if iy is not None:
            self.iy = iy.to(self.device)
        else:
            iy_np = generate_order_sorting_matrix(
                o_col=int(np.ceil(np.log2(n_cols_orig))),
                o_row=int(np.ceil(np.log2(n_rows_orig))),
            )
            self.iy = torch.as_tensor(iy_np, dtype=torch.long, device=device)

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the forward photocurrent mapping.

        Parameters
        ----------
        x : torch.Tensor
            The input image to which the photocurrent mapping is applied.
        out : None
            Legacy for tomosipo ``to_autograd``. Just ignore.

        Returns
        -------
        torch.Tensor
            The measurements obtained from the photocurrent mapping.
        """
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward photocurrent mapping.

        .. note::
            Prefer calling the instance of the PhotocurrentMapOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`
        """
        # Forward Hadamard
        return fwht(x.ravel(), order=False, dim=-1)[self.iy]

    def adjoint(self, y_sorted: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint photocurrent mapping.

        Args:
            y_sorted: Measurements in ordered-pattern order.

        Returns
        -------
            Reconstructed cropped image.
        """
        iy = self.iy
        n_row = self.n_row
        n_col = self.n_col
        n_rows_orig = self.n_rows_orig
        n_cols_orig = self.n_cols_orig

        n = int(iy.numel())

        y_h = torch.zeros(
            n,
            dtype=y_sorted.dtype,
            device=y_sorted.device,
        )
        y_h[iy] = y_sorted[:n]

        # The adjoint of WHT is WHT itself
        x = fwht(y_h.reshape(1, -1), order=False, dim=-1)[0]

        image = torch_reshape_fortran_2d(x, n_row, n_col)

        return image[:n_rows_orig, :n_cols_orig]

    def pseudo_inv(self, y_sorted: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct an image from sorted Hadamard measurements.

        Args:
            y_sorted: Measurements in ordered-pattern order.

        Returns
        -------
            Reconstructed cropped image.
        """
        iy = self.iy
        n_row = self.n_row
        n_col = self.n_col
        n_rows_orig = self.n_rows_orig
        n_cols_orig = self.n_cols_orig

        n = int(iy.numel())

        y_h = torch.zeros(
            n,
            dtype=y_sorted.dtype,
            device=y_sorted.device,
        )
        y_h[iy] = y_sorted[:n]

        # The inverse of WHT
        x = ifwht(y_h.reshape(1, -1), order=False, dim=-1)[0]

        image = torch_reshape_fortran_2d(x, n_row, n_col)

        return image[:n_rows_orig, :n_cols_orig]

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Return the shape of the image domain."""
        return (self.n_rows_orig, self.n_cols_orig)

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of the measurement range."""
        return (self.iy.numel(),)
