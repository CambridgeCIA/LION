# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Thanh Trung (Troy) Vu
# =============================================================================
"""2D orthogonal wavelet transform with periodised Daubechies wavelets."""

from __future__ import annotations

import numpy as np
import pywt
import torch

from LION.operators.Operator import Operator


class Wavelet2D(Operator):
    """The 2D orthogonal wavelet transform (PyWavelets) of images of ``shape``
    (H, W), as an operator: images to a flat vector of coefficients.
    ``wavelet_name``: e.g. "db4"; ``level``: the decomposition level (None: the
    largest valid); ``mode``: the signal extension ("periodization" makes it
    orthogonal); ``device``: where its tensors are returned."""

    def __init__(
        self,
        shape: tuple[int, int],
        wavelet_name: str = "db4",
        level: int | None = None,
        mode: str = "periodization",
        device: str | torch.device = "cpu",
    ):
        super().__init__(device=device)
        self.shape = shape
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.mode = mode

        if level is None:
            max_level = pywt.dwt_max_level(min(self.shape), self.wavelet.dec_len)
            self.level = max_level
        else:
            self.level = int(level)

        # Build coefficient layout by running once on zeros
        x0 = np.zeros(self.shape, dtype=np.float32)
        coeffs = pywt.wavedec2(
            x0,
            wavelet=self.wavelet,
            mode=self.mode,
            level=self.level,
        )
        arr, self.slices = pywt.coeffs_to_array(coeffs)
        self.arr_shape = arr.shape
        self.size = arr.size

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """The wavelet coefficients (Nw,) of the image ``x`` (H, W). ``out``:
        unused, kept for the ``__call__(x, out)`` signature."""
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The wavelet coefficients (Nw,) of the image ``x`` (H, W). Call the
        operator, ``op(x)``, rather than this method (as for torch modules)."""
        if x.dim() != 2:
            raise ValueError(
                f"Expected x with 2 dims (H, W), got shape {tuple(x.shape)}"
            )

        x_np = x.detach().cpu().numpy().astype(np.float32)

        # TODO: How to use the PyTorch backend and still ensure adjointness?
        coeffs = pywt.wavedec2(
            x_np,
            wavelet=self.wavelet,
            mode=self.mode,
            level=self.level,
        )
        arr, _ = pywt.coeffs_to_array(coeffs)
        w = torch.from_numpy(arr.astype(np.float32)).to(self.device)
        return w.reshape(-1)

    def adjoint(self, w: torch.Tensor) -> torch.Tensor:
        """The adjoint of the transform: its inverse, as it is orthogonal (with
        ``mode="periodization"``). The image (H, W) of the coefficients ``w``.
        """
        return self.inverse(w)

    def inverse(self, w: torch.Tensor) -> torch.Tensor:
        """The image (H, W) of the coefficients ``w`` (Nw,): the inverse
        transform."""
        w_np = w.detach().cpu().numpy().astype(np.float32)
        arr = w_np.reshape(self.arr_shape)
        coeffs = pywt.array_to_coeffs(arr, self.slices, output_format="wavedec2")

        # TODO: How to use the PyTorch backend and still ensure adjointness?
        x_np = pywt.waverec2(
            coeffs,
            wavelet=self.wavelet,
            mode=self.mode,
        )
        x_np = x_np.astype(np.float32)
        # Crop in case of off-by-one due to padding
        H, W = self.shape
        x_np = x_np[:H, :W]
        return torch.from_numpy(x_np).to(self.device)

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Return the shape of the input domain."""
        return self.shape

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of the output range."""
        return self.arr_shape
