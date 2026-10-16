"""2D orthogonal wavelet transform with periodised Daubechies wavelets."""

from __future__ import annotations

import numpy as np
import pywt
import torch

from LION.operators.Operator import Operator


class Wavelet2D(Operator):
    """2D orthogonal wavelet transform with periodised Daubechies wavelets.

    Args:
        shape: Image shape (H, W).
        wavelet_name: PyWavelets wavelet name, for example "db4".
        level: Decomposition level. If None, uses the maximum valid level.
        mode: Signal extension mode. "periodization" gives an orthogonal transform.
        device: Device on which tensors are returned.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        wavelet_name: str = "db4",
        level: int | None = None,
        mode: str = "periodization",
        device: str | torch.device = "cpu",
    ):
        """2D orthogonal wavelet transform with periodised Daubechies wavelets."""
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
        """Apply the forward wavelet transform.

        Wavelet analysis: image -> flat coefficient vector.

        Args:
            x: Input tensor whose last two dimensions have shape (H, W).
            out: Legacy for tomosipo ``to_autograd``. Just ignore.

        Returns:
            Tensor whose last dimension contains the flat wavelet coefficients.
        """
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wavelet analysis operator.

        .. note::
            Prefer calling the instance of the Wavelet2D operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        x_np = x.detach().cpu().numpy().astype(np.float32)
        batch_shape = x_np.shape[:-2]
        x_np = x_np.reshape(-1, *self.shape)

        w = []
        for image in x_np:
            # TODO: How to use the PyTorch backend and still ensure adjointness?
            coeffs = pywt.wavedec2(
                image, wavelet=self.wavelet, mode=self.mode, level=self.level
            )
            arr, _ = pywt.coeffs_to_array(coeffs)
            w.append(arr.reshape(-1))

        w = np.stack(w).reshape(*batch_shape, self.size)
        return torch.from_numpy(w.astype(np.float32)).to(self.device)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint of the wavelet analysis operator.

        The adjoint is equivalent to the inverse wavelet transform.
        TODO: Add reference.

        Args:
            y: Tensor whose last dimension contains the flat wavelet coefficients.

        Returns:
            Tensor whose last two dimensions have shape (H, W).
        """
        return self.inverse(y)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply wavelet synthesis: flat coefficient vector -> image.

        Args:
            y: Tensor whose last dimension contains the flat wavelet coefficients.

        Returns:
            Tensor whose last two dimensions have shape (H, W).
        """
        w_np = y.detach().cpu().numpy().astype(np.float32)
        batch_shape = w_np.shape[:-1]
        w_np = w_np.reshape(-1, self.size)

        x = []
        for coeff_vector in w_np:
            arr = coeff_vector.reshape(self.arr_shape)
            coeffs = pywt.array_to_coeffs(arr, self.slices, output_format="wavedec2")

            # TODO: How to use the PyTorch backend and still ensure adjointness?
            x_np = pywt.waverec2(coeffs, wavelet=self.wavelet, mode=self.mode)
            x_np = x_np.astype(np.float32)
            # Crop in case of off-by-one due to padding
            H, W = self.shape
            x.append(x_np[:H, :W])

        x = np.stack(x).reshape(*batch_shape, *self.shape)
        return torch.from_numpy(x).to(self.device)

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Return the shape of the input domain."""
        return self.shape

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of the output range."""
        return (self.size,)
