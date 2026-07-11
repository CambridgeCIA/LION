"""2D orthogonal wavelet transform with periodised Daubechies wavelets."""

from __future__ import annotations

import numpy as np
import pywt
import torch

from LION.operators.Operator import Operator


class Wavelet2D(Operator):
    """2D orthogonal wavelet transform with periodised Daubechies wavelets.

    Parameters
    ----------
    shape : tuple of int
        Image shape (H, W).
    wavelet_name : str
        PyWavelets wavelet name, for example 'db4'.
    level : int or None
        Decomposition level. If None, uses the maximum valid level.
    mode : str
        Signal extension mode. 'periodization' gives an orthogonal transform.
    device : str or torch.device
        Device on which tensors are returned.
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
        """Apply the forward wavelet transform."

        Wavelet analysis: image -> flat coefficient vector.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (H, W).
        out : None
            Legacy for tomosipo ``to_autograd``. Just ignore.

        Returns
        -------
        torch.Tensor
            Flat wavelet coefficient vector of shape (Nw,).
        """
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wavelet analysis operator.

        .. note::
            Prefer calling the instance of the Wavelet2D operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
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
        """Adjoint of the wavelet analysis operator.

        The adjoint is equivalent to the inverse wavelet transform.
        TODO: Add reference.

        Parameters
        ----------
        w : torch.Tensor
            Flat wavelet coefficient vector of shape (Nw,).

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape (H, W).
        """
        return self.inverse(w)

    def inverse(self, w: torch.Tensor) -> torch.Tensor:
        """Wavelet synthesis: flat coefficient vector -> image.

        Parameters
        ----------
        w : torch.Tensor
            Flat wavelet coefficient vector of shape (Nw,).

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape (H, W).
        """
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
