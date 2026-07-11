"""Composite linear operator A = Phi Psi^{-1} and its adjoint for compressed sensing."""

from __future__ import annotations

import torch

from LION.operators.Operator import Operator
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp


class CompositeOp(Operator):
    r"""Composite linear operator :math:`A = \Phi \Psi^{-1}` and its adjoint.

    Parameters
    ----------
    wavelet : Wavelet2D
        Wavelet transform object.
    phi : PhotocurrentMapOp
        Photocurrent mapping operator.
    device : str or torch.device, optional
        Device for computations. If None, uses wavelet.device.

    References
    ----------
    .. [Koutsourakis2021] G. Koutsourakis, A. Thompson, and J. C. Blakesley,
        "Toward Megapixel Resolution Compressed Sensing Current Mapping of
        Photovoltaic Devices Using Digital Light Processing", Solar RRL,
        5(11):2100467, 2021. doi:10.1002/solr.202100467
    """

    def __init__(
        self,
        wavelet: Operator,
        phi: PhotocurrentMapOp,
        device: str | torch.device | None = None,
    ):
        """Initialize the CompositeOp A = Phi Psi^{-1}."""
        self.wavelet = wavelet
        self.phi = phi
        self.device = device

    def __call__(self, w: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the forward projection A = Phi Psi^{-1}.

        Parameters
        ----------
        w : torch.Tensor
            Wavelet coefficients, shape (Nw,).
        out : None
            Legacy for tomosipo ``to_autograd``. Just ignore.

        Returns
        -------
        torch.Tensor
            Predicted measurements, shape (M,).
        """
        return self.forward(w)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Apply the forward projection A = Phi Psi^{-1}.

        .. note::
            Prefer calling the instance of the CompositeOp operator as ``operator(w)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if not isinstance(w, torch.Tensor):
            raise TypeError(f"Input w must be a torch.Tensor, got {type(w)}")

        w = w.to(self.device)
        x = self.wavelet.inverse(w)  # (H, W) on self.device

        # If PhotocurrentMapOp expects different shape, reshaping is applied here
        y = self.phi.forward(x)
        return y

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        """Apply A^T = Psi Phi^T.

        Parameters
        ----------
        r : torch.Tensor
            Residual in measurement space, shape (M,).

        Returns
        -------
        g : torch.Tensor
            Gradient in wavelet coefficient space, shape (Nw,).
        """
        r = r.to(self.device)
        x_back = self.phi.adjoint(r)  # expected shape (H, W)
        g = self.wavelet.forward(x_back)
        return g

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Return the shape of the wavelet coefficient domain."""
        # Opposite of range_shape of wavelet since wavelet is Psi^{-1}
        return self.wavelet.range_shape

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of the measurement range."""
        # Opposite of domain_shape of wavelet since wavelet is Psi^{-1}
        return self.wavelet.domain_shape
