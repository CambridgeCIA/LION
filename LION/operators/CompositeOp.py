# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Thanh Trung (Troy) Vu
# =============================================================================
"""Composite linear operator A = Phi Psi^{-1} and its adjoint for compressed sensing."""

from __future__ import annotations

import torch

from LION.operators.Operator import Operator


class CompositeOp(Operator):
    """The composite operator A = Phi Psi^-1 and its adjoint: the measurement
    operator ``phi`` (e.g. photocurrent mapping's,
    ``LION.operators.pcm.PhotocurrentMapOp``) applied to the image of wavelet
    coefficients (``wavelet``, a ``Wavelet2D``), for sparse recovery in the
    wavelet domain (Koutsourakis, Thompson and Blakesley, "Toward Megapixel
    Resolution Compressed Sensing Current Mapping of Photovoltaic Devices Using
    Digital Light Processing", Solar RRL 5(11):2100467, 2021,
    doi:10.1002/solr.202100467). ``device``: where it computes (None: unset).
    """

    def __init__(
        self,
        wavelet: Operator,
        phi: Operator,
        device: str | torch.device | None = None,
    ):
        super().__init__(device=device)
        self.wavelet = wavelet
        self.phi = phi
        self.device = device

    def __call__(self, w: torch.Tensor, out=None) -> torch.Tensor:
        """The measurements (M,) of the wavelet coefficients ``w`` (Nw,): Phi
        Psi^-1 w. ``out``: unused, kept for the ``__call__(x, out)`` signature.
        """
        return self.forward(w)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """The measurements (M,) of the wavelet coefficients ``w`` (Nw,): Phi
        Psi^-1 w. Call the operator, ``op(w)``, rather than this method (as for
        torch modules)."""
        if not isinstance(w, torch.Tensor):
            raise TypeError(f"Input w must be a torch.Tensor, got {type(w)}")

        w = w.to(self.device)
        x = self.wavelet.inverse(w)  # (H, W) on self.device

        # If phi expects a different shape, reshaping is applied here
        y = self.phi.forward(x)
        return y

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        """Psi Phiᵀ r: the gradient, in wavelet coefficients (Nw,), of a
        residual ``r`` (M,) in the measurements."""
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
        return self.phi.range_shape
