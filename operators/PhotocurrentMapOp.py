"""Photocurrent mapping operator using subsampled WHT and dyadic permutation."""

from __future__ import annotations

import numpy as np
import torch
from spyrit.core.torch import fwht, ifwht

from LION.operators.Operator import Operator


def normal_to_dyadic_permutation(J: int) -> np.ndarray:
    nbits = 2 * J
    n = 1 << nbits  # total length = 2^(2J)
    # ---- dyadic-by-scales permutation (dtype-safe bit ops) ----
    # v = dec2bin(0:n-1, 2J)=='1'; v = fliplr(v);
    k = np.arange(n, dtype=np.uint64)
    shifts = np.arange(nbits, dtype=np.uint64)  # <-- make shifts unsigned
    v = ((k[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)  # columns: LSB..MSB

    # p = reshape(1:2J, 2, J)'; p = p(:)  (1-based in MATLAB) -> odd columns then even (0-based here)
    p = np.r_[np.arange(0, nbits, 2), np.arange(1, nbits, 2)]
    v = v[:, p]

    # perm = v * [2.^((2J-1):-1:0)]' + 1  (build weights as unsigned)
    weights = np.uint64(1) << np.arange(nbits - 1, -1, -1, dtype=np.uint64)
    permutation = (v @ weights).astype(np.int64)  # reordered index -> standard index
    return permutation


class Subsampler:
    def __init__(
        self, n: int, delta: float, coarseJ: int, rng: np.random.Generator | None = None
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        # ---- random undersampling with coarseJ fully kept ----
        m_total = int(np.ceil(delta * n))
        m1 = min(1 << (2 * coarseJ), m_total)
        m2 = m_total - m1
        if m2 > 0:
            idx_tail = rng.choice(n - m1, size=m2, replace=False) + m1
            self._subsampled_indices = np.concatenate(
                [np.arange(m1, dtype=np.int64), idx_tail.astype(np.int64)]
            )
        else:
            self._subsampled_indices = np.arange(m1, dtype=np.int64)

    @property
    def subsampled_indices(self) -> np.ndarray:
        return self._subsampled_indices


class PhotocurrentMapOp(Operator):
    """Photocurrent mapping operator using subsampled WHT and dyadic permutation.

    Parameters
    ----------
    J : int
        The exponent such that the image size is (2^J, 2^J).
    subsampler : Subsampler
        The subsampler defining the measurement indices.
    wht_dim : int, optional
        The dimension along which to apply the WHT. Default is -1 (last dimension).
    device : str or torch.device
        Device where tensors are placed.
    """

    def __init__(
        self,
        J: int,
        subsampler: Subsampler,
        wht_dim: int = -1,
        device: torch.device | str | None = None,
    ):
        """Initialize the PhotocurrentMapOp."""
        super().__init__(device=device)
        self.N = 1 << J
        self.num_pixels = self.N * self.N
        self.wht_dim = wht_dim

        # TODO: Add batch size
        self._image_shape = (self.N, self.N)
        self._data_shape = (subsampler.subsampled_indices.shape[0],)

        self.normal_to_dyadic_perm = normal_to_dyadic_permutation(J=J)
        self.meas_indices_standard = torch.tensor(
            self.normal_to_dyadic_perm[subsampler.subsampled_indices],
            dtype=torch.long,
            device=self.device,
        )

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
        return fwht(x.ravel(), order=False, dim=self.wht_dim)[
            self.meas_indices_standard
        ]

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint photocurrent mapping.

        Parameters
        ----------
        y : torch.Tensor
            The input measurements to which the adjoint photocurrent mapping is applied.

        Returns
        -------
        torch.Tensor
            The reconstructed image from the adjoint photocurrent mapping.
        """
        # Inject measurements into full standard-order Hadamard vector
        y_standard_full = y.new_zeros(self.num_pixels)
        y_standard_full.index_copy_(0, self.meas_indices_standard, y)
        # The adjoint of WHT is WHT itself
        return fwht(y_standard_full, order=False, dim=self.wht_dim).view(
            *self._image_shape
        )

    def pseudo_inv(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the pseudo-inverse photocurrent mapping.

        Parameters
        ----------
        y : torch.Tensor
            The input measurements to which the pseudo-inverse photocurrent mapping is applied.

        Returns
        -------
        torch.Tensor
            The reconstructed image from the pseudo-inverse photocurrent mapping.
        """
        # Inject measurements into full standard-order Hadamard vector
        y_standard_full = y.new_zeros(self.num_pixels)
        y_standard_full.index_copy_(0, self.meas_indices_standard, y)
        # The inverse of WHT is WHT divided by the number of elements
        return ifwht(y_standard_full, order=False, dim=self.wht_dim).view(
            *self._image_shape
        )

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Return the shape of the image domain."""
        return self._image_shape

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of the measurement range."""
        return self._data_shape
