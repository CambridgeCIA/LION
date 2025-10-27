import numpy as np
from spyrit.core.torch import fwht, ifwht
import torch

from LION.operators import Operator


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
            self.subsampled_indices = np.concatenate(
                [np.arange(m1, dtype=np.int64), idx_tail.astype(np.int64)]
            )
        else:
            self.subsampled_indices = np.arange(m1, dtype=np.int64)

    def get_subsampled_indices(self) -> np.ndarray:
        return self.subsampled_indices


class PhotocurrentMapOp(Operator):
    def __init__(self, J: int, subsampler: Subsampler, wht_dim: int = -1):
        self.N = 1 << J
        self.num_pixels = self.N * self.N
        self.wht_dim = wht_dim

        # TODO: Check the shapes
        self._image_shape = (self.N, self.N)
        self._data_shape = (subsampler.subsampled_indices.shape[0],)

        self.normal_to_dyadic_perm = normal_to_dyadic_permutation(J=J)
        self.subsampler = subsampler

    @property
    def domain_shape(self):
        return self._image_shape

    @property
    def range_shape(self):
        return self._data_shape

    def forward(self, x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        # Forward Hadamard
        y_std = fwht(x.ravel(), order=False, dim=self.wht_dim)

        # apply reordering
        y_reordered = y_std[self.normal_to_dyadic_perm]

        y_subsampled = y_reordered[self.subsampler.get_subsampled_indices()]
        return y_subsampled

    def adjoint(self, y: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        y_reordered_full = torch.zeros(self.num_pixels, dtype=y.dtype)
        y_reordered_full[self.subsampler.get_subsampled_indices()] = y
        y_standard_full = torch.zeros(self.num_pixels, dtype=y.dtype)
        y_standard_full[self.normal_to_dyadic_perm] = y_reordered_full

        im_rec_vec: torch.Tensor = ifwht(y_standard_full, order=False, dim=self.wht_dim)
        im_rec = im_rec_vec.reshape(self._image_shape)
        return im_rec
