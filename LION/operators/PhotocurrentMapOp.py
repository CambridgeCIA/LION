"""Photocurrent mapping operator using subsampled WHT and dyadic permutation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from spyrit.core.torch import fwht, ifwht

from LION.operators.Operator import Operator
from LION.pcm_tools.ordering import normal_to_dyadic_permutation
from LION.utils.common_primitives import DeviceLikeType


class PartialPermutationOp(Operator):
    """Partial permutation operator that selects and reorders a subset of elements from a vector."""

    def __init__(
        self,
        indices: Sequence[int] | np.ndarray,
        vector_len: int,
        device: DeviceLikeType | None = None,
    ):
        """Initialize the partial permutation operator.

        This represents a matrix multiplication with a matrix P with only 0 and 1 entries,
        and each row or column has at most one non-zero element.

        Args:
            indices: The indices of the elements to be taken.
            vector_len: The length of the input vector.
            device: Device where tensors are placed.
        """
        super().__init__(device=device)
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        gram_mask_tensor = torch.zeros(vector_len, dtype=torch.bool, device=self.device)
        gram_mask_tensor[indices_tensor] = True
        # Register buffers to ensure they are moved to the correct device when the operator is moved.
        # Note: buffers are not trainable parameters.
        # These buffers are not persistent, meaning they won't be saved in the state_dict.
        self.register_buffer("_indices", indices_tensor, persistent=False)
        self.register_buffer("_gram_mask", gram_mask_tensor, persistent=False)
        self._domain_shape = (vector_len,)
        self._range_shape = (len(indices),)

    @property
    def indices(self) -> torch.Tensor:
        """The indices of the elements to be taken."""
        return self._indices

    @property
    def gram_mask(self) -> torch.Tensor:
        """The mask for the Gram (A^T A) operator."""
        return self._gram_mask

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the forward operator.

        Args:
            x: Tensor (representing set of 1D vectors) whose last dimension has the domain size.
            out: Legacy for tomosipo ``to_autograd``. Just ignore.

        Returns:
            Tensor Px, applied along the last dimension.
        """
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward operator.

        .. note::
            Prefer calling the instance of the PartialPermutationOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`
        """
        return x[..., self.indices]

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint, i.e. transpose, operator.

        Args:
            y: Tensor (representing set of 1D vectors) whose last dimension has the range size.

        Returns:
            Tensor P^T y, applied along the last dimension.
        """
        x = y.new_zeros(*y.shape[:-1], self._domain_shape[-1])
        x.index_copy_(-1, self.indices, y)
        return x

    def gram(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Gram operator, i.e. P^T P.

        Args:
            x: Tensor (representing set of 1D vectors) whose last dimension has the domain size.

        Returns:
            Tensor P^T P x, applied along the last dimension.
        """
        return x * self.gram_mask

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """The shape of the domain."""
        return self._domain_shape

    @property
    def range_shape(self) -> tuple[int, ...]:
        """The shape of the range."""
        return self._range_shape


class WHTOp(Operator):
    def __init__(
        self,
        dim: int = -1,
        sqrt_n: int | None = None,
        device: DeviceLikeType | None = None,
    ):
        """Initialize the Walsh-Hadamard Transform operator.

        Args:
            dim: The dimension along which to apply the WHT. Default is -1 (last dimension).
            sqrt_n: The square root of the number of elements. If None, no normalization is applied.
            device: Device where tensors are placed.
        """
        super().__init__(device=device)
        self.dim = dim
        self.sqrt_n = sqrt_n

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the WHT operator.

        Args:
            x: The input vector.
            out: Legacy for tomosipo ``to_autograd``. Just ignore.

        Returns:
            Vector Wx.
        """
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the WHT operator.

        .. note::
            Prefer calling the instance of the WalshHadamardTransformOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`
        """
        if self.sqrt_n is not None:
            return fwht(x, order=False, dim=self.dim) / self.sqrt_n
        else:
            return fwht(x, order=False, dim=self.dim)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint of the WHT operator.

        Args:
            y: The input vector.

        Returns:
            Vector W^T y.
        """
        if self.sqrt_n is not None:
            return fwht(y, order=False, dim=self.dim) / self.sqrt_n
        else:
            return fwht(y, order=False, dim=self.dim)

    def inv(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the inverse of the WHT operator.

        Args:
            y: The input vector.

        Returns:
            Vector W^{-1} y.
        """
        if self.sqrt_n is not None:
            return fwht(y, order=False, dim=self.dim) / self.sqrt_n
        else:
            return ifwht(y, order=False, dim=self.dim)


class PhotocurrentMapOp(Operator):
    """Photocurrent mapping normalized operator using subsampled WHT and dyadic permutation."""

    def __init__(
        self,
        j_order: int,
        s_indices: Sequence[int] | np.ndarray | None = None,
        wht_dim: int = -1,
        normalized: bool = True,
        device: DeviceLikeType | None = None,
    ):
        """Initialize the PhotocurrentMapOp.

        Args:
            j_order: The exponent such that the image size is (2^j_order, 2^j_order).
            s_indices: The indices of the measurements to be taken, in coarse-to-fine (dyadic) order.
            wht_dim: The dimension along which to apply the WHT. Default is -1 (last dimension).
            normalized: Whether to normalize the WHT by dividing by the number of elements. Default is True.
            device: Device where tensors are placed.
        """
        super().__init__(device=device)
        side_len = 1 << j_order
        self.num_pixels = side_len * side_len

        if s_indices is None:
            s_indices = np.arange(self.num_pixels, dtype=np.int64)
        p_indices = normal_to_dyadic_permutation(j_order=j_order)
        sp_indices = p_indices[s_indices]

        self.s = PartialPermutationOp(
            indices=s_indices, vector_len=self.num_pixels, device=self.device
        )
        self.p = PartialPermutationOp(
            indices=p_indices, vector_len=self.num_pixels, device=self.device
        )
        self.sp = PartialPermutationOp(
            indices=sp_indices, vector_len=self.num_pixels, device=self.device
        )

        self.w = WHTOp(
            dim=wht_dim, sqrt_n=side_len if normalized else None, device=self.device
        )

        self._image_shape = (side_len, side_len)
        self._data_shape = (len(s_indices),)

    def vectorize(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorize the last two image dimensions.

        Args:
            x: Tensor (representing set of 2D images) whose last two dimensions are the image dimensions.

        Returns:
            Tensor with the last two dimensions flattened.
        """
        # return x.view(-1)
        return x.flatten(start_dim=-2)

    def unvectorize(self, x: torch.Tensor) -> torch.Tensor:
        """Unvectorize the last dimension to the image shape.

        Args:
            x: Tensor (representing set of 1D vectors) whose last dimension contains the vectorized image.

        Returns:
            Tensor whose last two dimensions are the image dimensions.
        """
        return x.unflatten(-1, self._image_shape)

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        r"""Apply the forward photocurrent mapping.

        Args:
            x: Tensor (representing set of 2D images) whose last two dimensions have the domain shape.
            out: Legacy for tomosipo ``to_autograd``. Just ignore.

        Returns:
            Tensor whose last dimension is SPW x_vectorized.
        """
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward photocurrent mapping.

        .. note::
            Prefer calling the instance of the PhotocurrentMapOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`
        """
        # SPW x_vectorized
        return self.sp(self.w(self.vectorize(x)))

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint photocurrent mapping.

        Args:
            y: Tensor (representing set of 1D vectors) whose last dimension has the range shape.

        Returns:
            Tensor whose last two dimensions have the domain shape.
        """
        # W^T S^T P^T y, then reshape the last dimension to the image shape
        return self.unvectorize(self.w.adjoint(self.sp.adjoint(y)))

    def pseudo_inv(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the pseudo-inverse photocurrent mapping.

        Args:
            y: Tensor (representing set of 1D vectors) whose last dimension has the range shape.

        Returns:
            Tensor whose last two dimensions have the domain shape.
        """
        return self.unvectorize(self.w.inv(self.sp.adjoint(y)))

    @property
    def normalization_factor(self) -> float:
        """Return the multiplicative WHT normalization of this operator."""
        if self.w.sqrt_n is None:
            return 1.0
        return 1.0 / self.w.sqrt_n

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """The shape of one sample in the image domain."""
        return self._image_shape

    @property
    def range_shape(self) -> tuple[int, ...]:
        """The shape of one sample in the measurement range."""
        return self._data_shape
