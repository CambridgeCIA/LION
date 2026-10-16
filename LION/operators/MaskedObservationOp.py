"""Dense masked-observation operator for subsampled data, including sHSI cubes."""

from __future__ import annotations

import numpy as np
import torch

from LION.operators.Operator import Operator
from LION.utils.common_primitives import DeviceLikeType


class MaskedObservationOp(Operator):
    """Select observed entries while keeping the original tensor shape.

    The operator represents A(x) = M * x, where M is a fixed binary mask.
    It accepts an (H, W, D) hyperspectral cube or a (N, D) matrix, with
    optional leading batch dimensions.

    Both the input and output have the same shape. Missing entries are zero,
    matching the dense measurement format used by the sHSI reference code.
    """

    def __init__(
        self, mask: np.ndarray | torch.Tensor, device: DeviceLikeType | None = None
    ):
        """Initialize a masked-observation operator.

        Args:
            mask: Two- or three-dimensional array with binary entries.
            device: Device on which to keep the mask.
        """
        super().__init__(device=device)
        mask_tensor = torch.as_tensor(mask)

        if mask_tensor.ndim not in (2, 3):
            raise ValueError("mask must have shape (N, D) or (H, W, D)")
        if not torch.all((mask_tensor == 0) | (mask_tensor == 1)):
            raise ValueError("mask entries must be 0 or 1")

        mask_tensor = mask_tensor.to(dtype=torch.bool)
        if device is not None:
            mask_tensor = mask_tensor.to(device=device)

        self.register_buffer("_mask", mask_tensor.clone(), persistent=False)
        self._shape = tuple(int(dim) for dim in mask_tensor.shape)

    @property
    def mask(self) -> torch.Tensor:
        """Return the binary observation mask."""
        return self._mask

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Return the shape of an input sample."""
        return self._shape

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of an output sample."""
        return self._shape

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the operator, retaining the legacy `out` parameter."""
        del out
        return self.forward(x)

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.ndim < len(self._shape)
            or tuple(x.shape[-len(self._shape) :]) != self._shape
        ):
            raise ValueError(
                f"Expected trailing dimensions {self._shape}, got {tuple(x.shape)}"
            )
        return x * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the mask to an input tensor."""
        return self._apply_mask(x)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint; a real binary mask is self-adjoint."""
        return self._apply_mask(y)

    def gram(self, x: torch.Tensor) -> torch.Tensor:
        """Apply A* A; a binary mask is an orthogonal projection."""
        return self._apply_mask(x)
