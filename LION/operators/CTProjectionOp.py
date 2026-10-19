"""Thin wrapper around tomosipo's Operator class for linear tomographic projection."""

from __future__ import annotations

import torch
from tomosipo.Operator import Operator as TomosipoOperator

from LION.operators.Operator import Operator
from LION.utils.common_primitives import DeviceLikeType


class CTProjectionOp(Operator):
    """Tomographic projection operator and its adjoint using tomosipo.

    Args:
        ts_operator: Tomosipo operator implementing the tomographic projection and its adjoint.
        device: Device for computations. If None, computations are done on the default device.
    """

    def __init__(
        self, ts_operator: TomosipoOperator, device: DeviceLikeType | None = None
    ):
        """Initialize the CTProjectionOp."""
        super().__init__(device=device)
        self._ts = ts_operator

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the forward projection.

        Args:
            x: The input volume dataset to which the forward projection is applied.
            out: Optional output holder to store the output projections. Needed for ``to_autograd`` functionality.

        Returns:
            The projection dataset on which the volume has been forward projected.
        """
        return self._ts._fp(x, out=out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward projection.

        .. note::
            Prefer calling the instance of the operator as ``operator(x)`` over directly calling this method.
            See PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return self._ts._fp(x)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the backprojection.

        Args:
            y: The input projection dataset to which the backprojection is applied.

        Returns:
            The volume dataset on which the projection dataset has been backprojected.
        """
        return self._ts._bp(y)

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying tomosipo operator."""
        return getattr(self._ts, name)

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Return the shape of the volume domain."""
        return self._ts.domain_shape

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of the projection range."""
        return self._ts.range_shape
