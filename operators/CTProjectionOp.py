"""Thin wrapper around tomosipo's Operator class for linear tomographic projection."""

from __future__ import annotations

import tomosipo as ts
import torch

from LION.operators.Operator import Operator


class CTProjectionOp(Operator):
    """Tomographic projection operator and its adjoint using tomosipo.

    Parameters
    ----------
    ts_operator : tomosipo.Operator.Operator
        Tomosipo operator implementing the tomographic projection and its
        adjoint.
    device : torch.device | str | None, optional
        Device for computations. If None, computations are done on the
        default device.
    """

    def __init__(
        self,
        ts_operator: ts.Operator.Operator,
        device: torch.device | str | None = None,
    ):
        """Initialize the CTProjectionOp."""
        super().__init__(device=device)
        self._ts = ts_operator

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the forward projection.

        Parameters
        ----------
        x : torch.Tensor
            The input volume dataset to which the forward projection is applied.
        out : torch.Tensor | None, optional
            Optional output holder to store the output projections.
            It is needed for the ``to_autograd`` functionality.

        Returns
        -------
        torch.Tensor
            The projection dataset on which the volume has been forward
            projected.
        """
        return self._ts._fp(x, out=out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward of CTProjectionOp.

        .. note::
            Prefer calling the instance of the CTProjectionOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return self._ts._fp(x)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the backprojection.

        Parameters
        ----------
        y : torch.Tensor
            The input projection dataset to which the backprojection is applied.

        Returns
        -------
        torch.Tensor
            The volume dataset on which the projection dataset has been
            backprojected.
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
