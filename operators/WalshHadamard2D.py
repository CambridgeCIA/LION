"""Walsh-Hadamard Transform 2D Operator."""

from __future__ import annotations

import torch
from spyrit.core.torch import fwht

from LION.operators.Operator import Operator


class WalshHadamard2D(Operator):
    """Walsh-Hadamard Transform operator.

    Parameters
    ----------
    height : int
        Height of the 2D input vector. Must be a power of two.
    width : int
        Width of the 2D input vector. Must be a power of two.
    device : str or torch.device
        Device where tensors are placed.
    """

    def __init__(
        self, height: int, width: int, device: str | torch.device | None = None
    ):
        """Walsh-Hadamard Transform operator.

        Parameters
        ----------
        height : int
            Height of the 2D input vector. Must be a power of two.
        width : int
            Width of the 2D input vector. Must be a power of two.
        device : str or torch.device
            Device where tensors are placed.
        """
        super().__init__(device=device)
        if (height & (height - 1)) != 0:
            raise ValueError("height must be a power of two.")
        if (width & (width - 1)) != 0:
            raise ValueError("width must be a power of two.")
        self.height = height
        self.width = width
        self.N = height * width

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the Walsh-Hadamard Transform.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of shape (height, width).
        out : None
            Legacy for tomosipo ``to_autograd``. Just ignore.

        Returns
        -------
        torch.Tensor
            Transformed vector of shape (N,).
        """
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Walsh-Hadamard Transform.

        .. note::
            Prefer calling the instance of the WalshHadamard2D operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if len(x.shape) != 2 or x.shape != (self.height, self.width):
            raise ValueError(
                f"Expected input of shape ({self.height}, {self.width}), got {x.shape}"
            )
        return fwht(x.to(self.device).ravel(), order=False)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint of the Walsh-Hadamard Transform.

        Parameters
        ----------
        y : torch.Tensor
            Input vector of shape (N,).

        Returns
        -------
        torch.Tensor
            Adjoint transformed vector of shape (height, width).
        """
        if len(y.shape) != 1 or y.shape[0] != self.N:
            raise ValueError(f"Expected input of shape ({self.N},), got {y.shape}")

        return fwht(y.to(self.device), order=False).view(self.height, self.width)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the pseudo-inverse of the Walsh-Hadamard Transform.

        Parameters
        ----------
        y : torch.Tensor
            Input vector of shape (N,).

        Returns
        -------
        torch.Tensor
            Pseudo-inverse transformed vector of shape (N,).
        """
        return self.adjoint(y) / self.N

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Return the shape of the input domain."""
        return (self.height, self.width)

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of the output range."""
        return (self.N,)
