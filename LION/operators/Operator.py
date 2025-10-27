"""Base class for operators in LION."""

import torch


class Operator:
    """
    Base class for operators in LION.

    Operators represent mathematical operations that can be applied to data,
    such as forward and backward projections in imaging.

    This class should be subclassed to implement specific operators.
    """

    @property
    def domain_shape(self) -> torch.Size:
        """
        Get the shape of the image domain.

        Returns
        -------
        torch.Size
            Shape of the image domain.
        """
        raise NotImplementedError("property must be implemented by subclasses.")

    @property
    def range_shape(self) -> torch.Size:
        """
        Get the shape of the data (measurement) domain.

        Returns
        -------
        torch.Size
            Shape of the data (measurement) domain.
        """
        raise NotImplementedError("property must be implemented by subclasses.")

    def forward(self, x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        """
        Apply the forward operation of the operator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to which the operator is applied.
        out : torch.Tensor, optional
            Optional output tensor to store the result.

        Returns
        -------
        torch.Tensor
            Result of applying the forward operation.
        """
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def adjoint(self, y: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        """
        Apply the adjoint (backward) operation of the operator.

        Parameters
        ----------
        y : torch.Tensor
            Input tensor to which the adjoint operator is applied.
        out : torch.Tensor, optional
            Optional output tensor to store the result.

        Returns
        -------
        torch.Tensor
            Result of applying the adjoint operation.
        """
        raise NotImplementedError("Adjoint method must be implemented by subclasses.")

    def __call__(self, x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        return self.forward(x, out=out)
