import torch


class Operator:
    """
    Base class for operators in LION.

    Operators represent mathematical operations that can be applied to data,
    such as forward and backward projections in imaging.

    This class should be subclassed to implement specific operators.
    """

    @property
    def image_shape(self) -> torch.Size:
        """
        Get the shape of the image domain.

        Returns
        -------
        torch.Size
            Shape of the image domain.
        """
        raise NotImplementedError("image_shape property must be implemented by subclasses.")

    @property
    def data_shape(self) -> torch.Size:
        """
        Get the shape of the data domain.

        Returns
        -------
        torch.Size
            Shape of the data domain.
        """
        raise NotImplementedError("data_shape property must be implemented by subclasses.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward operation of the operator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to which the operator is applied.

        Returns
        -------
        torch.Tensor
            Result of applying the forward operation.
        """
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the adjoint (backward) operation of the operator.

        Parameters
        ----------
        y : torch.Tensor
            Input tensor to which the adjoint operator is applied.

        Returns
        -------
        torch.Tensor
            Result of applying the adjoint operation.
        """
        raise NotImplementedError("Adjoint method must be implemented by subclasses.")

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        Alias for the adjoint operation.

        Parameters
        ----------
        y : torch.Tensor
            Input tensor to which the transpose operator is applied.

        Returns
        -------
        torch.Tensor
            Result of applying the transpose operation.
        """
        return self.adjoint(y)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def T(self, y: torch.Tensor) -> torch.Tensor:
        return self.adjoint(y)

    def A(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def AT(self, y: torch.Tensor) -> torch.Tensor:
        return self.adjoint(y)
