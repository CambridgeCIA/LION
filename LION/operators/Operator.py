"""Base class for operators in LION."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from LION.utils.common_primitives import DeviceLikeType
from LION.utils.mrpro_import import import_mrpro


class Operator(torch.nn.Module, ABC):
    """Base class for operators in LION.

    Operators represent forward and backward projections, mapping one tensor to another tensor.

    This class defines a common interface for all operators in LION. The most important API includes
    ``forward``, ``adjoint``, and ``domain_shape`` / ``range_shape``.

    This class should be subclassed to implement specific operators.

    Attributes:
        device : Device for computations. If None, computations are done on the default device.
    """

    def __init__(self, device: DeviceLikeType | None = None):
        """Initialize the Operator."""
        super().__init__()
        self.device = device

    @abstractmethod
    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the forward operation of the operator.

        .. note::
            Usually this is just a wrapper for the ``forward`` method: ``return self.forward(x)``.
            Users should write this explicitly in their subclasses so that they can add the docstring
            in the ``__call__`` method instead of the ``forward`` method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.

        Args:
            x: Input to which the operator is applied.
            out: Optional output tensor to store the result. Most subclasses may ignore this parameter,
                except for tomosipo-based operators which need it for the ``to_autograd`` functionality.

        Returns:
            Result of applying the forward operation.
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward operation of Operator.

        .. note::
            Prefer calling the instance of the Operator operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint (backward) operation of the operator.

        Args:
            y: Input to which the adjoint operator is applied.

        Returns:
            Result of applying the adjoint operation.
        """
        raise NotImplementedError("Adjoint method must be implemented by subclasses.")

    def gram(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Gram operator.

        For a LinearOperator :math:`A`, the self-adjoint Gram operator is defined as :math:`A^H A`.

        .. note::
           This is the inherited default implementation.
        """
        return self.adjoint(self.forward(x))

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Computes the inverse of the operator applied to input x if exists.

        This method should only be implemented by subclasses that have a well-defined inverse.

        Args:
            y: The input data to which the inverse operator is applied.

        Returns:
            The result of applying the inverse operator to x.
        """
        raise NotImplementedError(
            "The inverse method is not implemented, possibly because "
            "the operator does not have a well-defined inverse."
        )

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Get the shape of the image domain.

        Returns:
            Shape of the image domain.
        """
        raise NotImplementedError(
            "property `domain_shape` must be implemented by subclasses."
        )

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Get the shape of the data (measurement) domain.

        Returns:
            Shape of the data (measurement) domain.
        """
        raise NotImplementedError(
            "property `range_shape` must be implemented by subclasses."
        )

    def to_device(self, device: DeviceLikeType) -> Operator:
        """Move the operator and its suboperators to the specified device and update their ``device`` attributes.

        Note: Calling ``nn.Module.to(device)`` does not automatically update the ``device`` attribute.
        This method ensures that the ``device`` attribute is updated for all suboperators as well.

        Args:
            device: Target device.

        Returns:
            The operator moved to the specified device.
        """
        super().to(device=device)

        for module in self.modules():  # Note: ``nn.Module.modules()`` is recursive
            if isinstance(module, Operator):
                module.device = device

        return self


_MrproLinearOp = None


def _mrpro_linear_op():
    """MrproLinearOp, defined on first use: it subclasses mrpro's
    LinearOperator, and mrpro is only imported when needed (like tomosipo)."""
    global _MrproLinearOp
    if _MrproLinearOp is None:
        LinearOperator = import_mrpro(
            "mrpro.operators.LinearOperator", "MrproLinearOp"
        ).LinearOperator

        class MrproLinearOp(LinearOperator):
            """Thin mrpro LinearOperator wrapper around any LION operator.

            The wrapped operator must be linear, differentiable with PyTorch
            autograd, take ``torch.Tensor``, return a ``torch.Tensor``,
            implement ``adjoint()``.
            """

            def __init__(self, op: Operator):
                super().__init__()
                self.op = op

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
                return (self.op(x),)

            def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
                return (self.op.adjoint(x),)

        MrproLinearOp.__module__ = __name__
        MrproLinearOp.__qualname__ = "MrproLinearOp"
        _MrproLinearOp = MrproLinearOp
    return _MrproLinearOp


def __getattr__(name):
    # `from LION.operators.Operator import MrproLinearOp` imports mrpro then
    if name == "MrproLinearOp":
        return _mrpro_linear_op()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
