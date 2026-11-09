# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Thanh Trung (Troy) Vu, Ander Biguri
# =============================================================================
"""Base class for operators in LION."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from LION.utils.common_primitives import DeviceLikeType
from LION.utils.mrpro_import import import_mrpro


def inverse_reconstruction(y: torch.Tensor, op: "Operator") -> torch.Tensor:
    """The operator's own inverse (or pseudo-inverse) of ``y``: the default
    reconstruction of operators that have one (e.g. the Walsh-Hadamard
    transform)."""
    return op.inverse(y)


class Operator(torch.nn.Module, ABC):
    """Base class for operators in LION.

    Operators represent forward and backward projections, mapping one tensor to another tensor.

    This class defines a common interface for all operators in LION. The most important API includes
    ``forward``, ``adjoint``, and ``domain_shape`` / ``range_shape``.

    LION uses the general words, as an operator can be anything:

    - **domain**: what the operator is applied to, usually the image ``x``;
    - **range**: what it gives, the measurements ``y``.

    In CT, the operator is the projection: its domain is the image (the
    geometry's ``image_shape``) and its range the sinogram. Some operators'
    domains are not images, e.g. a wavelet synthesis's is its coefficients.

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

    def autograd(self):
        """The operator and its adjoint as torch autograd functions of batches,
        ``(forward, adjoint)``: functions whose gradients torch knows (the
        gradient of A is A^T, and of A^T is A), for models and solvers whose
        gradients go through the operator (e.g. LPD, Sparse2Inverse).

        An operator that can be differentiated defines it; the base class
        raises, so one that can't isn't used there by mistake.

        - An operator torch can't differentiate (e.g. CT's, a tomosipo/astra
          call) wraps itself, as ``CTProjectionOp`` does with tomosipo's
          ``to_autograd``.
        - An operator made only of torch operations (no numpy, no external
          library) is differentiated by torch as it is. If you know yours is,
          say so::

              def autograd(self):
                  return self, self.adjoint
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no autograd(): it can't be used where "
            "gradients go through the operator. If torch differentiates it as it "
            "is (only torch operations), define autograd() returning "
            "(self, self.adjoint); otherwise return its forward and adjoint as "
            "torch autograd functions (as tomosipo's to_autograd does for CT)"
        )

    def default_reconstruction_fn(self):
        """The operator's usual reconstruction, as a function ``f(y, op)`` of
        measurements and operator that a record can name (e.g. ``fdk`` for
        CT). It is what ``"default"`` stands for in a model's or reconstructor's
        ``initial_reconstruction`` and a solver's ``recon_fn``, and they record
        it in its place; it is also the baseline methods are compared with
        (``LION.evaluation.baseline``). An operator with an inverse
        (``inverse()``) has that, ``inverse_reconstruction``; others that have
        one define it."""
        if type(self).inverse is not Operator.inverse:
            return inverse_reconstruction
        raise NotImplementedError(
            f"{type(self).__name__} has no default reconstruction: set the "
            "model's initial reconstruction (model.set_initial_reconstruction) "
            "or the solver's recon_fn"
        )

    def default_reconstruction(self, y: torch.Tensor) -> torch.Tensor:
        """The operator's usual reconstruction of measurements ``y``
        (``default_reconstruction_fn``, e.g. FDK for CT)."""
        return self.default_reconstruction_fn()(y, self)

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
        """The shape of the operator's domain: what it is applied to, usually
        the image (in CT, the geometry's ``image_shape``)."""
        raise NotImplementedError(
            "property `domain_shape` must be implemented by subclasses."
        )

    @property
    def range_shape(self) -> tuple[int, ...]:
        """The shape of the operator's range: the measurements it gives (in
        CT, the sinogram)."""
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
