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
    """The base class of LION's operators: a linear map, as a torch module,
    with its adjoint and the shapes it maps between.

    LION uses the general words, as an operator can be anything:

    - **domain**: what the operator is applied to, usually the image ``x``;
    - **range**: what it gives, the measurements ``y``.

    In CT, the operator is the projection: its domain is the image (the
    geometry's ``image_shape``) and its range the sinogram. Some operators'
    domains are not images, e.g. a wavelet synthesis's is its coefficients.

    A subclass defines ``forward``, ``__call__`` (``return self.forward(x)``),
    ``adjoint``, ``domain_shape`` and ``range_shape``; ``autograd()`` if it can
    be differentiated through; ``inverse`` if it has one;
    ``default_reconstruction_fn`` if it has a usual reconstruction. ``device``:
    where it computes (None: unset; ``to_device`` moves it)."""

    def __init__(self, device: DeviceLikeType | None = None):
        super().__init__()
        self.device = device

    @abstractmethod
    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """The operator applied to ``x`` (``A x``): ``forward``.

        Abstract so that a subclass defines it with this signature, ``(x,
        out=None)``: tomosipo's ``to_autograd`` calls it with ``out``, where to
        write the result; other operators ignore it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The operator applied to ``x`` (``A x``), defined by a subclass. Call
        the operator, ``op(x)``, rather than this method (as for torch
        modules)."""
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """The adjoint applied to ``y`` (``Aᵀ y``): from the range (the
        measurements) to the domain (the images)."""
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
        """The Gram operator, AᵀA, applied to ``x``."""
        return self.adjoint(self.forward(x))

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """The inverse applied to ``y``, for operators that have one (e.g. an
        orthogonal transform): a subclass defines it only then, and its default
        reconstruction is then the inverse."""
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
        """Moves the operator and its sub-operators to ``device``, updating
        their ``device`` attributes (which ``nn.Module.to`` doesn't). Returns
        the operator."""
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
