"""Class for Symmetrized Gradient Operator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.RearrangeOp import RearrangeOp
from zsu.operators.FiniteDifferenceOp import FiniteDifferenceOp, PadModeIn


class SymmetrizedGradientOp(LinearOperator):
    r"""Discrete symmetrized gradient operator using finite differences.

    Based on finite differences along the axes listed in ``dim``
    (e.g. ``dim=(-2, -1)`` for the last two axes),
    this pointwise operator computes the symmetrized gradient of a discrete vector field,
    i.e. a :math:`(d+1)`-dimensional tensor ``v`` (``v.shape[0] == len(dim)``),
    where each ``v[j]`` is a :math:`d`-dimensional vector component.
    The output is a :math:`(d+2)`-dimensional tensor ``w``
    (``w.shape[0] == w.shape[1] == len(dim)``) where each ``w[i, j]``
    contains the symmetric part of the discrete gradient of ``v``,
    computed along the axes listed in ``dim``.
    Note that ``dim`` must not contain the :math:`0^{\text{th}}` axis.

    The symmetrized gradient ``E(v)`` using the finite difference operator ``nabla`` can be written as

    .. code-block:: python

        w = E(v) = 0.5 * (nabla(v) + nabla(v).transpose(0, 1))

    or more explicitly as

    .. code-block:: python

        w[i, j] = E(v)[i, j] = 0.5 * (nabla(v)[i, j] + nabla(v)[j, i])

    for every ``i``, ``j`` in ``[0, ..., len(dim) - 1]``.

    A common use case of the symmetrized gradient is Total Generalized Variation (TGV) regularization,
    with the 2D case (i.e. ``len(dim) == 2``) shown in [TGV]_.

    References
    ----------
    .. [TGV] Bredies, K. Recovering piecewise smooth multichannel images by minimization of convex
       functionals with total generalized variation penalty. In: Bruhn, A., Pock, T., Tai, X.C. (eds)
       Efficient Algorithms for Global Optimization Methods in Computer Vision. Lecture Notes in Computer Science,
       vol. 8293, Springer, Berlin, Heidelberg, 2014, pp. 44-77. https://doi.org/10.1007/978-3-642-54774-4_3
    """

    def __init__(
        self,
        dim: Sequence[int],
        transpose_dim: int = 1,
        mode: Literal["central", "forward", "backward"] = "backward",
        pad_mode: PadModeIn | Sequence[PadModeIn] = "zeros",
    ) -> None:
        r"""Symmetrized gradient operator.

        Parameters
        ----------
        dim
            Dimensions along which finite differences are calculated.
            It must not contain the :math:`0^{\text{th}}` axis.
        mode
            Type of finite difference operator
        pad_mode
            Padding to ensure output has the same size as the input
        """
        super().__init__()
        finite_difference_op = FiniteDifferenceOp(dim, mode=mode, pad_mode=pad_mode)

        assert (
            transpose_dim >= 1
        ), f"transpose_dim must be at least 1, got {transpose_dim}."
        k = transpose_dim
        middle = "  ".join(
            f"d{i}" for i in range(1, k)
        )  # 'd1  d2  ...  d{k-1}'  or  '' if k == 1
        before = "d0  " + middle + f"  d{k}"  # d0  d1  ...  d{k-1}  d{k}
        after = f"d{k}  " + middle + "  d0"  # d{k}  d1  ...  d{k-1}  d0
        transpose_op = RearrangeOp(f"{before}  ...   ->   {after}  ...")
        self._operator = 0.5 * (1 + transpose_op) @ finite_difference_op

    def __call__(
        self, v: torch.Tensor
    ) -> tuple[torch.Tensor,]:
        r"""Apply forward of symmetrized gradient operator.

        The length of the first axis of ``v`` (``v.shape[0]``) must match
        the number of dimensions specified in ``dim`` during initialization.

        Parameters
        ----------
        v
            :math:`(d+1)`-dimensional input tensor with
            the first dimension indexing the :math:`d` vector components.

        Returns
        -------
            A single-element tuple (``w``, ) containing a :math:`(d+2)`-dimensional tensor ``w``
            which represents the symmetrized gradient of ``v``.
        """
        return super().__call__(v)

    def forward(
        self, v: torch.Tensor
    ) -> tuple[torch.Tensor,]:
        """Apply forward of SymmetrizedGradientOp.

        .. note::
            Prefer calling the instance of the SymmetrizedGradientOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return self._operator(v)

    def adjoint(
        self, w: torch.Tensor
    ) -> tuple[torch.Tensor,]:  # type: ignore[explicit-override]
        r"""Apply adjoint of symmetrized gradient operator.

        The lengths of the first two axes of ``w`` (``w.shape[0]`` and ``w.shape[1]``) must equal each other and
        match the number of dimensions specified in ``dim`` during initialization.

        Parameters
        ----------
        w
            :math:`(d+2)`-dimensional input tensor representing the symmetrized gradient.

        Returns
        -------
            A single-element tuple (``v``, ) containing the :math:`(d+1)`-dimensional tensor ``v``
            which represents the adjoint of the symmetrized gradient.
        """
        return self._operator.adjoint(w)
