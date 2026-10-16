from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from mrpro.operators import (
    LinearOperator,
    LinearOperatorMatrix,
    ProximableFunctional,
    ProximableFunctionalSeparableSum,
    RearrangeOp,
)
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional

from LION.models.BasePdhgNet import BasePdhgNet
from LION.operators.FiniteDifferenceOp import FiniteDifferenceOp, PadModeIn


class AdaptiveTVNetwork(BasePdhgNet, ABC):
    r"""Unrolled primal dual hybrid gradient with spatially adaptive regularization parameter maps for TV.

    Solves the minimization problem

        :math:`\min_x \frac{1}{2}\| Ax - y\|_2^2 + \| \Lambda_{\theta} \nabla x\|_1`,
    where :math:`A` is the forward linear operator, :math:`\nabla` is the gradient operator,
    and :math:`\Lambda_{\theta}` is a strictly positive regularization parameter map that is estimated from
    an input image with a network :math:`u_{\theta}` with trainable parameters :math:`\theta`.
    """

    def init_ops(self, fd_pad_mode: PadModeIn | Sequence[PadModeIn]) -> None:
        finite_differences_operator = FiniteDifferenceOp(
            dim=self.dim, mode="forward", pad_mode=fd_pad_mode
        )
        self.zero_functional = ZeroFunctional()
        self.gradient_operator = (
            RearrangeOp("grad batch ... -> batch grad ... ")
            @ finite_differences_operator
        )

    def split_lambda(
        self, lambda_reg: torch.Tensor | float
    ) -> tuple[torch.Tensor | float, None]:
        return lambda_reg, None

    @property
    def g(self) -> ProximableFunctionalSeparableSum | ProximableFunctional:
        return self.zero_functional

    def make_f(
        self,
        measurement: torch.Tensor,
        *,
        lambda_reg: torch.Tensor | float,
        l2_weight: torch.Tensor | None = None,
    ) -> ProximableFunctionalSeparableSum:
        if l2_weight is None:
            l2_weight = torch.ones(
                (), device=measurement.device, dtype=measurement.real.dtype
            )
        l2_norm_squared = 0.5 * L2NormSquared(target=measurement, weight=l2_weight)

        lambda_map = self.prepare_lambda(lambda_reg)
        l1_norm = L1NormViewAsReal(weight=lambda_map)

        return ProximableFunctionalSeparableSum(l2_norm_squared, l1_norm)

    def make_stacked_operator(self, forward_op: LinearOperator) -> LinearOperatorMatrix:
        return LinearOperatorMatrix(((forward_op,), (self.gradient_operator,)))

    def make_initial_values(
        self, initial_image: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return (initial_image,)

    @abstractmethod
    def prepare_lambda(self, lambda_reg: torch.Tensor | float) -> torch.Tensor | float:
        ...
