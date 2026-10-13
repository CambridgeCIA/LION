from __future__ import annotations

from abc import ABC
from collections.abc import Sequence

import torch
from mrpro.operators import (
    IdentityOp,
    LinearOperator,
    LinearOperatorMatrix,
    ProximableFunctional,
    ProximableFunctionalSeparableSum,
    RearrangeOp,
    ZeroOp,
)
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional

from LION.models.BasePdhgNet import BasePdhgNet
from LION.operators.FiniteDifferenceOp import FiniteDifferenceOp, PadModeIn
from LION.operators.SymmetrizedGradientOp import SymmetrizedGradientOp


class AdaptiveTGVNetwork(BasePdhgNet, ABC):
    def init_ops(self, fd_pad_mode: PadModeIn | Sequence[PadModeIn]) -> None:
        self.zero_op = ZeroOp()
        self._g = ProximableFunctionalSeparableSum(*(ZeroFunctional(),) * 2)

        # # Define the operator matrix K
        # # 1. First row [A, 0] (corresponding to the L2-norm data term): Ax + 0
        # data_term_row = (acquisition_operator, mrpro.operators.ZeroOp())

        # 2. Second row [\nabla, -I] (corresponding to the first L1-norm regularization term): \nabla x - v
        flip_grad_batch_op = RearrangeOp("grad  batch  ...  ->  batch  grad  ...")
        gradient_op = flip_grad_batch_op @ FiniteDifferenceOp(
            dim=self.dim, mode="forward", pad_mode=fd_pad_mode
        )
        # NOTE: Squeeze to reduce the number of dimensions of the image by one before applying finite difference
        # so that output shape matches the auxiliary tensor v.
        # squeeze_op = RearrangeOp('batch  1  ...  ->  batch  ...')
        # self.grad_term_row = (gradient_op @ squeeze_op, -1 * IdentityOp())
        self.grad_term_row = (gradient_op, -1 * IdentityOp())

        # 3. Third row [0, \mathcal{E}] (corresponding to the second L1-norm regularization term): 0 + \mathcal{E} v
        sym_grad_op = SymmetrizedGradientOp(
            dim=self.dim, transpose_dim=2, mode="backward", pad_mode=fd_pad_mode
        )
        self.sym_grad_term_row = (ZeroOp(), flip_grad_batch_op @ sym_grad_op)

    @property
    def g(self) -> ProximableFunctionalSeparableSum | ProximableFunctional:
        return self._g

    def make_f(
        self,
        measurement: torch.Tensor,
        *,
        lambda_reg: torch.Tensor | tuple[float, float],
        l2_weight: torch.Tensor | None = None,
    ) -> ProximableFunctionalSeparableSum:
        if l2_weight is None:
            l2_weight = torch.ones(
                (), device=measurement.device, dtype=measurement.real.dtype
            )
        data_term = 0.5 * L2NormSquared(target=measurement, weight=l2_weight)

        grad_term_weight, sym_grad_term_weight = self.split_lambda(lambda_reg)
        grad_term = L1NormViewAsReal(weight=grad_term_weight)  # lambda1
        assert sym_grad_term_weight is not None
        sym_grad_term = L1NormViewAsReal(weight=sym_grad_term_weight)  # lambda0
        minimization_sum = ProximableFunctionalSeparableSum(
            data_term, grad_term, sym_grad_term
        )
        return minimization_sum

    def make_initial_values(
        self, initial_image: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        batch_size = initial_image.shape[0]
        auxiliary_v_tensor = initial_image.new_zeros(
            (batch_size, len(self.dim), *initial_image.shape[1:])
        )
        return (initial_image, auxiliary_v_tensor)

    def make_stacked_operator(self, forward_op: LinearOperator) -> LinearOperatorMatrix:
        return LinearOperatorMatrix(
            ((forward_op, self.zero_op), self.grad_term_row, self.sym_grad_term_row)
        )
