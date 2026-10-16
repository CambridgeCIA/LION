from __future__ import annotations

import torch
from mrpro.operators import LinearOperator

from LION.classical_algorithms.pdhg_direct import tgv_pdhg
from LION.models.AdaptiveTGVNetwork import AdaptiveTGVNetwork
from LION.operators.Operator import Operator
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp


class AdaptiveTGVPCMNetwork(AdaptiveTGVNetwork):
    """2-D real-valued TGV specialization for PCM reconstruction."""

    def split_lambda(
        self, lambda_reg: torch.Tensor | tuple[float, float]
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        if isinstance(lambda_reg, tuple):
            return lambda_reg[0], lambda_reg[1]
        if lambda_reg.ndim != 4 or lambda_reg.shape[1] != 2:
            raise ValueError(
                f"TGV lambda maps must have shape (batch, 2, height, width), got {tuple(lambda_reg.shape)}"
            )

        grad_term_weight = lambda_reg[:, 0, ...][:, None, None, ...]
        sym_grad_term_weight = lambda_reg[:, 1, ...][:, None, None, None, ...]
        return grad_term_weight, sym_grad_term_weight

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def postprocess(self, output: torch.Tensor) -> torch.Tensor:
        return output

    def forward(
        self,
        *,
        input_image: torch.Tensor | None,
        initial_image: torch.Tensor,
        measurement: torch.Tensor,
        key: int,
        forward_op: LinearOperator | Operator,
        lambda_reg: torch.Tensor | tuple[float, float] | None = None,
        l2_weight: torch.Tensor | None = None,
        max_iterations: int | None = None,
        primal_stepsize: float | None = None,
        dual_stepsize: float | None = None,
        use_checkpoint: bool = False,
        checkpoint_chunk_size: int | None = None,
        checkpoint_use_reentrant: bool = False,
        use_tqdm: bool = False,
    ) -> torch.Tensor:
        """Run the selected PCM PDHG backend."""
        if self.backend in ("new", "mrpro"):
            assert isinstance(forward_op, LinearOperator)
            return super().forward(
                input_image=input_image,
                initial_image=initial_image,
                measurement=measurement,
                key=key,
                forward_op=forward_op,
                lambda_reg=lambda_reg,
                l2_weight=l2_weight,
                max_iterations=max_iterations,
                primal_stepsize=primal_stepsize,
                dual_stepsize=dual_stepsize,
                use_checkpoint=use_checkpoint,
                checkpoint_chunk_size=checkpoint_chunk_size,
                checkpoint_use_reentrant=checkpoint_use_reentrant,
                use_tqdm=use_tqdm,
            )

        if self.backend != "direct":
            raise ValueError(f"Unsupported PCM SAPM backend: {self.backend}")

        if lambda_reg is None:
            assert input_image is not None
            lambda_reg = self.estimate_lambda_map(input_image, key)

        assert l2_weight is None
        pcm_op: PhotocurrentMapOp = (
            forward_op if isinstance(forward_op, PhotocurrentMapOp) else forward_op.op
        )

        return tgv_pdhg(
            measurement=measurement,
            initial_image=initial_image,
            forward_op=pcm_op,
            lambda_reg=lambda_reg,
            max_iterations=max_iterations
            if max_iterations is not None
            else self.n_iterations,
            primal_stepsize=primal_stepsize
            if primal_stepsize is not None
            else self.primal_stepsize,
            dual_stepsize=dual_stepsize
            if dual_stepsize is not None
            else self.dual_stepsize,
            theta=self.direct_theta,
            use_checkpoint=use_checkpoint,
            checkpoint_chunk_size=checkpoint_chunk_size,
            checkpoint_use_reentrant=checkpoint_use_reentrant,
            use_tqdm=use_tqdm,
        )
