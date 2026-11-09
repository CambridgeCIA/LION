# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Alvaro Exposito Martinez, Ander Biguri
# =============================================================================
from typing import Optional
from LION.models.LIONmodel import LIONmodel
import torch
from torch.optim.optimizer import Optimizer
from LION.optimizers.LIONsolver import LIONsolver, warn_if_not_mse
from LION.utils.parameter import LIONParameter, OperatorParameters

from LION.utils.image_transforms import gaussian_blur
from LION.utils.citation import citation


class Noisier2InverseSolver(LIONsolver):
    """Noisier2Inverse.

    The training loss is the mean squared error: the target y - M estimates the
    clean projections only in the mean-squared sense (the minimiser of the
    expected MSE is the conditional mean). ``loss_fn`` is only used as the
    default validation and testing function.
    """

    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn,
        operator_params: OperatorParameters = None,
        solver_params: Optional[LIONParameter] = None,
        model_regularization=None,
        scheduler=None,
        scheduler_step: str = "epoch",
        verbose: int = 1,
        device: torch.device = None,
        seed: Optional[int] = 0,
        deterministic: bool = False,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            operator_params=operator_params,
            solver_params=solver_params,
            model_regularization=model_regularization,
            scheduler=scheduler,
            scheduler_step=scheduler_step,
            verbose=verbose,
            device=device,
            seed=seed,
            deterministic=deterministic,
        )
        warn_if_not_mse(loss_fn, "Noisier2Inverse")

        self.op_params = self.operator_params
        self.projector, _ = self.op.autograd()

    @staticmethod
    def default_parameters() -> LIONParameter:
        params = LIONParameter()
        params.sigma = 3
        params.delta = 1
        # How measurements are reconstructed, recon_fn(y, op). "default": the
        # operator's default reconstruction (FDK for CT), replaced by that
        # function when the solver is made.
        params.recon_fn = "default"
        return params

    def mini_batch_step(self, y, targets):
        sigma = self.solver_params.sigma
        delta = self.solver_params.delta
        ks = int(sigma * 3) * 2 + 1

        N = torch.randn_like(y) * delta
        N = gaussian_blur(N, ks, sigma)
        z = y + N

        input_recon = self.reconstruct_measurements(z, self.op)
        output_recon = self.model(input_recon)
        output_sino = self.projector(output_recon)
        target_sino = y - N

        # Sobolev Loss
        # res = output_sino - target_sino
        # grad_x = res[:, :, :, 1:] - res[:, :, :, :-1]
        # grad_y = res[:, :, 1:, :] - res[:, :, :-1, :]

        # batch_loss = ((output_sino - target_sino)**2).mean() + (grad_x**2).mean() + (grad_y**2).mean()
        batch_loss = ((output_sino - target_sino) ** 2).mean()
        return self._regularized(batch_loss, input_recon, output_recon)

    def _forward(self, data):
        # validation and testing evaluate the reconstruction
        return self.reconstruct(data)

    def reconstruct(self, y):
        input_recon = self.reconstruct_measurements(y, self.op)
        output_recon = self.model(input_recon)
        return output_recon

    @staticmethod
    def cite(cite_format="bib"):
        """Where Noisier2Inverse comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{gruber2025noisier2inverse,
            title={Noisier2Inverse: Self-Supervised Learning for Image Reconstruction with Correlated Noise},
            author={Gruber, Nadja and Schwab, Johannes and Haltmeier, Markus and Biguri, Ander and Dlaska, Clemens and Hwang, Gyeongha},
            journal={arXiv preprint arXiv:2503.19468},
            year={2025}
            }
            """,
            mla="""
            Gruber, Nadja, et al. "Noisier2Inverse: Self-Supervised Learning for Image Reconstruction with Correlated Noise." arXiv preprint arXiv:2503.19468 (2025).
            """,
        )
