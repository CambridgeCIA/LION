# This file is part of LION library
# License : GPL-3
#
# Author  : Zakhar Shumaylow, Charlie Shoebridge
# Modifications: Ander Biguri
# =============================================================================
from __future__ import annotations

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from LION.classical_algorithms.fdk import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.exceptions.exceptions import LIONSolverException
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
from LION.reconstructors.LearnedRegularizerReconstructor import (
    LearnedRegularizerReconstructor,
    estimate_alpha,
)


class ARParams(SolverParams):
    """Parameters of ARSolver.

    Args:
        lambd_gp: Weight of the gradient penalty in the WGAN training loss.
        n_steps, step_size, backtracking: Settings of the variational
            reconstruction used for validation and testing (see
            LearnedRegularizerReconstructor).
    """

    def __init__(
        self,
        lambd_gp: float = 1e-1,
        n_steps: int = 150,
        step_size: float = 1e-6,
        backtracking: float = 0.95,
    ):
        super().__init__()
        self.lambd_gp = lambd_gp
        self.n_steps = n_steps
        self.step_size = step_size
        self.backtracking = backtracking


class ARSolver(LIONsolver):
    """Trains an adversarial regulariser (Lunz et al., 2018).

    Training compares clean images with FDK reconstructions (WGAN loss with
    gradient penalty). Validation and testing reconstruct with the regulariser
    being trained, through LearnedRegularizerReconstructor; use that class
    directly to reconstruct with a trained regulariser.
    """

    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        geometry: Geometry = None,
        verbose: bool = True,
        device: torch.device = None,
        solver_params: ARParams | None = None,
    ) -> None:
        super().__init__(
            model, optimizer, self.wgan_loss, geometry, verbose, device, solver_params
        )
        self.reconstructor = LearnedRegularizerReconstructor(
            self.op,
            self.model,
            n_steps=self.solver_params.n_steps,
            step_size=self.solver_params.step_size,
            backtracking=self.solver_params.backtracking,
        )

    @staticmethod
    def default_parameters() -> ARParams:
        return ARParams()

    def set_validation(
        self,
        validation_loader,
        validation_freq,
        validation_fn=None,
        validation_fname=None,
    ):
        # The training loss (WGAN) is not a reconstruction metric: by default,
        # validate with the MSE between reconstruction and target.
        if validation_fn is None:
            validation_fn = torch.nn.MSELoss()
        super().set_validation(
            validation_loader, validation_freq, validation_fn, validation_fname
        )

    def set_testing(self, test_loader, testing_fn=None):
        if testing_fn is None:
            testing_fn = torch.nn.MSELoss()
        super().set_testing(test_loader, testing_fn)

    def _forward(self, data: torch.Tensor) -> torch.Tensor:
        # validation and testing reconstruct with the current regulariser
        return self.reconstructor.reconstruct(data)

    def _set_alpha(self, loader):
        self.reconstructor.alpha = estimate_alpha(self.op, loader, self.device)
        if self.verbose:
            print(f"Estimated alpha: {self.reconstructor.alpha}")

    def validate(self):
        if self.validation_loader is not None:
            self._set_alpha(self.validation_loader)
        return super().validate()

    def test(self):
        if self.test_loader is not None:
            self._set_alpha(self.test_loader)
        return super().test()

    def wgan_loss(self, sino_batch, target_batch):
        bad_recon = fdk(sino_batch, self.op)
        if self.do_normalise and self.model.get_input_type() == ModelInputType.IMAGE:
            bad_recon = self.model.normalise(bad_recon)
            target_batch = self.model.normalise(target_batch)

        epsilon = torch.Tensor(
            np.random.random((target_batch.shape[0], 1, 1, 1))
        ).type_as(target_batch)
        interpolates = (
            epsilon * target_batch + ((1 - epsilon) * bad_recon)
        ).requires_grad_(True)

        net_interpolates = self.model(interpolates)

        fake = (
            torch.Tensor(net_interpolates.shape)
            .fill_(1.0)
            .type_as(target_batch)
            .requires_grad_(False)
        )
        gradients = torch.autograd.grad(
            outputs=net_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.shape[0], -1)
        loss = (
            (self.model(target_batch)).mean()
            - (self.model(bad_recon)).mean()
            + self.solver_params.lambd_gp
            * (((gradients.norm(2, dim=1) - 1)) ** 2).mean()
        )
        return loss

    def mini_batch_step(self, sino_batch, target_batch) -> float:
        return self.wgan_loss(sino_batch, target_batch)

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Lunz, Sebastian, Ozan Öktem, and Carola-Bibiane Schönlieb")
            print('"Adversarial regularizers in inverse problems."')
            print("neural information processing systems 31 (2018).")
            print("arXiv:2008.02839 (2020).")
        elif cite_format == "bib":
            string = """
            @article{lunz2018adversarial,
            title={Adversarial regularizers in inverse problems},
            author={Lunz, Sebastian and {\"O}ktem, Ozan and Sch{\"o}nlieb, Carola-Bibiane},
            journal={Advances in neural information processing systems},
            volume={31},
            year={2018}
            }
            """
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )
