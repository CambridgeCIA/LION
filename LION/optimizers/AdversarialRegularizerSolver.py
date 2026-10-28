# This file is part of LION library
# License : GPL-3
#
# Author  : Zakhar Shumaylow, Charlie Shoebridge
# Modifications: Ander Biguri
# =============================================================================
from __future__ import annotations

import numpy as np
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from LION.classical_algorithms.fdk import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.exceptions.exceptions import LIONSolverException
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
from LION.utils.citation import citation


class ARParams(SolverParams):
    """Parameters of ARSolver.

    Args:
        lambd_gp: Weight of the gradient penalty in the WGAN training loss.
    """

    def __init__(self, lambd_gp: float = 1e-1):
        super().__init__()
        self.lambd_gp = lambd_gp


class ARSolver(LIONsolver):
    """Trains an adversarial regulariser (Lunz et al., 2018).

    Training compares clean images with FDK reconstructions (WGAN loss with
    gradient penalty). The WGAN loss doesn't measure reconstructions, so
    validation and testing reconstruct with a ``LearnedRegularizerReconstructor``
    you make and give (``set_validation(..., reconstructor=...)``): the
    solver uses it with the regulariser being trained. Its settings (e.g.
    ``alpha``, which ``estimate_alpha`` estimates as in the paper) are the
    reconstructor's. To reconstruct with a trained regulariser, use the same
    class with the model.
    """

    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        geometry: Geometry = None,
        verbose: bool = True,
        device: torch.device = None,
        solver_params: ARParams | None = None,
        seed: Optional[int] = 0,
        deterministic: bool = False,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            self.wgan_loss,
            geometry,
            verbose,
            device,
            solver_params,
            seed=seed,
            deterministic=deterministic,
        )

    # Its models reconstruct only through a reconstructor (LION.evaluation
    # refuses their files alone).
    reconstructed_with = "LearnedRegularizerReconstructor"

    @staticmethod
    def default_parameters() -> ARParams:
        return ARParams()

    def set_validation(
        self,
        validation_loader,
        validation_freq,
        validation_fn=None,
        validation_fname=None,
        reconstructor=None,
    ):
        """As ``LIONsolver.set_validation``; ``reconstructor`` (a
        ``LearnedRegularizerReconstructor``) is needed. By default, validate
        with the MSE between reconstruction and target."""
        _needs(reconstructor, "set_validation")
        if validation_fn is None:
            validation_fn = torch.nn.MSELoss()
        super().set_validation(
            validation_loader,
            validation_freq,
            validation_fn,
            validation_fname,
            reconstructor,
        )

    def set_testing(self, test_loader, testing_fn=None, reconstructor=None):
        """As ``LIONsolver.set_testing``; ``reconstructor`` is needed."""
        _needs(reconstructor, "set_testing")
        if testing_fn is None:
            testing_fn = torch.nn.MSELoss()
        super().set_testing(test_loader, testing_fn, reconstructor)

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
    def cite(cite_format="bib"):
        """Where ARSolver comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{lunz2018adversarial,
            title={Adversarial regularizers in inverse problems},
            author={Lunz, Sebastian and {\\"O}ktem, Ozan and Sch{\\"o}nlieb, Carola-Bibiane},
            journal={Advances in neural information processing systems},
            volume={31},
            year={2018}
            }
            """,
            mla="""
            Lunz, Sebastian, Ozan Öktem, and Carola-Bibiane Schönlieb
            "Adversarial regularizers in inverse problems."
            neural information processing systems 31 (2018).
            arXiv:2008.02839 (2020).
            """,
        )


def _needs(reconstructor, where):
    if reconstructor is None:
        raise LIONSolverException(
            f"ARSolver.{where} needs a reconstructor: the regulariser alone "
            "doesn't reconstruct. Give e.g. "
            "reconstructor=LearnedRegularizerReconstructor(operator, "
            "parameters=...) (alpha: estimate_alpha(op, loader))"
        )
