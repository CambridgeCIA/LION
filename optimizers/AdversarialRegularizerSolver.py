# This file is part of LION library
# License : BSD-3
#
# Author  : Zakhar Shumaylow, Charlie Shoebridge
# Modifications: Ander Biguri
# =============================================================================
from __future__ import annotations

import numpy as np
import torch
from tomosipo.torch_support import to_autograd
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from LION.classical_algorithms.fdk import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.exceptions.exceptions import LIONSolverException
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
from LION.utils.parameter import LIONParameter


class ARParams(SolverParams):
    def __init__(
        self,
        optimizer_params: LIONParameter,
        early_stopping: bool = False,
        no_steps: int = 150,
        step_size: float = 1e-6,
        beta_rate: float = 0.95,
        lambd_gp: float = 1e-1,
    ):
        super().__init__()
        self.early_stopping = early_stopping
        self.no_steps = no_steps
        self.step_size = step_size
        self.optimizer_params = optimizer_params
        self.beta_rate = beta_rate
        self.lambd_gp = lambd_gp


class ARSolver(LIONsolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        recon_optimizer_type,
        geometry: Geometry = None,
        verbose: bool = True,
        device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}"),
        solver_params: ARParams | None = None,
    ) -> None:
        super().__init__(
            model, optimizer, self.wgan_loss, geometry, verbose, device, solver_params
        )
        self.recon_optimizer_type = recon_optimizer_type
        self.A = to_autograd(self.op, num_extra_dims=1)
        self.AT = to_autograd(self.op.T, num_extra_dims=1)
        self.alpha = 1.0

    def estimate_alpha(self, dataloader):
        if dataloader is not None:
            residual = 0.0
            for _, (sino, target) in enumerate(dataloader):
                residual += torch.linalg.vector_norm(
                    self.AT(self.A(target) - sino), dim=(2, 3)
                ).mean()
                # residual += torch.sqrt(((self.AT(self.A(target) - data))**2).sum())
            self.alpha = residual / len(dataloader.dataset)
        if self.verbose:
            print("Estimated alpha: " + str(self.alpha))

    @staticmethod
    def default_parameters() -> ARParams:
        raise LIONSolverException(
            "Cannot construct default parameters for ARSolver, need to provide validation/testing optimizer parameters"
        )

    def var_energy(self, recon, sino):
        return (
            0.5 * ((self.A(recon) - sino) ** 2).sum()
            + self.alpha * (self.model(recon)).sum()
        )

    def validate(self):
        if self.check_validation_ready() != 0:
            raise LIONSolverException(
                "Solver not ready for validation. Please call set_validation."
            )

        # these always pass if the above does, this is just to placate static type checker
        assert self.validation_loader is not None
        assert self.validation_fn is not None

        status = self.model.training
        self.model.eval()

        self.estimate_alpha(
            self.validation_loader if self.validation_loader is not None else None
        )

        validation_loss = np.array([])
        for sino, target in tqdm(self.validation_loader):
            recon = fdk(sino, self.op)

            recon = torch.nn.Parameter(recon)

            optimizer = self.recon_optimizer_type(
                [recon],
                *self.solver_params.optimizer_params.unpack(),
                lr=self.solver_params.step_size,
            )
            lr = self.solver_params.step_size

            for _ in range(self.solver_params.no_steps):
                optimizer.zero_grad()

                energy = self.var_energy(recon, sino)
                energy.backward(retain_graph=True)

                assert recon.grad is not None
                while (
                    self.var_energy(recon - recon.grad * lr, sino)
                    > energy - 0.5 * lr * (recon.grad.norm(dim=(2, 3)) ** 2).mean()
                ):
                    lr = self.solver_params.beta_rate * lr
                for g in optimizer.param_groups:
                    g["lr"] = lr

                optimizer.step()

            validation_loss = np.append(
                validation_loss, self.validation_fn(recon, target).item()
            )

        if self.verbose:
            print(
                f"Validation loss mean: {validation_loss.mean()} - Validation loss std: {validation_loss.std()}"
            )

        # return to train if it was in train
        if status:
            self.model.train()

        return np.mean(validation_loss)

    def test(self):
        if self.check_testing_ready() != 0:
            raise LIONSolverException(
                "Solver not ready for testing. Please call set_testing."
            )

        # these always pass if the above does, this is just to placate static type checker
        assert self.test_loader is not None
        assert self.testing_fn is not None

        status = self.model.training
        self.model.eval()

        if self.verbose:
            print(f"Testing model after {self.current_epoch} epochs of training")

        self.estimate_alpha(self.test_loader)

        test_loss = np.array([])
        for sino, target in tqdm(self.test_loader):
            bad_recon = fdk(sino, self.op)

            recon = torch.nn.Parameter(bad_recon.clone()).requires_grad_(True)

            optimizer = self.recon_optimizer_type(
                [recon],
                *self.solver_params.optimizer_params.unpack(),
                lr=self.solver_params.step_size,
            )
            lr = self.solver_params.step_size

            for _ in tqdm(range(self.solver_params.no_steps)):
                optimizer.zero_grad()

                energy = self.var_energy(recon, sino)
                energy.backward(retain_graph=True)

                assert recon.grad is not None
                while (
                    self.var_energy(recon - recon.grad * lr, sino)
                    > energy - 0.5 * lr * (recon.grad.norm(dim=(2, 3)) ** 2).mean()
                ):
                    lr = self.solver_params.beta_rate * lr
                for g in optimizer.param_groups:
                    g["lr"] = lr

                optimizer.step()

                with torch.no_grad():
                    test_loss = np.append(
                        test_loss, self.testing_fn(recon, target).item()
                    )

        if self.verbose:
            print(
                f"Testing loss mean: {test_loss.mean()} - Testing loss std: {test_loss.std()}"
            )

        # return to train if it was in train
        if status:
            self.model.train()

        return test_loss

    def wgan_loss(self, sino_batch, target_batch):
        bad_recon = fdk(sino_batch, self.op)
        if self.do_normalise and self.model.get_input_type() == ModelInputType.IMAGE:
            bad_recon = self.model.normalise.normalise(bad_recon)
            target_batch = self.model.normalise.normalise(target_batch)

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
