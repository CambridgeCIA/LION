from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
from LION.CTtools.ct_geometry import Geometry
from LION.classical_algorithms.fdk import fdk
from tomosipo.torch_support import to_autograd
from LION.exceptions.exceptions import LIONSolverException
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.optimizers.LIONsolver import LIONsolver, SolverParams


class ARParams(SolverParams):
    def __init__(
        self,
        early_stopping: bool = False,
        no_steps: int = 150,
        step_size: float = 1e-6,
        momentum: float = 0.5,
        beta_rate: float = 0.95,
        lambd_gp: float = 1e-3,
    ):
        super().__init__()
        self.early_stopping = early_stopping
        self.no_steps = no_steps
        self.step_size = step_size
        self.momentum = momentum
        self.beta_rate = beta_rate
        self.lambd_gp = lambd_gp


class ARSolver(LIONsolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        recon_optimizer_type,
        geometry: Geometry,
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
        return ARParams()

    def var_energy(self, x, y):
        return 0.5 * ((self.A(x) - y) ** 2).sum() + self.alpha * (self.model(x)).sum()

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
            sino.requires_grad_()
            recon = fdk(sino, self.op)
            if self.do_normalize:
                recon = self.normalize(recon)
                target = self.normalize(target)

            recon = torch.nn.Parameter(recon)

            optimizer = self.recon_optimizer_type(
                [recon],
                lr=self.solver_params.step_size,
                momentum=self.solver_params.momentum,
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
        printed = 0
        for sino, target in tqdm(self.test_loader):
            sino.requires_grad_()
            bad_recon = fdk(sino, self.op)
            if self.do_normalize:
                bad_recon = self.normalize(bad_recon)
                target = self.normalize(target)

            recon = torch.nn.Parameter(bad_recon.clone())

            optimizer = self.recon_optimizer_type(
                [recon],
                lr=self.solver_params.step_size,
                momentum=self.solver_params.momentum,
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

            if printed < 3:
                printed += 1
                print(f"Printing img {printed}")
                plt.figure()
                plt.subplot(131)
                plt.imshow(target[0].detach().cpu().numpy().T)
                plt.clim(torch.min(target[0]).item(), torch.max(target[0]).item())
                plt.gca().set_title("Ground Truth")
                plt.subplot(132)
                # should cap max / min of plots to actual max / min of gt
                plt.imshow(bad_recon[0].detach().cpu().numpy().T)
                plt.clim(torch.min(target[0]).item(), torch.max(target[0]).item())
                plt.gca().set_title("FBP")
                plt.subplot(133)
                plt.imshow(recon[0].detach().cpu().numpy().T)
                plt.clim(torch.min(target[0]).item(), torch.max(target[0]).item())
                plt.gca().set_title("AR")
                # reconstruct filepath with suffix i
                plt.savefig(f"ar_test{printed}.png", dpi=700)
                plt.close()

            with torch.no_grad():
                test_loss = np.append(test_loss, self.testing_fn(recon, target).item())

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
        if self.do_normalize and self.model.get_input_type() == ModelInputType.IMAGE:
            bad_recon = self.normalize(bad_recon)
            target_batch = self.normalize(target_batch)

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
            print("Mukherjee, Subhadip, et al.")
            print('"Data-Driven Convex Regularizers for Inverse Problems."')
            print(
                "ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024"
            )
            print("arXiv:2008.02839 (2020).")
        elif cite_format == "bib":
            string = """
            @inproceedings{mukherjee2024data,
            title={Data-Driven Convex Regularizers for Inverse Problems},
            author={Mukherjee, S and Dittmer, S and Shumaylov, Z and Lunz, S and {\"O}ktem, O and Sch{\"o}nlieb, C-B},
            booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
            pages={13386--13390},
            year={2024},
            organization={IEEE}
            }
            """
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )
