import random
from typing import Callable
import torch
import torchvision.transforms.functional as TF
from torch.optim.optimizer import Optimizer
from tomosipo.torch_support import to_autograd
from LION.CTtools.ct_geometry import Geometry
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
from LION.utils.parameter import LIONParameter


def get_rotation_matrix(angle: float):
    theta = torch.tensor(angle)
    s = torch.sin(theta)
    c = torch.cos(theta)
    return torch.tensor([[c, -s], [s, c]])


class EquivariantSolver(LIONsolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn: Callable,
        geometry: Geometry,
        verbose: bool = True,
        device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}"),
        solver_params: SolverParams | None = None,
    ) -> None:
        super().__init__(
            model, optimizer, loss_fn, geometry, verbose, device, solver_params
        )
        self.transformation_group = self.solver_params.transformation_group
        self.alpha = self.solver_params.equivariance_strength
        self.A = to_autograd(self.op, num_extra_dims=1)
        self.AT = to_autograd(self.op.T, num_extra_dims=1)

    @staticmethod
    def rotation_group(cardinality: int):
        assert 360 % cardinality == 0
        angle_increment = 360 / cardinality

        return [lambda x: TF.rotate(x, i * angle_increment) for i in range(cardinality)]

    @staticmethod
    def default_parameters() -> LIONParameter:
        param = LIONParameter()
        param.transformation_group = EquivariantSolver.rotation_group(360)
        param.equivariance_strength = 100
        return param

    def mini_batch_step(self, sino_batch, target_batch) -> torch.Tensor:
        random_transform = random.choice(self.transformation_group)

        if needs_image := (
            self.model.model_parameters.model_input_type == ModelInputType.IMAGE
        ):
            recon1 = self.model(fdk(sino_batch, self.op))
        else:
            recon1 = self.model(sino_batch)

        transformed_recon1 = random_transform(recon1)

        if needs_image:
            recon2 = self.model(fdk(self.A(transformed_recon1), self.op))
        else:
            recon2 = self.model(self.A(transformed_recon1))

        return self.loss_fn(self.A(recon1), sino_batch) + self.alpha * self.loss_fn(
            recon2, transformed_recon1
        )
        # data consistency + equivariance

    @staticmethod
    def cite(cite_format="MLA"):

        if cite_format == "MLA":
            print("Chen, Dongdong, Julián Tachella, and Mike E. Davies.")
            print('"Equivariant imaging: Learning beyond the range space."')
            print(
                "\x1B[3m Proceedings of the IEEE/CVF International Conference on Computer Vision. \x1B[0m"
            )
            print("2021")
        elif cite_format == "bib":
            string = """
            @inproceedings{chen2021equivariant,
            title={Equivariant imaging: Learning beyond the range space},
            author={Chen, Dongdong and Tachella, Juli{\'a}n and Davies, Mike E},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            pages={4379--4388},
            year={2021}
            }"""
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )
