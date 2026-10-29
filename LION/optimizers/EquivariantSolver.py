from __future__ import annotations

import functools
import random
from typing import Callable, Optional

import torch
from torch.optim.optimizer import Optimizer

from LION.classical_algorithms.fdk import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
from LION.utils.parameter import LIONParameter
from LION.utils.image_transforms import rotate
from LION.utils.tomosipo_import import import_tomosipo
from LION.utils.citation import citation


def get_rotation_matrix(angle: float):
    theta = torch.tensor(angle)
    s = torch.sin(theta)
    c = torch.cos(theta)
    return torch.tensor([[c, -s], [s, c]])


class RotationGroup:
    """The rotations of an image by the multiples of ``360 / cardinality``
    degrees, as a sequence of functions (``random.choice`` picks one). Kept as
    its ``cardinality``, so solver parameters holding it can be saved."""

    def __init__(self, cardinality: int = 360):
        if 360 % cardinality != 0:
            raise ValueError(f"cardinality must divide 360, got {cardinality}")
        self.cardinality = cardinality

    def __len__(self):
        return self.cardinality

    def __getitem__(self, i):
        if not -self.cardinality <= i < self.cardinality:
            raise IndexError(i)
        i %= self.cardinality
        return functools.partial(rotate, angle=i * (360 / self.cardinality))


class EquivariantSolver(LIONsolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn: Callable,
        geometry: Geometry,
        verbose: bool = True,
        device: torch.device = None,
        solver_params: SolverParams | None = None,
        seed: Optional[int] = 0,
        deterministic: bool = False,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            geometry,
            verbose,
            device,
            solver_params,
            seed=seed,
            deterministic=deterministic,
        )
        self.transformation_group = self.solver_params.transformation_group
        self.alpha = self.solver_params.equivariance_strength
        to_autograd = import_tomosipo(
            "tomosipo.torch_support", "EquivariantSolver"
        ).to_autograd

        self.A = to_autograd(self.op, num_extra_dims=1)
        self.AT = to_autograd(self.op.T, num_extra_dims=1)

    @staticmethod
    def rotation_group(cardinality: int) -> RotationGroup:
        return RotationGroup(cardinality)

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
            # differentiable: the gradient also flows through the network's
            # input here, as in the EI paper (and deepinv's EILoss)
            recon2 = self.model(
                fdk(self.A(transformed_recon1), self.op, differentiable=True)
            )
        else:
            recon2 = self.model(self.A(transformed_recon1))

        return self.loss_fn(self.A(recon1), sino_batch) + self.alpha * self.loss_fn(
            recon2, transformed_recon1
        )
        # data consistency + equivariance

    @staticmethod
    def cite(cite_format="bib"):
        """Where EquivariantSolver comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @inproceedings{chen2021equivariant,
            title={Equivariant imaging: Learning beyond the range space},
            author={Chen, Dongdong and Tachella, Juli{'a}n and Davies, Mike E},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            pages={4379--4388},
            year={2021}
            }
            """,
            mla="""
            Chen, Dongdong, Julián Tachella, and Mike E. Davies.
            "Equivariant imaging: Learning beyond the range space."
             Proceedings of the IEEE/CVF International Conference on Computer Vision. 
            2021
            """,
        )
