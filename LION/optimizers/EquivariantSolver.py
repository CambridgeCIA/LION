# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri, Charlie Shoebridge
# =============================================================================
from __future__ import annotations

import functools
import random
from typing import Callable, Optional

import torch
from torch.optim.optimizer import Optimizer

from LION.operators.ct.fdk import fdk
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.optimizers.LIONsolver import LIONsolver
from LION.utils.parameter import LIONParameter, OperatorParameters
from LION.utils.image_transforms import rotate
from LION.utils.citation import citation


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
    """Equivariant imaging (Chen, Tachella and Davies, 2021): trains from
    measurements alone. The loss is data consistency, ``loss_fn(A(f(y)), y)``,
    plus ``equivariance_strength`` times the equivariance of the whole
    reconstruction to a random transformation T of the group:
    ``loss_fn(f(A(T f(y))), T f(y))``. For CT, f reconstructs (FDK, then the
    model, for image-domain models).
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
        self.transformation_group = self.solver_params.transformation_group
        self.alpha = self.solver_params.equivariance_strength
        # the operator and its adjoint as torch autograd functions
        self.A, _ = self.op.autograd()

    @staticmethod
    def rotation_group(cardinality: int) -> RotationGroup:
        """The rotations by multiples of ``360 / cardinality`` degrees."""
        return RotationGroup(cardinality)

    @staticmethod
    def default_parameters() -> LIONParameter:
        """``transformation_group``: the transformations (the 360 rotations
        by whole degrees); ``equivariance_strength``: the weight of the
        equivariance term (100)."""
        param = LIONParameter()
        param.transformation_group = EquivariantSolver.rotation_group(360)
        param.equivariance_strength = 100
        return param

    def mini_batch_step(self, y, target_batch) -> torch.Tensor:
        """The loss: data consistency plus the weighted equivariance term
        (see the class; ``target_batch`` isn't used: self-supervised)."""
        random_transform = random.choice(self.transformation_group)

        if needs_image := (
            self.model.model_parameters.model_input_type == ModelInputType.IMAGE
        ):
            first_input = fdk(y, self.op)
        else:
            first_input = y
        recon1 = self.model(first_input)

        transformed_recon1 = random_transform(recon1)

        if needs_image:
            # differentiable: the gradient also flows through the network's
            # input here, as in the EI paper (and deepinv's EILoss)
            recon2 = self.model(
                fdk(self.A(transformed_recon1), self.op, differentiable=True)
            )
        else:
            recon2 = self.model(self.A(transformed_recon1))

        # data consistency + equivariance
        loss = self.loss_fn(self.A(recon1), y) + self.alpha * self.loss_fn(
            recon2, transformed_recon1
        )
        return self._regularized(loss, first_input, recon1)

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
