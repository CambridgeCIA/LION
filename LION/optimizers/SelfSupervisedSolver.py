# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
# numerical imports
from typing import Optional

import torch
from tqdm import tqdm
from torch.optim.optimizer import Optimizer

# Import base class
from LION.utils.parameter import OperatorParameters
from LION.models.LIONmodel import LIONmodel
from LION.optimizers.LIONsolver import LIONsolver
from LION.utils.parameter import LIONParameter
from LION.models.LIONmodel import ModelInputType


class SelfSupervisedSolver(LIONsolver):
    """Training with a loss of the model and its input alone,
    ``loss_fn(input, model)`` (e.g. SURE): the input is the initial
    reconstruction for image-domain models, the measurements otherwise. The
    targets aren't used. No settings of its own."""

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
    ):

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
        if verbose >= 2:
            tqdm.write(f"Self-supervised solver training on device: {self.device}")

    def mini_batch_step(self, y, target):
        """``loss_fn(input, model)`` (``target`` isn't used)."""
        # Forward pass
        if self.model.get_input_type() == ModelInputType.IMAGE:
            data = self.model.initial_reconstruction(y, self.op)
            if self.do_normalise:
                data = self.model.normalise(data)
        else:
            data = y

        return self._regularized(self.loss_fn(data, self.model), data)

    @staticmethod
    def default_parameters() -> LIONParameter:
        """None: the solver has no settings."""
        return LIONParameter()
