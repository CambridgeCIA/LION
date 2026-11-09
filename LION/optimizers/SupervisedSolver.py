# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri, Charlie Shoebridge
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

# standard imports


class SupervisedSolver(LIONsolver):
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
            tqdm.write(f"Supervised solver training on device: {self.device}")

    def mini_batch_step(self, y, target):
        """
        This function isresponsible for performing a single mini-batch step of the optimization.
        returns the loss of the mini-batch
        """
        output = self._forward(y)
        loss = self.loss_fn(output, target)
        return self._regularized(loss, y, output, target, self._forward)

    @staticmethod
    def default_parameters() -> LIONParameter:
        return LIONParameter()
