# numerical imports
from typing import Optional

import torch
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
        verbose: bool = False,
        device: torch.device = None,
        save_folder: str = None,
        seed: Optional[int] = 0,
        deterministic: bool = False,
    ):

        super().__init__(
            model,
            optimizer,
            loss_fn,
            operator_params=operator_params,
            verbose=verbose,
            device=device,
            solver_params=LIONParameter(),
            save_folder=save_folder,
            seed=seed,
            deterministic=deterministic,
        )
        if verbose:
            print("Supervised solver training on device: ", self.device)

    def mini_batch_step(self, sino, target):
        """
        This function isresponsible for performing a single mini-batch step of the optimization.
        returns the loss of the mini-batch
        """
        output = self._forward(sino)
        return self.loss_fn(output, target)

    @staticmethod
    def default_parameters() -> LIONParameter:
        return LIONParameter()
