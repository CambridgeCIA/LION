# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  :
# =============================================================================
# numerical imports
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer

# Import base class
from LION.utils.parameter import OperatorParameters
from LION.models.LIONmodel import LIONmodel
from LION.optimizers.LIONsolver import LIONsolver
from LION.utils.parameter import LIONParameter
from LION.models.LIONmodel import ModelInputType

# standard imports

""" 
This file contains the implementation of the SelfSupervisedSolver class, which is a subclass of the LIONsolver class.
The main difference with other solver is the shape of the loss function. 
A self-supervised loss should take 
"""


class SelfSupervisedSolver(LIONsolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn,
        operator_params: OperatorParameters = None,
        verbose: bool = False,
        model_regularization=None,
        device: torch.device = None,
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
            seed=seed,
            deterministic=deterministic,
        )
        if verbose:
            print("Self-supervised solver training on device: ", self.device)

    def mini_batch_step(self, sino, target):
        """
        This function is responsible for performing a single mini-batch step of the optimization.
        returns the loss of the mini-batch
        """
        # Forward pass
        if self.model.get_input_type() == ModelInputType.IMAGE:
            data = self.model.initial_reconstruction(sino, self.op)
            if self.do_normalise:
                data = self.model.normalise(data)
        else:
            data = sino

        return self.loss_fn(data, self.model)

    @staticmethod
    def default_parameters() -> LIONParameter:
        return LIONParameter()
