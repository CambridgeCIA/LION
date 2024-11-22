# numerical imports
import torch
from torch.optim.optimizer import Optimizer
import numpy as np

# Import base class
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.exceptions.exceptions import LIONSolverException
from LION.models.LIONmodel import LIONmodel
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import ModelInputType

# standard imports
from tqdm import tqdm

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
        geometry: Geometry = None,
        verbose: bool = False,
        model_regularization=None,
        device: torch.device = None,
    ):

        super().__init__(
            model,
            optimizer,
            loss_fn,
            geometry=geometry,
            verbose=verbose,
            device=device,
            solver_params=SolverParams(),
        )
        if verbose:
            print("Supervised solver training on device: ", device)
        self.op = make_operator(self.geometry)

    def mini_batch_step(self, sino, target):
        """
        This function is responsible for performing a single mini-batch step of the optimization.
        returns the loss of the mini-batch
        """
        # Forward pass
        if self.model.get_input_type() == ModelInputType.IMAGE:
            data = fdk(sino, self.op)
            if self.do_normalize:
                data = self.normalize(data)
        else:
            data = sino

        return self.loss_fn(data, self.model)

    @staticmethod
    def default_parameters() -> SolverParams:
        return SolverParams()
