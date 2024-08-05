# numerical imports
import torch
from torch.optim.optimizer import Optimizer
import numpy as np

# Import base class
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.exceptions.exceptions import LIONSolverException
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.optimizers.losses.LIONloss import LIONtrainingLoss
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
from LION.classical_algorithms.fdk import fdk

# standard imports
from tqdm import tqdm


class SupervisedSolver(LIONsolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn: LIONtrainingLoss | torch.nn.Module,
        geo: Geometry,
        verbose: bool = False,
        model_regularization=None,
        device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}"),
    ):
        super().__init__(
            model,
            optimizer,
            loss_fn,
            geo,
            verbose=verbose,
            device=device,
            solver_params=SolverParams(),
        )

        self.op = make_operator(self.geo)

    def mini_batch_step(self, sino, target):
        """
        This function isresponsible for performing a single mini-batch step of the optimization.
        returns the loss of the mini-batch
        """
        # Forward pass
        if self.model.get_input_type() == ModelInputType.IMAGE:
            data = fdk(sino, self.op)
            if self.do_normalize:
                data = self.normalize(data)
        else:
            data = sino

        output = self.model(data)
        return self.loss_fn(output, target)

    @staticmethod
    def default_parameters() -> SolverParams:
        return SolverParams()
