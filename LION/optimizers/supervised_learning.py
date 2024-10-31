# numerical imports
import torch
from torch.optim.optimizer import Optimizer
import numpy as np

# Import base class
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.exceptions.exceptions import LIONSolverException
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
from LION.classical_algorithms.fdk import fdk

# standard imports
from tqdm import tqdm


class SupervisedSolver(LIONsolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn,
        geo: Geometry = None,
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

    def validate(self):
        """
        This function is responsible for performing a single validation set of the optimization.
        returns the average loss of the validation set this epoch.
        """
        """
        This function is responsible for performing a single validation set of the optimization.
        returns the average loss of the validation set this epoch.
        """
        if self.check_validation_ready() != 0:
            raise LIONSolverException(
                "Solver not ready for validation. Please call set_validation."
            )

        # these always pass if the above does, this is just to placate static type checker
        assert self.validation_loader is not None
        assert self.validation_fn is not None

        status = self.model.training
        self.model.eval()

        with torch.no_grad():
            validation_loss = np.array([])
            for data, targets in tqdm(self.validation_loader):
                print(self.model.model_parameters.model_input_type)
                if self.model.get_input_type() == ModelInputType.IMAGE:
                    data = fdk(data, self.op)
                print(data.shape)
                outputs = self.model(data)
                validation_loss = np.append(
                    validation_loss, self.validation_fn(targets, outputs)
                )

        if self.verbose:
            print(
                f"Testing loss: {validation_loss.mean()} - Testing loss std: {validation_loss.std()}"
            )

        # return to train if it was in train
        if status:
            self.model.train()

        return np.mean(validation_loss)

    @staticmethod
    def default_parameters() -> SolverParams:
        return SolverParams()
