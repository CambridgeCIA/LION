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


class GaussianDenoiserSolver(LIONsolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn,
        geometry: Geometry = None,
        verbose: bool = False,
        model_regularization=None,
        device: torch.device = None,
        save_folder: str = None,
        noise_level: np.ndarray = np.array([0.05]),
    ):

        super().__init__(
            model,
            optimizer,
            loss_fn,
            geometry=geometry,
            verbose=verbose,
            device=device,
            solver_params=SolverParams(),
            save_folder=save_folder,
        )
        if verbose:
            print("Gaussian Denoiser solver training on device: ", device)
        self.op = make_operator(self.geometry)
        self.patch = None

        # Make range of noise levels if noise_level is a single value
        if noise_level.ndim == 1 and noise_level.size == 1:
            noise_level = np.array([noise_level[0], noise_level[0]])
        elif noise_level.ndim != 1 or noise_level.size != 2:
            raise LIONSolverException(
                "noise_level must be a numpy array of length 2, or a single value."
            )
        self.noise_level = noise_level

    def mini_batch_step(self, sino, target):
        """
        This function isresponsible for performing a single mini-batch step of the optimization.
        returns the loss of the mini-batch
        """
        # Forward pass
        if self.do_normalize:
            target = self.normalize(target)

        if self.patch is not None:
            data = self.patch.random_erasing(
                torch.cat(
                    [
                        self.patch.random_crop(target)
                        for _ in range(self.patch.n_patches)
                    ],
                    dim=0,
                )
            )

        noise_level = np.random.uniform(
            self.noise_level[0], self.noise_level[1], size=(1)
        )
        y = data + noise_level * torch.randn_like(data)

        # if the model accepsts noise level as input
        if hasattr(self.model, "use_noise_level") and self.model.use_noise_level:
            y = torch.cat([y, noise_level * torch.ones_like(y)], dim=1)
        output = self.model(y)
        return self.loss_fn(output, target)

    @staticmethod
    def default_parameters() -> SolverParams:
        return SolverParams()

    def set_patch_strategy(self, n_patches, patch_size):
        """
        Set the patch strategy for the solver.
        :param n_patches: Number of patches to extract.
        :param patch_size: Size of each patch.
        """
        self.patch = LIONParameter()
        self.patch.n_patches = n_patches
        self.patch.random_crop = RandomCrop((patch_size, patch_size))
        self.patch.random_erasing = RandomErasing()

    def validate(self):
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
            for _, targets in tqdm(self.validation_loader):
                targets = targets.to(self.device)
                if self.do_normalize:
                    targets = self.normalize(targets)

                if self.patch is not None:
                    data = self.patch.random_erasing(
                        torch.cat(
                            [
                                self.patch.random_crop(targets)
                                for _ in range(self.patch.n_patches)
                            ],
                            dim=0,
                        )
                    )

                noise_level = np.random.uniform(
                    self.noise_level[0], self.noise_level[1], size=(1)
                )
                y = data + noise_level * torch.randn_like(data)

                # if the model accepsts noise level as input
                if (
                    hasattr(self.model, "use_noise_level")
                    and self.model.use_noise_level
                ):
                    y = torch.cat([y, noise_level * torch.ones_like(y)], dim=1)
                outputs = self.model(y)
                validation_loss = np.append(
                    validation_loss,
                    self.validation_fn(y.to(self.device), outputs.to(self.device))
                    .cpu()
                    .numpy(),
                )

        if self.verbose:
            print(
                f"Testing loss: {validation_loss.mean()} - Testing loss std: {validation_loss.std()}"
            )

        # return to train if it was in train
        if status:
            self.model.train()

        return np.mean(validation_loss)
