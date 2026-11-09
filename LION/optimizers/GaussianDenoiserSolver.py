# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
from __future__ import annotations

# numerical imports
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer
import numpy as np

# Import base class
from LION.exceptions.exceptions import LIONSolverException
from LION.models.LIONmodel import LIONmodel
from LION.optimizers.LIONsolver import LIONsolver
from LION.utils.parameter import LIONParameter, OperatorParameters
from LION.utils.image_transforms import random_crop, random_erasing
from LION.utils.rng import seeded

# standard imports
from tqdm import tqdm


class GaussianDenoiserSolver(LIONsolver):
    """Trains a Gaussian denoiser: clean images (the targets) with Gaussian
    noise added are its inputs.

    ``solver_params`` (``default_parameters()``):

    - ``noise_level``: the noise's standard deviation, one value or a range
      ``[low, high]`` (drawn uniformly for each mini-batch);
    - ``patches``: None (whole images), or ``n_patches`` random crops of
      ``patch_size`` x ``patch_size`` of each image, each with a random
      rectangle erased half of the time (``LION.utils.image_transforms``).
    """

    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn,
        operator_params: OperatorParameters = None,
        solver_params: Optional[LIONParameter] = None,
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
            scheduler=scheduler,
            scheduler_step=scheduler_step,
            verbose=verbose,
            device=device,
            seed=seed,
            deterministic=deterministic,
        )
        if verbose >= 2:
            tqdm.write(f"Gaussian Denoiser solver training on device: {self.device}")
        self._noise_range()  # checks it

    def mini_batch_step(self, sino, target):
        """
        This function isresponsible for performing a single mini-batch step of the optimization.
        returns the loss of the mini-batch
        """
        # Forward pass
        if self.do_normalise:
            target = self.model.normalise(target)

        target = self._patches(target)
        noise_level = self._noise_level()
        y = target + noise_level * torch.randn_like(target)

        # if the model accepts noise level as input
        if self.model.takes_noise_level():
            output = self.model(y, noise_level=noise_level)
        else:
            output = self.model(y)
        return self.loss_fn(output, target)

    def _noise_range(self) -> tuple[float, float]:
        """``solver_params.noise_level`` as a range ``(low, high)``."""
        level = np.atleast_1d(np.asarray(self.solver_params.noise_level, dtype=float))
        if level.ndim != 1 or level.shape[0] not in (1, 2):
            raise ValueError(
                "noise_level must be one value or a range [low, high], not "
                f"{self.solver_params.noise_level!r}"
            )
        return float(level[0]), float(level[-1])

    def _noise_level(self) -> float:
        """A noise level drawn uniformly from the range of noise levels."""
        low, high = self._noise_range()
        return low + (high - low) * torch.rand(1).item()

    def _patches(self, images: torch.Tensor) -> torch.Tensor:
        """``images`` as ``solver_params.patches`` says: themselves (None),
        or ``n_patches`` random crops of each, with random erasing."""
        patches = self.solver_params.patches
        if patches is None:
            return images
        size = (patches.patch_size, patches.patch_size)
        crops = [random_crop(images, size) for _ in range(patches.n_patches)]
        return random_erasing(torch.cat(crops, dim=0))

    @staticmethod
    def default_parameters() -> LIONParameter:
        params = LIONParameter()
        params.noise_level = 0.05
        params.patches = None
        return params

    @staticmethod
    def patches(n_patches: int, patch_size: int) -> LIONParameter:
        """``solver_params.patches``: ``n_patches`` random crops of
        ``patch_size`` x ``patch_size`` of each image."""
        return LIONParameter(n_patches=n_patches, patch_size=patch_size)

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
            for batch, (_, targets) in enumerate(
                tqdm(self.validation_loader, disable=self.verbose < 2, leave=False)
            ):
                targets = targets.to(self.device)
                if self.do_normalise:
                    targets = self.model.normalise(targets)

                # the same patches and noise in every validation (per batch, in
                # the loader's order), so values compare across epochs and runs
                with seeded(self.seed, "validation", batch, device=targets.device):
                    data = self._patches(targets)
                    noise_level = self._noise_level()
                    y = data + noise_level * torch.randn_like(data)

                # if the model accepts noise level as input
                if self.model.takes_noise_level():
                    outputs = self.model(y, noise_level=noise_level)
                else:
                    outputs = self.model(y)
                validation_loss = np.append(
                    validation_loss,
                    self.validation_fn(outputs.to(self.device), data.to(self.device))
                    .cpu()
                    .numpy(),
                )

        if self.verbose >= 2:
            tqdm.write(
                f"Validation loss: {validation_loss.mean()} - Validation loss std: {validation_loss.std()}"
            )

        # return to train if it was in train
        if status:
            self.model.train()

        return np.mean(validation_loss)
