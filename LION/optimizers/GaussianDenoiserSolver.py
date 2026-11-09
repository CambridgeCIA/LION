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
from LION.utils.image_transforms import (
    random_crop,
    random_erasing,
    random_flip_rotate,
)
from LION.utils.rng import seeded

# standard imports
from tqdm import tqdm


class GaussianDenoiserSolver(LIONsolver):
    """Trains a Gaussian denoiser (e.g. for Plug-and-Play): clean images (the
    targets), with Gaussian noise added, are its inputs. As in the
    literature (DnCNN, FFDNet and DRUNet in KAIR, Zhang et al.; the
    gradient-step denoiser, Hurault et al.), each image gets its own noise
    level, drawn from a range, and is randomly flipped and rotated.

    **Noise levels are in the units of the data the model sees**: the noise
    is added after the model's normalisation (``model.set_normalisation()``),
    if any. The literature, and LION, assume images in [0, 1] (e.g. a
    "dataset" or "custom" normalisation to [0, 1]), where its noise levels
    are given as k/255: 25/255 = 0.098, 50/255 = 0.196. With another range
    (e.g. no normalisation), scale the noise levels with it. The same holds
    for the noise level a PnP reconstruction gives the denoiser.

    ``solver_params`` (``default_parameters()``):

    - ``noise_level``: the noise's standard deviation: one value, or a range
      ``[low, high]``, drawn uniformly for each image (or patch). A model
      that takes the noise level (``model.takes_noise_level()``) is given
      one per image.
    - ``patches``: None (whole images), or ``n_patches`` random crops of
      ``patch_size`` x ``patch_size`` of each image
      (``GaussianDenoiserSolver.patches(n_patches, patch_size)``).
    - ``augment``: each training image (or patch) flipped and rotated by a
      random one of the 8 symmetries of the square (of the rectangle's 4, for
      images that aren't square).
    - ``random_erasing``: each training image (or patch), half of the time,
      with a random rectangle set to 0 (``image_transforms.random_erasing``).
    - ``validation_noise_levels``: None, validation draws the noise levels
      as training does (the same ones every time); or a list of them (e.g.
      ``[15 / 255, 25 / 255, 50 / 255]``), every validation image at each,
      the validation value being their mean.
    - ``jacobian_penalty``: None, or a penalty on the denoiser's Jacobian
      added to the loss, for PnP methods that converge
      (``LION.losses.jacobian_penalty``): ``FirmlyNonexpansivePenalty()``
      (Pesquet et al. 2021) or ``ProximalDenoiserPenalty()`` (Hurault et al.
      2022, for a gradient-step denoiser). Both papers add it after training
      without it (e.g. ``train_again``, or ``LIONmodel.load`` and another
      solver). Cite the one used (its ``cite()``).
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

    def mini_batch_step(self, y, target):
        """The loss of the model on ``target`` (clean images) with noise
        added (``y`` isn't used: the inputs are made from the targets)."""
        if self.do_normalise:
            target = self.model.normalise(target)
        target = self._training_images(target)
        noise_level = self._noise_levels(target.shape[0], target.device)
        noisy = self._noisy(target, noise_level)
        output = self._model(noisy, noise_level)
        loss = self.loss_fn(output, target)
        penalty = self.solver_params.jacobian_penalty
        if penalty is not None:
            denoise = lambda x: self._model(x, noise_level)  # noqa: E731
            loss = loss + penalty(denoise, target, noisy, output)
        return loss

    @staticmethod
    def _noisy(clean: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        """``clean`` with Gaussian noise of standard deviation ``noise_level``
        (one per image) added."""
        shape = (-1,) + (1,) * (clean.dim() - 1)
        return clean + noise_level.reshape(shape) * torch.randn_like(clean)

    def _model(self, noisy: torch.Tensor, noise_level: torch.Tensor):
        """The model's output on ``noisy``, given ``noise_level`` if it takes
        one."""
        if self.model.takes_noise_level():
            return self.model(noisy, noise_level=noise_level)
        return self.model(noisy)

    def _denoise(self, clean: torch.Tensor, noise_level: torch.Tensor):
        """The model's output on ``clean`` with noise added."""
        return self._model(self._noisy(clean, noise_level), noise_level)

    def _noise_range(self) -> tuple[float, float]:
        """``solver_params.noise_level`` as a range ``(low, high)``."""
        level = np.atleast_1d(np.asarray(self.solver_params.noise_level, dtype=float))
        if level.ndim != 1 or level.shape[0] not in (1, 2):
            raise ValueError(
                "noise_level must be one value or a range [low, high], not "
                f"{self.solver_params.noise_level!r}"
            )
        return float(level[0]), float(level[-1])

    def _noise_levels(self, n: int, device) -> torch.Tensor:
        """``n`` noise levels drawn uniformly from the range, one per image."""
        low, high = self._noise_range()
        return (low + (high - low) * torch.rand(n)).to(device)

    def _patches(self, images: torch.Tensor) -> torch.Tensor:
        """``images`` as ``solver_params.patches`` says: themselves (None),
        or ``n_patches`` random crops of each."""
        patches = self.solver_params.patches
        if patches is None:
            return images
        size = (patches.patch_size, patches.patch_size)
        return torch.cat(
            [random_crop(images, size) for _ in range(patches.n_patches)], dim=0
        )

    def _training_images(self, images: torch.Tensor) -> torch.Tensor:
        """The images trained on: patches, flipped and rotated, erased, as
        ``solver_params`` says."""
        images = self._patches(images)
        if self.solver_params.augment:
            images = random_flip_rotate(images)
        if self.solver_params.random_erasing:
            images = random_erasing(images)
        return images

    @staticmethod
    def default_parameters() -> LIONParameter:
        params = LIONParameter()
        params.noise_level = 0.05
        params.patches = None
        params.augment = True
        params.random_erasing = False
        params.validation_noise_levels = None
        params.jacobian_penalty = None
        return params

    @staticmethod
    def patches(n_patches: int, patch_size: int) -> LIONParameter:
        """``solver_params.patches``: ``n_patches`` random crops of
        ``patch_size`` x ``patch_size`` of each image."""
        return LIONParameter(n_patches=n_patches, patch_size=patch_size)

    def validate(self):
        """The validation value: ``validation_fn`` of the model's outputs on
        the validation images with noise added, against the images (on
        patches, if training is; not flipped, rotated or erased). The noise is
        the same in every validation."""
        if self.check_validation_ready() != 0:
            raise LIONSolverException(
                "Solver not ready for validation. Please call set_validation."
            )

        # these always pass if the above does, this is just to placate static type checker
        assert self.validation_loader is not None
        assert self.validation_fn is not None

        status = self.model.training
        self.model.eval()

        fixed = self.solver_params.validation_noise_levels
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
                    n = data.shape[0]
                    if fixed is None:
                        levels = [self._noise_levels(n, data.device)]
                    else:
                        levels = [
                            torch.full((n,), float(level), device=data.device)
                            for level in fixed
                        ]
                    outputs = [self._denoise(data, level) for level in levels]
                values = [
                    self.validation_fn(output.to(self.device), data.to(self.device))
                    .reshape(-1)
                    .cpu()
                    .numpy()
                    for output in outputs
                ]
                validation_loss = np.append(validation_loss, np.mean(values, axis=0))

        if self.verbose >= 2:
            tqdm.write(
                f"Validation loss: {validation_loss.mean()} - Validation loss std: {validation_loss.std()}"
            )

        # return to train if it was in train
        if status:
            self.model.train()

        return np.mean(validation_loss)
