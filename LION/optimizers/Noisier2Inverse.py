from typing import Callable, Optional
import warnings
import numpy as np
from tqdm import tqdm
from LION.CTtools.ct_geometry import Geometry
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import LIONmodel
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from LION.optimizers.LIONsolver import LIONsolver, warn_if_not_mse
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_utils as ct_utils
from LION.utils.tomosipo_import import import_tomosipo

from LION.utils.image_transforms import gaussian_blur


class Noisier2Inverse(LIONsolver):
    """Noisier2Inverse.

    The training loss is the mean squared error: the target y - M estimates the
    clean projections only in the mean-squared sense (the minimiser of the
    expected MSE is the conditional mean). ``loss_fn`` is only used as the
    default validation and testing function.
    """

    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn,
        solver_params: Optional[LIONParameter] = None,
        geometry: Geometry = None,
        verbose: bool = True,
        device: torch.device = None,
        seed: Optional[int] = 0,
        deterministic: bool = False,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            geometry,
            verbose,
            device,
            solver_params=solver_params,
            seed=seed,
            deterministic=deterministic,
        )
        warn_if_not_mse(loss_fn, "Noisier2Inverse")

        self.operator = self.op  # the base solver already built it
        self.model.geometry = self.geometry
        self.model.operator = self.operator
        to_autograd = import_tomosipo(
            "tomosipo.torch_support", "Noisier2Inverse"
        ).to_autograd
        self.projector = to_autograd(self.operator, num_extra_dims=1)
        self.recon_fn = self.solver_params.recon_fn

    @staticmethod
    def default_parameters() -> LIONParameter:
        params = LIONParameter()
        params.sigma = 3
        params.delta = 1
        params.recon_fn = fdk
        return params

    def mini_batch_step(self, sinos, targets):
        sigma = self.solver_params.sigma
        delta = self.solver_params.delta
        ks = int(sigma * 3) * 2 + 1

        N = torch.randn_like(sinos) * delta
        N = gaussian_blur(N, ks, sigma)
        z = sinos + N

        input_recon = self.recon_fn(z, self.model.operator)
        output_recon = self.model(input_recon)
        output_sino = self.projector(output_recon)
        target_sino = sinos - N

        # Sobolev Loss
        # res = output_sino - target_sino
        # grad_x = res[:, :, :, 1:] - res[:, :, :, :-1]
        # grad_y = res[:, :, 1:, :] - res[:, :, :-1, :]

        # batch_loss = ((output_sino - target_sino)**2).mean() + (grad_x**2).mean() + (grad_y**2).mean()
        batch_loss = ((output_sino - target_sino) ** 2).mean()
        return batch_loss

    def _forward(self, data):
        # validation and testing evaluate the reconstruction
        return self.reconstruct(data)

    def reconstruct(self, sinos):
        input_recon = self.recon_fn(sinos, self.operator)
        output_recon = self.model(input_recon)
        return output_recon
