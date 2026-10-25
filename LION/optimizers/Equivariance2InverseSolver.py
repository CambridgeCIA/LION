from typing import Callable, Optional
import numpy as np
from LION.CTtools.ct_geometry import Geometry
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import LIONmodel
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from LION.optimizers.LIONsolver import LIONsolver, warn_if_not_mse
from LION.utils.parameter import LIONParameter
import tomosipo as ts
import LION.CTtools.ct_utils as ct_utils
from tomosipo.torch_support import to_autograd
import random
import torchvision.transforms as TT


class Equivariance2InverseSolver(LIONsolver):
    """Equivariance2Inverse.

    The training loss is the mean squared error, as in the method's
    formulation (masked data consistency plus equivariance). ``loss_fn`` is
    only used as the default validation and testing function.
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
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            geometry,
            verbose,
            device,
            solver_params=solver_params,
        )
        warn_if_not_mse(loss_fn, "Equivariance2InverseSolver")

        self.operator = self.op  # the base solver already built it
        self.model.geometry = self.geometry
        self.model.operator = self.operator
        self.projector = to_autograd(self.operator, num_extra_dims=1)
        self.recon_fn = self.solver_params.recon_fn

    @staticmethod
    def default_parameters() -> LIONParameter:
        params = LIONParameter()
        params.recon_fn = fdk
        params.I0 = 500
        params.sigma = (50) ** (0.5)
        params.sigma_blur = 0.8
        # noise model of ct_utils.sinogram_add_noise for the rotated sinograms;
        # I0, sigma and sigma_blur above are for the relative model
        params.noise_model = "relative"
        return params

    def mini_batch_step(self, sinos, targets):
        # masking
        NP = sinos.shape[2]
        YJ_num = torch.randint(0, NP, (1,)).item()
        YJ = sinos[:, :, YJ_num, :]

        YJc = sinos.clone()
        YJc[:, :, YJ_num, :] = 0

        weight = NP / (NP - 1)
        RJc = self.recon_fn(YJc * weight, self.model.operator)
        output_recon_1 = self.model(RJc)

        output_sino_1 = self.projector(output_recon_1)
        AJ = output_sino_1[:, :, YJ_num, :]
        batch_loss = ((AJ - YJ) ** 2).mean()

        angle = random.uniform(0, 360)
        rotated_output_recon_1 = TT.functional.rotate(
            output_recon_1, angle, interpolation=TT.InterpolationMode.BILINEAR
        )
        rotated_output_recon_1 = torch.clamp(rotated_output_recon_1, min=0.0)
        rotated_sinogram = self.projector(rotated_output_recon_1)
        rotated_sinogram = torch.clamp(rotated_sinogram, min=0.0)
        rotated_sinogram_noisy = ct_utils.sinogram_add_noise(
            rotated_sinogram,
            I0=self.solver_params.I0,
            sigma=self.solver_params.sigma,
            sigma_blur=self.solver_params.sigma_blur,
            ks_value=3,
            flat_field=None,
            dark_field=None,
            # the relative model, unless the solver parameters say otherwise
            # (set them to match the noise of the experiment's data)
            model=getattr(self.solver_params, "noise_model", "relative"),
        )

        rotated_noisy_image = self.recon_fn(rotated_sinogram_noisy, self.model.operator)
        output_recon_2 = self.model(rotated_noisy_image)
        batch_loss += ((output_recon_2 - rotated_output_recon_1) ** 2).mean()
        return batch_loss

    def _forward(self, data):
        # validation and testing evaluate the reconstruction
        return self.reconstruct(data)

    def reconstruct(self, sinos):
        input_recon = self.recon_fn(sinos, self.operator)
        output_recon = self.model(input_recon)
        return output_recon
