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
from LION.optimizers.LIONsolver import LIONsolver
from LION.utils.parameter import LIONParameter
import tomosipo as ts
import LION.CTtools.ct_utils as ct_utils
from tomosipo.torch_support import to_autograd

import torchvision.transforms.functional as TF


class Noisier2Inverse(LIONsolver):
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
        print(device)
        super().__init__(
            model,
            optimizer,
            loss_fn,
            geometry,
            verbose,
            device,
            solver_params=solver_params,
        )

        self.operator = ct_utils.make_operator(self.geometry)
        self.model.geometry = self.geometry
        self.model.operator = self.operator
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
        N = TF.gaussian_blur(N, kernel_size=[ks, ks], sigma=[sigma, sigma])
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

    # No validation in Noisier2Inverse
    def validate(self):
        return 0

    def reconstruct(self, sinos):
        input_recon = self.recon_fn(sinos, self.operator)
        output_recon = self.model(input_recon)
        return output_recon
