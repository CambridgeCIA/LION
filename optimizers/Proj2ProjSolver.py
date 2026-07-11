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


class Proj2ProjSolver(LIONsolver):
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
        self.global_step = 0

    def get_mask(self, shape, step):
        # shape: (B, C, H, W)
        mask = torch.ones(shape, device=self.device)
        grid = self.solver_params.grid_size

        for b in range(shape[0]):
            idx = (step + b) % (grid * grid)
            r = idx // grid
            c = idx % grid

            mask[b, :, r::grid, c::grid] = 0
        return mask

    def fill_mean(self, sinos, mask):
        kernel = (
            torch.tensor(
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                dtype=torch.float32,
                device=self.device,
            )
            / 4.0
        )

        kernel = kernel.view(1, 1, 3, 3)
        local_mean_sino = F.conv2d(sinos, kernel, padding=1)

        filled_sinos = (sinos * mask) + (local_mean_sino * (1 - mask))
        return filled_sinos

    @staticmethod
    def default_parameters() -> LIONParameter:
        params = LIONParameter()
        params.grid_size = 4
        params.recon_fn = fdk
        return params

    def mini_batch_step(self, sinos, targets):
        mask = self.get_mask(sinos.shape, self.global_step)
        self.global_step += sinos.shape[0]
        input_sino = self.fill_mean(sinos, mask)

        input_recon = self.recon_fn(input_sino, self.model.operator)
        output_recon = self.model(input_recon)
        output_sino = self.projector(output_recon)

        output_sino_mask = output_sino * (1 - mask)
        target_sino = sinos * (1 - mask)

        batch_loss = ((output_sino_mask - target_sino) ** 2).mean()
        return batch_loss

    # No validation in Proj2Proj
    def validate(self):
        return 0

    def reconstruct(self, sinos):
        input_recon = self.recon_fn(sinos, self.operator)
        output_recon = self.model(input_recon)
        return output_recon
