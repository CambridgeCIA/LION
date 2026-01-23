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

class Sparse2InverseSolver(LIONsolver):
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

        self.model.geometry = self.geometry  
        self.model._make_operator()
        self.A_full = self.model.A
        self.sino_split_count = self.solver_params.sino_split_count
        self.recon_fn = self.solver_params.recon_fn
        self.split_combinations = self.two_two_strategy(self.sino_split_count)
        self._make_sub_operators()

    @classmethod
    def two_two_strategy(cls, sino_split_count) -> list[tuple[int,int]]:
        #to return all 2 element combinations from 0 to sino_split_count-1
            combos = []
            for i in range(sino_split_count):
                for j in range(i + 1, sino_split_count):
                    combos.append((i, j))
            return combos
    
    def _make_sub_operators(self) -> list[ts.Operator.Operator]:
        self.sub_ops = [] 
        angles = self.geometry.angles.copy()
        n = len(angles)
        k = self.sino_split_count
        angles_per_group = n // k
        remainder = n % k
        self.subgroup_indices = []
        start = 0
        for i in range(k):
            end = start + angles_per_group + (1 if i < remainder else 0)
            self.subgroup_indices.append(list(range(start, end)))
            start = end

        for idx_group in range(k):
            sub_geom = Geometry(
                image_shape=tuple(self.geometry.image_shape),          # tupla, ints
                image_size=tuple(self.geometry.image_size),            # tupla, floats
                angles=[angles[i] for i in self.subgroup_indices[idx_group]],  # list, floats
                voxel_size=tuple(self.geometry.voxel_size),            # tupla, floats
                mode=self.geometry.mode,
                dso=float(self.geometry.dso),
                dsd=float(self.geometry.dsd),
                detector_shape=tuple(self.geometry.detector_shape),    # tupla, ints
                detector_size=tuple(self.geometry.detector_size),      # tupla, floats
                pixel_size=tuple(self.geometry.pixel_size),            # tupla, floats
                image_pos=tuple(self.geometry.image_pos)               # tupla, floats
            )
            sub_op = ct_utils.make_operator(sub_geom)
            self.sub_ops.append(sub_op)

        self.combo_ops_autograd = {} 
        for combo in self.split_combinations:
            combo_angles = []
            for split_idx in combo:
                combo_angles.extend(self.subgroup_indices[split_idx])
            combo_angles = sorted(combo_angles)
            combo_geom = Geometry(
                image_shape=tuple(self.geometry.image_shape),
                image_size=tuple(self.geometry.image_size),
                angles=[angles[i] for i in combo_angles],
                voxel_size=tuple(self.geometry.voxel_size),
                mode=self.geometry.mode,
                dso=float(self.geometry.dso),
                dsd=float(self.geometry.dsd),
                detector_shape=tuple(self.geometry.detector_shape),
                detector_size=tuple(self.geometry.detector_size),
                pixel_size=tuple(self.geometry.pixel_size),
                image_pos=tuple(self.geometry.image_pos)
            )
            combo_op = ct_utils.make_operator(combo_geom)
            self.combo_ops_autograd[combo] = to_autograd(combo_op, num_extra_dims=1)

    def _calculate_noisy_sub_recons(self, sinos):
        subgroup_recons = {}
        for combo in self.split_combinations:
            temp_recons = []
            for split_idx in combo:
                sub_sino = sinos[:, :, self.subgroup_indices[split_idx], :]
                sub_recon = self.recon_fn(sub_sino, self.sub_ops[split_idx])
                temp_recons.append(sub_recon)
            mean_subgroup_recon = torch.mean(torch.stack(temp_recons, dim=1), dim=1)
            subgroup_recons[combo] = mean_subgroup_recon
        return subgroup_recons


    @staticmethod
    def default_parameters() -> LIONParameter:
        params = LIONParameter()
        params.sino_split_count = 4
        params.recon_fn = fdk
        return params
    
    def mini_batch_step(self, sinos, targets):
        batch_size = sinos.shape[0]
        subgroup_recons = self._calculate_noisy_sub_recons(sinos)
        batch_loss = 0.0
        total_pixels = 0
        for combo, mean_recon in subgroup_recons.items():
            output_recon = self.model(mean_recon)
            remaining_splits = [i for i in range(self.sino_split_count) if i not in combo]
            projector_combo = tuple(sorted(remaining_splits))
            projector = self.combo_ops_autograd[projector_combo]
            for b in range(batch_size):
                projected_sino = projector(output_recon[b:b+1])
                target_sino = torch.cat([sinos[b:b+1, :, self.subgroup_indices[i], :] for i in remaining_splits],dim=2)  
                batch_loss += ((projected_sino - target_sino) ** 2).sum()
                total_pixels += projected_sino.numel()
        batch_loss /= total_pixels
        return batch_loss


    # No validation in Sparse2Inverse as it is unsupervised learning
    def validate(self):
        return 0

    def reconstruct(self, sinos):
        subgroup_recons = self._calculate_noisy_sub_recons(sinos)
        outputs = torch.zeros(
            (sinos.shape[0], *self.geometry.image_shape), device=self.device
        )
        for combo, mean_recon in subgroup_recons.items():
           outputs += self.model(mean_recon)
        outputs /= len(subgroup_recons)
        return outputs
