import copy
import itertools
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
import LION.CTtools.ct_utils as ct_utils
from LION.utils.tomosipo_import import import_tomosipo


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
        # The input sinogram's angles are split into k interleaved subsets. Each
        # network input averages the FBPs of p of them (every p-subset is used);
        # the target is the measured projections at the remaining angles.
        self.sino_split_count = self.solver_params.sino_split_count  # k
        # parameter files from before input_split_count existed used p = 2
        self.input_split_count = getattr(self.solver_params, "input_split_count", 2)
        k, p = self.sino_split_count, self.input_split_count
        if not 1 <= p < k:
            # at least one split must be left over as the target
            raise ValueError(
                "Sparse2Inverse needs 1 <= input_split_count < sino_split_count, "
                f"got input_split_count={p}, sino_split_count={k}"
            )
        self.recon_fn = self.solver_params.recon_fn
        self.split_combinations = self.input_combinations(k, p)
        self._make_sub_operators()

    @staticmethod
    def input_combinations(
        sino_split_count: int, input_split_count: int = 2
    ) -> list[tuple[int, ...]]:
        """All combinations of ``input_split_count`` of the ``sino_split_count``
        splits, e.g. (0, 1), (0, 2), ... for 2 of 4."""
        return list(itertools.combinations(range(sino_split_count), input_split_count))

    @classmethod
    def two_two_strategy(cls, sino_split_count) -> list[tuple[int, ...]]:
        """All pairs of splits (input_split_count = 2). Kept for compatibility."""
        return cls.input_combinations(sino_split_count, 2)

    def _geometry_with_angles(self, angle_indices) -> Geometry:
        """Copy of self.geometry keeping only the given projection angles."""
        geometry = copy.deepcopy(self.geometry)
        geometry.angles = np.asarray(self.geometry.angles)[list(angle_indices)]
        return geometry

    def _complement(self, combo) -> tuple[int, ...]:
        """Splits not in ``combo``: their projections are the training target."""
        return tuple(i for i in range(self.sino_split_count) if i not in combo)

    def _make_sub_operators(self):
        to_autograd = import_tomosipo(
            "tomosipo.torch_support", "Sparse2InverseSolver"
        ).to_autograd

        n = len(self.geometry.angles)
        k = self.sino_split_count
        # split i contains every k-th angle, starting at angle i
        self.subgroup_indices = [list(range(i, n, k)) for i in range(k)]

        self.sub_ops = [
            ct_utils.make_operator(self._geometry_with_angles(indices))
            for indices in self.subgroup_indices
        ]

        # One differentiable projector per set of target splits (the splits not
        # used as input), keyed by that set.
        self.combo_ops_autograd = {}
        for combo in self.split_combinations:
            target_splits = self._complement(combo)
            angle_indices = sorted(
                i for split in target_splits for i in self.subgroup_indices[split]
            )
            op = ct_utils.make_operator(self._geometry_with_angles(angle_indices))
            self.combo_ops_autograd[target_splits] = to_autograd(op, num_extra_dims=1)

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
        # number of interleaved angular splits of each input sinogram (k)
        params.sino_split_count = 4
        # number of splits whose FBPs are averaged to form one network input (p)
        params.input_split_count = 2
        params.recon_fn = fdk
        return params

    def mini_batch_step(self, sinos, targets):
        batch_size = sinos.shape[0]
        subgroup_recons = self._calculate_noisy_sub_recons(sinos)
        batch_loss = 0.0
        total_pixels = 0
        for combo, mean_recon in subgroup_recons.items():
            output_recon = self.model(mean_recon)
            remaining_splits = self._complement(combo)
            projector = self.combo_ops_autograd[remaining_splits]
            objective_idx = []
            for i in remaining_splits:
                objective_idx.extend(self.subgroup_indices[i])
            objective_idx.sort()
            for b in range(batch_size):
                projected_sino = projector(output_recon[b : b + 1])
                target_sino = sinos[b : b + 1, :, objective_idx, :]
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
