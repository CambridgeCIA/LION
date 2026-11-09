# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri, Alvaro Exposito
# =============================================================================
import itertools
from typing import Optional
from LION.models.LIONmodel import LIONmodel
import torch
from torch.optim.optimizer import Optimizer
from LION.optimizers.LIONsolver import LIONsolver
from LION.utils.parameter import LIONParameter, OperatorParameters
from LION.utils.citation import citation


class Sparse2InverseSolver(LIONsolver):
    """Sparse2Inverse: the measurements are split into ``sino_split_count``
    interleaved parts (for CT, every k-th angle). The model's input is the mean
    reconstruction of ``input_split_count`` parts (``recon_fn``, by default the
    operator's default reconstruction), and its output, projected by the operator of the other parts, is
    compared to their measurements.

    Works with any operator whose parameters can be split
    (``OperatorParameters.split``); others raise when the solver is made.
    """

    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn,
        solver_params: Optional[LIONParameter] = None,
        operator_params: OperatorParameters = None,
        verbose: bool = True,
        device: torch.device = None,
        seed: Optional[int] = 0,
        deterministic: bool = False,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            operator_params,
            verbose,
            device,
            solver_params=solver_params,
            seed=seed,
            deterministic=deterministic,
        )

        self.model.operator_params = self.operator_params
        self.model._make_operator()
        # The input measurements are split into k interleaved subsets (for CT,
        # of the angles). Each network input averages the reconstructions of p
        # of them (every p-subset is used); the target is the measurements of
        # the remaining ones.
        self.sino_split_count = self.solver_params.sino_split_count  # k
        self.input_split_count = self.solver_params.input_split_count
        k, p = self.sino_split_count, self.input_split_count
        if not 1 <= p < k:
            # at least one split must be left over as the target
            raise ValueError(
                "Sparse2Inverse needs 1 <= input_split_count < sino_split_count, "
                f"got input_split_count={p}, sino_split_count={k}"
            )
        self.split_combinations = self.input_combinations(k, p)
        self._make_sub_operators()

    @staticmethod
    def input_combinations(
        sino_split_count: int, input_split_count: int = 2
    ) -> list[tuple[int, ...]]:
        """All combinations of ``input_split_count`` of the ``sino_split_count``
        splits, e.g. (0, 1), (0, 2), ... for 2 of 4."""
        return list(itertools.combinations(range(sino_split_count), input_split_count))

    def _complement(self, combo) -> tuple[int, ...]:
        """Splits not in ``combo``: their projections are the training target."""
        return tuple(i for i in range(self.sino_split_count) if i not in combo)

    def _make_sub_operators(self):
        n = self.operator_params.split_count
        k = self.sino_split_count
        if k > n:
            raise ValueError(
                f"Can't split {n} measurement units into sino_split_count={k} parts"
            )
        # split i contains every k-th measurement unit (for CT, angle),
        # starting at unit i
        self.subgroup_indices = [list(range(i, n, k)) for i in range(k)]

        self.sub_ops = [
            self.operator_params.split(indices).make_operator()
            for indices in self.subgroup_indices
        ]

        # One differentiable projector per set of target splits (the splits not
        # used as input), keyed by that set.
        self.combo_ops_autograd = {}
        for combo in self.split_combinations:
            target_splits = self._complement(combo)
            indices = sorted(
                i for split in target_splits for i in self.subgroup_indices[split]
            )
            op = self.operator_params.split(indices).make_operator()
            forward, _ = op.autograd()
            self.combo_ops_autograd[target_splits] = forward

    def _calculate_noisy_sub_recons(self, sinos):
        subgroup_recons = {}
        for combo in self.split_combinations:
            temp_recons = []
            for split_idx in combo:
                sub_sino = self.operator_params.split_measurements(
                    sinos, self.subgroup_indices[split_idx]
                )
                sub_recon = self.reconstruct_measurements(
                    sub_sino, self.sub_ops[split_idx]
                )
                temp_recons.append(sub_recon)
            mean_subgroup_recon = torch.mean(torch.stack(temp_recons, dim=1), dim=1)
            subgroup_recons[combo] = mean_subgroup_recon
        return subgroup_recons

    @staticmethod
    def default_parameters() -> LIONParameter:
        params = LIONParameter()
        # number of interleaved angular splits of each input sinogram (k)
        params.sino_split_count = 4
        # number of splits whose reconstructions are averaged to form one network input (p)
        params.input_split_count = 2
        # How measurements are reconstructed, recon_fn(y, op). "default": the
        # operator's default reconstruction (FDK for CT), replaced by that
        # function when the solver is made.
        params.recon_fn = "default"
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
                target_sino = self.operator_params.split_measurements(
                    sinos[b : b + 1], objective_idx
                )
                batch_loss += ((projected_sino - target_sino) ** 2).sum()
                total_pixels += projected_sino.numel()
        batch_loss /= total_pixels
        return batch_loss

    def _forward(self, data):
        # validation and testing evaluate the reconstruction
        return self.reconstruct(data)

    def reconstruct(self, sinos):
        subgroup_recons = self._calculate_noisy_sub_recons(sinos)
        outputs = torch.zeros(
            (sinos.shape[0], *self.op.domain_shape), device=self.device
        )
        for combo, mean_recon in subgroup_recons.items():
            outputs += self.model(mean_recon)
        outputs /= len(subgroup_recons)
        return outputs

    @staticmethod
    def cite(cite_format="bib"):
        """Where Sparse2Inverse comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{gruber2024sparse2inverse,
            title={Sparse2Inverse: Self-supervised inversion of sparse-view CT data},
            author={Gruber, Nadja and Schwab, Johannes and Gizewski, Elke and Haltmeier, Markus},
            journal={arXiv preprint arXiv:2402.16921},
            year={2024}
            }
            """,
            mla="""
            Gruber, Nadja, Johannes Schwab, Elke Gizewski, and Markus Haltmeier. "Sparse2Inverse: Self-supervised inversion of sparse-view CT data." arXiv preprint arXiv:2402.16921 (2024).
            """,
        )
