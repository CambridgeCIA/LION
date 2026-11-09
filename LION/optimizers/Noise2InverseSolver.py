# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  :
# =============================================================================
from typing import Optional
import numpy as np
from LION.models.LIONmodel import LIONmodel
import torch
from torch.optim.optimizer import Optimizer
from LION.optimizers.LIONsolver import LIONsolver
from LION.utils.parameter import LIONParameter, OperatorParameters
from LION.utils.citation import citation


class Noise2InverseSolver(LIONsolver):
    """Noise2Inverse: the measurements are split into ``sino_split_count``
    interleaved parts (for CT, every k-th angle), each reconstructed with
    ``recon_fn`` (by default the operator's default reconstruction, FDK for
    CT), and the model learns to map the mean
    of some parts' reconstructions to the mean of the others'.

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

        self.sino_split_count = self.solver_params.sino_split_count
        cali_J = getattr(self.solver_params, "cali_J", None)
        if cali_J is None:  # the X:1 strategy for the split count used
            cali_J = self.X_one_strategy(self.sino_split_count)
        if any(j < 1 or j > self.sino_split_count for J in cali_J for j in J):
            raise ValueError(
                f"cali_J {cali_J} has splits outside 1..sino_split_count "
                f"({self.sino_split_count}); J's are 1-indexed"
            )
        self.cali_J = np.array(cali_J)
        self.sub_ops = self._make_sub_operators()

    @classmethod
    def X_one_strategy(cls, sino_split_count) -> list[list[int]]:
        return [[i + 1] for i in range(sino_split_count)]

    @classmethod
    def one_X_strategy(cls, sino_split_count) -> frozenset[frozenset[int]]:
        raise NotImplementedError("Sorry, not implemented this yet!")

    def _make_sub_operators(self) -> list:
        """One operator per split: split j has every k-th measurement unit
        (for CT, angle), starting at unit j."""
        n, k = self.operator_params.split_count, self.sino_split_count
        if k > n:
            raise ValueError(
                f"Can't split {n} measurement units into sino_split_count={k} parts"
            )
        self.split_indices = [list(range(j, n, k)) for j in range(k)]
        return [
            self.operator_params.split(indices).make_operator()
            for indices in self.split_indices
        ]

    def _calculate_noisy_sub_recons(self, sinos):
        # sinos is batched
        bad_recons = None
        for j in range(self.sino_split_count):
            sub_sino_j = self.operator_params.split_measurements(
                sinos, self.split_indices[j]
            )
            # expect recon_fn to be batched
            sub_recon_j = self.reconstruct_measurements(sub_sino_j, self.sub_ops[j])
            if bad_recons is None:
                bad_recons = torch.zeros(
                    size=(
                        sinos.shape[0],
                        self.sino_split_count,
                        *sub_recon_j.shape[1:],
                    ),
                    device=self.device,
                )
            bad_recons[:, j, :, :, :] = sub_recon_j  # b, s, c, w, h
        assert bad_recons is not None
        return bad_recons

    @staticmethod
    def default_parameters() -> LIONParameter:
        params = LIONParameter()
        params.sino_split_count = 4
        # The J sets (1-indexed splits) used as targets. None: the X:1 strategy
        # for sino_split_count, i.e. [[1], [2], ..., [sino_split_count]],
        # computed when the solver is made, so changing sino_split_count after
        # default_parameters() changes the J sets too.
        params.cali_J = None
        # How measurements are reconstructed, recon_fn(y, op). "default": the
        # operator's default reconstruction (FDK for CT), replaced by that
        # function when the solver is made.
        params.recon_fn = "default"
        return params

    def mini_batch_step(self, sinos, targets):
        # sinos batch of sinos
        noisy_sub_recons = self._calculate_noisy_sub_recons(sinos)
        # b, split, c, w, h

        # almost certain this can be made more efficient
        # use all the Js, this is different from Ander's
        loss = torch.zeros(len(self.cali_J), device=self.device)
        for i, J in enumerate(self.cali_J):
            # fix indexing J's are 1 indexed for user convenience
            # maybe something to change
            J_zero_indexing = list(map(lambda i: i - 1, J))
            J_c = [
                i for i in np.arange(self.sino_split_count) if i not in J_zero_indexing
            ]

            # calculate mean sub_recons
            jnsr = noisy_sub_recons[:, J_zero_indexing, :, :, :]
            jcnsr = noisy_sub_recons[:, J_c, :, :, :]
            mean_target_recons = torch.mean(jnsr, dim=1)
            mean_input_recons = torch.mean(jcnsr, dim=1)

            output = self.model(mean_input_recons)
            loss[i] = self.loss_fn(output, mean_target_recons)

        return loss.sum() / len(self.cali_J)

    def _forward(self, data):
        # validation and testing evaluate the reconstruction
        return self.reconstruct(data)

    def reconstruct(self, sinos):
        noisy_sub_recons = self._calculate_noisy_sub_recons(sinos)  # b, split, c, w, h

        outputs = torch.zeros(
            (sinos.shape[0], *self.op.domain_shape), device=self.device
        )

        for _, J in enumerate(self.cali_J):
            # fix indexing; J's are 1 indexed for user convenience
            J_zero_indexing = list(map(lambda n: n - 1, J))
            J_c = [
                n for n in np.arange(self.sino_split_count) if n not in J_zero_indexing
            ]

            # calculate mean sub_recons
            jcnsr = noisy_sub_recons[:, J_c, :, :, :]
            mean_input_recons = torch.mean(jcnsr, dim=1)

            # pump it through NN
            outputs += self.model(mean_input_recons)
        outputs /= len(self.cali_J)
        return outputs

    @staticmethod
    def cite(cite_format="bib"):
        """Where Noise2InverseSolver comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{hendriksen2020noise2inverse,
            title={Noise2inverse: Self-supervised deep convolutional denoising for tomography},
            author={Hendriksen, Allard Adriaan and Pelt, Dani{\\"e}l Maria and Batenburg, K Joost},
            journal={IEEE Transactions on Computational Imaging},
            volume={6},
            pages={1320--1335},
            year={2020},
            publisher={IEEE}
            }
            """,
            mla="""
            Hendriksen, Allard Adriaan, Daniël Maria Pelt, and K. Joost Batenburg
             "Noise2inverse: Self-supervised deep convolutional denoising for tomography."
              IEEE Transactions on Computational Imaging 
            6 (2020): 1320-1335.
            """,
        )
