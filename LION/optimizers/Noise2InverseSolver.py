from typing import Callable, Optional
import warnings
import numpy as np
from tqdm import tqdm
from LION.CTtools.ct_geometry import Geometry
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import LIONmodel
import torch
from torch.optim.optimizer import Optimizer
from LION.optimizers.LIONsolver import LIONsolver
from LION.utils.parameter import LIONParameter
import tomosipo as ts
import LION.CTtools.ct_utils as ct


class Noise2InverseSolver(LIONsolver):
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

        self.sino_split_count = self.solver_params.sino_split_count
        self.recon_fn = self.solver_params.recon_fn
        self.cali_J = np.array(self.solver_params.cali_J)
        self.sub_ops = self._make_sub_operators()

    @classmethod
    def X_one_strategy(cls, sino_split_count) -> list[list[int]]:
        return [[i + 1] for i in range(sino_split_count)]

    @classmethod
    def one_X_strategy(cls, sino_split_count) -> frozenset[frozenset[int]]:
        raise NotImplementedError("Sorry, not implemented this yet!")

    def _make_sub_operators(self) -> list[ts.Operator.Operator]:
        ops = []
        # maintain a copy of the original angles to restore later
        angles = self.geometry.angles.copy()
        assert (
            len(angles) % self.sino_split_count == 0
        ), f"Cannot construct {self.sino_split_count} sinogram splits from {len(angles)} view angles. Ensure that sino_split_count divides #view angles"
        for k in range(self.sino_split_count):
            self.geometry.angles = angles[k :: self.sino_split_count]
            sub_op = ct.make_operator(self.geometry)
            ops.append(sub_op)
        # restore self.geometry.angles
        self.geometry.angles = angles
        return ops

    def _calculate_noisy_sub_recons(self, sinos):
        # sinos is batched
        bad_recons = None
        for j in range(self.sino_split_count):
            sub_sino_j = sinos[
                :, :, j :: self.sino_split_count, :
            ]  # is this right? What is sinos.shape? B, C, W, H? Yes, it is.
            # expect recon_fn to be batched
            sub_recon_j = self.recon_fn(sub_sino_j, self.sub_ops[j])
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
        params.recon_fn = fdk
        params.cali_J = Noise2InverseSolver.X_one_strategy(params.sino_split_count)
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

    # No validation in Noise2Inverse (is this right?)
    def validate(self):
        return 0

    def test(self):
        """
        This function performs a testing step
        """
        if self.check_testing_ready() != 0:
            warnings.warn("Solver not setup for testing. Please call set_testing")
            return np.array([])
        assert self.test_loader is not None
        assert self.testing_fn is not None

        was_training = self.model.training
        self.model.eval()

        # do we want to be able to use this on a trained model? Surely yes?
        with torch.no_grad():
            test_loss = np.array([])
            for sinos, targets in tqdm(self.test_loader):
                outputs = self.reconstruct(sinos)
                test_loss = np.append(test_loss, self.testing_fn(targets, outputs))

        if self.verbose:
            print(
                f"Testing loss: {test_loss.mean()} - Testing loss std: {test_loss.std()}"
            )

        if was_training:
            self.model.train()

        return test_loss

    def reconstruct(self, sinos):
        noisy_sub_recons = self._calculate_noisy_sub_recons(sinos)  # b, split, c, w, h

        outputs = torch.zeros(
            (sinos.shape[0], *self.geometry.image_shape), device=self.device
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
    def cite(cite_format="MLA"):

        if cite_format == "MLA":
            print(
                "Hendriksen, Allard Adriaan, Daniël Maria Pelt, and K. Joost Batenburg"
            )
            print(
                ' "Noise2inverse: Self-supervised deep convolutional denoising for tomography."'
            )
            print("\x1B[3m  IEEE Transactions on Computational Imaging \x1B[0m")
            print("6 (2020): 1320-1335.")
        elif cite_format == "bib":
            string = """
            @article{hendriksen2020noise2inverse,
            title={Noise2inverse: Self-supervised deep convolutional denoising for tomography},
            author={Hendriksen, Allard Adriaan and Pelt, Dani{\"e}l Maria and Batenburg, K Joost},
            journal={IEEE Transactions on Computational Imaging},
            volume={6},
            pages={1320--1335},
            year={2020},
            publisher={IEEE}
            }"""
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )
