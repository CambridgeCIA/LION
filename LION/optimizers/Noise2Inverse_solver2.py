import pathlib
from typing import Callable
import numpy as np
from tqdm import tqdm
from LION.CTtools.ct_geometry import Geometry
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import LIONmodel
import torch
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
import tomosipo as ts
import LION.CTtools.ct_utils as ct


class Noise2InverseParams(SolverParams):
    def __init__(
        self,
        sino_split_count: int,
        recon_fn: Callable[[torch.Tensor, ts.Operator.Operator], torch.Tensor],
        cali_J: list[list[int]],
    ):
        super().__init__()

        self.sino_split_count = sino_split_count
        self.recon_fn = recon_fn
        self.cali_J = cali_J


class Noise2InverseSolver(LIONsolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        solver_params: Noise2InverseParams,
        verbose: bool,
        geo: Geometry,
        save_folder: str | pathlib.Path,
        final_result_fname: str,
        device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}"),
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            save_folder,
            final_result_fname,
            solver_params,
            verbose,
            device,
        )
        self.sino_split_count = solver_params.sino_split_count
        if geo is None:
            raise ValueError(
                "Argument 'geo' is None. Noise2InverseSolver requires a geometry"
            )
        self.geo = geo  # don't necessarily need to keep this around as is, but thought we might as well
        self.recon_fn = solver_params.recon_fn
        self.cali_J = np.array(solver_params.cali_J)
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
        angles = self.geo.angles.copy()
        assert (
            len(angles) % self.sino_split_count == 0
        ), f"Cannot construct {self.sino_split_count} sinogram splits from {len(angles)} view angles. Ensure that sino_split_count divides #view angles"
        for k in range(self.sino_split_count):
            self.geo.angles = angles[k :: self.sino_split_count]
            sub_op = ct.make_operator(self.geo)
            ops.append(sub_op)
        # restore self.geo.angles
        self.geo.angles = angles
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
                    size=(sinos.shape[0], *sub_recon_j.shape), device=self.device
                )
            bad_recons[:, j, :, :, :] = sub_recon_j
        assert bad_recons is not None
        return bad_recons

    @staticmethod
    def default_parameters() -> Noise2InverseParams:
        sino_split_count = 10
        recon_fn = fdk
        cali_J = Noise2InverseSolver.X_one_strategy(sino_split_count)
        return Noise2InverseParams(
            sino_split_count,
            recon_fn,
            cali_J,
        )

    def mini_batch_step(self, sinos):
        # sinos batch of sinos
        noisy_sub_recons = self._calculate_noisy_sub_recons(sinos)
        # b, split, c, w, h

        self.model.train()
        self.optimizer.zero_grad()

        # almost certain this can be made more efficient
        # use all the Js, this is different from Ander's
        loss = torch.zeros(len(self.cali_J), device=self.device)
        for i, J in enumerate(self.cali_J):
            # fix indexing J's are 1 indexed for user convenience
            J_zero_indexing = list(map(lambda i: i - 1, J))
            J_c = [i for i in np.arange(self.sino_split_count) if i not in J_zero_indexing]

            # calculate mean sub_recons
            jnsr = noisy_sub_recons[:, J_zero_indexing, :, :, :]
            jcnsr = noisy_sub_recons[:, J_c, :, :, :]
            mean_target_recons = torch.mean(jnsr, dim=1)
            mean_input_recons = torch.mean(jcnsr, dim=1)

            output = self.model(mean_input_recons)
            loss[i] = self.loss_fn(output, mean_target_recons)

        self.loss = loss.sum()
        self.loss.backward()
        self.optimizer.step()

        return self.loss.item()

    def train_step(self):
        """
        This function is responsible for performing a single tranining set epoch of the optimization.
        returns the average loss of the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        # needs modifying, need some sort of guarantee as to what the dataset looks like.
        # only makes sense to use noise2inverse if we only have the sinos.
        for index, (sino, _) in enumerate(tqdm(self.train_loader)):
            epoch_loss += self.mini_batch_step(sino)
        return epoch_loss / len(self.train_loader)

    # No validation in Noise2Inverse (is this right?)
    def validate(self):
        return 0

    def epoch_step(self, epoch):
        """
        This function is responsible for performing a single epoch of the optimization.
        """
        self.train_loss[epoch] = self.train_step()

    def train(self, n_epochs):
        """
        This function is responsible for performing the optimization.
        """
        assert n_epochs > 0, "Number of epochs must be a positive integer"
        # Make sure all parameters are set
        self.check_complete()

        self.epochs = n_epochs
        self.train_loss = np.zeros(self.epochs)

        # train loop
        for epoch in tqdm(range(self.epochs)):
            print(f"Training epoch {epoch + 1}")
            self.epoch_step(epoch)
            if (epoch + 1) % self.checkpoint_freq == 0:
                self.save_checkpoint(epoch)
