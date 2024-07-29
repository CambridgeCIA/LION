from typing import Callable, Optional
import warnings
import numpy as np
from tqdm import tqdm
from LION.CTtools.ct_geometry import Geometry
from LION.classical_algorithms.fdk import fdk
from LION.exceptions.exceptions import NoDataException
from LION.models.LIONmodel import LIONmodel
import torch
from torch.optim.optimizer import Optimizer
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
import tomosipo as ts
import LION.CTtools.ct_utils as ct
from LION.optimizers.losses.LIONloss import LIONtrainingLoss


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
        optimizer: Optimizer,
        loss_fn: LIONtrainingLoss | torch.nn.Module,
        solver_params: Optional[Noise2InverseParams],
        verbose: bool,
        geo: Geometry,
        device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}"),
    ) -> None:
        super().__init__(
            model, optimizer, loss_fn, geo, verbose, device, solver_params=solver_params
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
    def default_parameters() -> Noise2InverseParams:
        sino_split_count = 4
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

        self.optimizer.zero_grad()

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

            if self.do_normalize:
                mean_input_recons = self.normalize(mean_input_recons)
                mean_target_recons = self.normalize(mean_target_recons)

            output = self.model(mean_input_recons)
            loss[i] = self.loss_fn(output, mean_target_recons)

        self.loss = loss.sum()
        self.loss.backward()
        self.optimizer.step()

        return self.loss.item() / len(self.cali_J)

    def train_step(self):
        """
        This function is responsible for performing a single tranining set epoch of the optimization.
        returns the average loss of the epoch
        """
        if self.train_loader is None:
            raise NoDataException(
                "Training dataloader not set: Please call set_training"
            )
        self.model.train()
        epoch_loss = 0.0
        # needs modifying, need some sort of guarantee as to what the dataset looks like.
        # only makes sense to use noise2inverse if we only have the sinos.
        for index, (sino, _) in enumerate(tqdm(self.train_loader)):
            epoch_loss += self.mini_batch_step(sino)
        return epoch_loss / (len(self.train_loader) * sino.shape[0])

    # No validation in Noise2Inverse (is this right?)
    def validate(self):
        return 0

    def epoch_step(self, epoch):
        """
        This function is responsible for performing a single epoch of the optimization.
        """
        self.train_loss[epoch] = self.train_step()
        print(f"Training Loss: {self.train_loss[epoch]}")

    def train(self, n_epochs):
        """
        This function is responsible for performing the optimization.
        Runs n_epochs additional epochs.
        """
        assert n_epochs > 0, "Number of epochs must be a positive integer"
        # Make sure all parameters are set
        self.check_complete()

        if self.do_load_checkpoint:
            print("Loading checkpoint...")
            self.current_epoch = self.load_checkpoint()
            self.train_loss = np.append(self.train_loss, np.zeros((n_epochs)))
        else:
            self.train_loss = np.zeros(n_epochs)

        self.model.train()
        # train loop
        final_total_epochs = self.current_epoch + n_epochs
        while self.current_epoch < final_total_epochs:
            print(f"Training epoch {self.current_epoch + 1}")
            self.epoch_step(self.current_epoch)

            if (self.current_epoch + 1) % self.checkpoint_freq == 0:
                if self.verbose:
                    print(f"Checkpointing at epoch {self.current_epoch}")
                self.save_checkpoint(self.current_epoch)

            self.current_epoch += 1

    def test(self):
        """
        This function performs a testing step
        """
        if self.check_testing_ready() != 0:
            warnings.warn("Solver not setup for testing. Please call set_testing")
            return np.array([])

        was_training = self.model.training
        self.model.eval()

        # do we want to be able to use this on a trained model? Surely yes?
        with torch.no_grad():
            test_loss = np.array([])
            for sinos, targets in tqdm(self.test_loader):
                outputs = self.process(sinos)
                test_loss = np.append(test_loss, self.testing_fn(targets, outputs))

        if self.verbose:
            print(
                f"Testing loss: {test_loss.mean()} - Testing loss std: {test_loss.std()}"
            )

        if was_training:
            self.model.train()

        return test_loss

    # not convinced by this name
    def process(self, sinos):
        noisy_sub_recons = self._calculate_noisy_sub_recons(sinos)  # b, split, c, w, h

        outputs = torch.zeros(
            (sinos.shape[0], *self.geo.image_shape), device=self.device
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
