# numerical imports
import torch
import numpy as np
import torch.nn as nn

# Import base class
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
from LION.exceptions.exceptions import LIONSolverException, NoDataException
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.optimizers.LIONsolver import LIONsolver, SolverParams

# Parameter class
from LION.utils.parameter import LIONParameter

# standard imports
from tqdm import tqdm


class WeaklySupervisedSolver(LIONsolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        verbose: bool,
        geo: Geometry,
        model_regularization=None,
        device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}"),
    ):
        super().__init__(
            model,
            optimizer,
            loss_fn,
            geo,
            verbose=verbose,
            device=device,
            model_regularization=model_regularization,
            solver_params=SolverParams(),
        )
        self.op = make_operator(self.geo)

    def mini_batch_step(self, data, ground_truth):
        """
        This function isresponsible for performing a single mini-batch step of the optimization.
        returns the loss of the mini-batch
        """
        # set model to train
        self.model.train()
        # Zero gradients
        self.optimizer.zero_grad()
        if self.model.model_parameters.model_input_type == ModelInputType.IMAGE:
            data = fdk(data, self.op)

        # Forward pass
        output = self.model(data)
        self.loss = self.loss_fn(
            output, ground_truth
        )
        # Update optimizer and model
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

    def train_step(self):
        """
        This function is responsible for performing the optimization.
        Runs n_epochs additional epochs.
        """
        if self.train_loader is None:
            raise NoDataException(
                "Training dataloader not set: Please call set_training"
            )
        self.model.train()
        epoch_loss = 0.0
        for _, (data, target) in enumerate(tqdm(self.train_loader)):
            epoch_loss += self.mini_batch_step(
                data.to(self.device), target.to(self.device)
            )
        return epoch_loss / len(self.train_loader.dataset)

    def validate(self):
        """
        This function is responsible for performing a single validation set of the optimization.
        returns the average loss of the validation set this epoch.
        """
        if self.check_validation_ready() != 0:
            raise LIONSolverException(
                "Solver not ready for validation. Please call set_validation."
            )

        # these always pass if the above does, this is just to placate static type checker
        assert self.validation_loader is not None
        assert self.validation_fn is not None

        status = self.model.training
        self.model.eval()

        with torch.no_grad():
            validation_loss = np.array([])
            for data, targets in tqdm(self.test_loader):
                if self.model.model_parameters.model_input_type == ModelInputType.SINOGRAM:
                    data = fdk(data, self.op)
                outputs = self.model(data)
                validation_loss = np.append(validation_loss, self.testing_fn(targets, outputs))

        if self.verbose:
            print(
                f"Testing loss: {validation_loss.mean()} - Testing loss std: {validation_loss.std()}"
            )

        # return to train if it was in train
        if status:
            self.model.train()

        return np.mean(validation_loss)

    def epoch_step(self, epoch):
        """
        This function is responsible for performing a single epoch of the optimization.
        """
        self.train_loss[epoch] = self.train_step()
        # actually make sure we're doing validation
        if (epoch + 1) % self.validation_freq == 0 and self.validation_loss is not None:
            self.validation_loss[epoch] = self.validate()
            if self.verbose:
                print(
                    f"Epoch {epoch+1} - Training loss: {self.train_loss[epoch]} - Validation loss: {self.validation_loss[epoch]}"
                )

            if self.validation_fname is not None and self.validation_loss[
                epoch
            ] <= np.min(self.validation_loss[np.nonzero(self.validation_loss)]):
                self.save_validation(epoch)
        elif self.verbose:
            print(f"Epoch {epoch+1} - Training loss: {self.train_loss[epoch]}")
        elif self.validation_freq is not None and self.validation_loss is not None:
            self.validation_loss[epoch] = self.validate()

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
        # self.model.eval()
        test_loss = np.zeros(len(self.test_loader))
        print(test_loss, test_loss.shape)
        # with torch.no_grad():
        for index, (data, target) in enumerate(self.test_loader):
            output = self.model(data.to(self.device))
            test_loss[index] = self.testing_fn(output, target.to(self.device))

        if self.verbose:
            print(
                f"Testing loss: {test_loss.mean()} - Testing loss std: {test_loss.std()}"
            )
        return test_loss