# numerical imports
import torch
import numpy as np

# Import base class
from LION.optimizers.LIONsolver import LIONsolver

# Parameter class
from LION.utils.parameter import LIONParameter

# standard imports
from tqdm import tqdm


class supervisedSolver(LIONsolver):
    def __init__(self, model, optimizer, loss_fn, optimizer_params=None, verbose=True):
        # this does the necessary checks
        super().__init__(model, optimizer, loss_fn)
        # Initialize some parameters

        if optimizer_params is None:
            optimizer_params = supervisedSolver.default_parameters()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = optimizer_params.device
        self.validation_loader = optimizer_params.validation_loader
        self.validation_fn = optimizer_params.validation_fn
        self.validation_freq = optimizer_params.validation_freq
        self.save_folder = optimizer_params.save_folder
        self.checkpoint_freq = optimizer_params.checkpoint_freq
        self.final_result_fname = optimizer_params.final_result_fname
        self.checkpoint_fname = optimizer_params.checkpoint_fname
        self.validation_fname = optimizer_params.validation_fname
        self.verbose = verbose

    @staticmethod
    def default_parameters():
        param = LIONParameter()
        param.model = None
        param.optimizer = None
        param.loss_fn = None
        param.device = torch.cuda.current_device()
        param.validation_loader = None
        param.validation_fn = None
        param.validation_freq = 10
        param.save_folder = None
        param.checkpoint_freq = 10
        param.final_result_fname = None
        param.checkpoint_fname = None
        param.validation_fname = None
        param.epoch = None
        return param

    @staticmethod
    def adam_default_parameters():
        param = supervisedSolver.default_parameters()
        param.optimizer = torch.optim.Adam
        return param

    def mini_batch_step(self, data, target):
        """
        This function isresponsible for performing a single mini-batch step of the optimization.
        returns the loss of the mini-batch
        """
        # set model to train
        self.model.train()
        # Zero gradients
        self.optimizer.zero_grad()
        # Forward pass
        output = self.model(data)
        # Compute loss
        self.loss = self.loss_fn(output, target)
        # Update optimizer and model
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
        for index, (data, target) in enumerate(self.train_loader):
            epoch_loss += self.mini_batch_step(data, target)
        return epoch_loss / len(self.train_loader)

    def validate(self):
        """
        This function is responsible for performing a single validation set of the optimization.
        returns the average loss of the validation set this epoch.
        """
        status = self.model.training
        self.model.eval()
        validation_loss = 0.0
        if self.verbose:
            print("Validating...")
        for index, (data, target) in tqdm(
            enumerate(self.validation_loader), disable=not self.verbose
        ):
            with torch.no_grad():
                output = self.model(data)
                validation_loss = self.validation_fn(output, target)

        # return to train if it was in train
        if status:
            self.model.train()
        return validation_loss / len(self.validation_loader)

    def epoch_step(self, epoch):
        """
        This function is responsible for performing a single epoch of the optimization.
        """
        self.train_loss[epoch] = self.train_step()
        if (epoch + 1) % self.validation_freq == 0:
            self.validation_loss[epoch] = self.validate()
            if self.verbose:
                print(
                    f"Epoch {epoch} - Training loss: {self.train_loss[epoch]} - Validation loss: {self.validation_loss[epoch]}"
                )
        elif self.verbose:
            print(f"Epoch {epoch} - Training loss: {self.train_loss[epoch]}")
        elif self.validation_freq is not None:
            self.validation_loss[epoch] = self.validate()

        pass

    def train(self, n_epochs):
        """
        This function is responsible for performing the optimization.
        """
        assert n_epochs > 0, "Number of epochs must be a positive integer"
        # Make sure all parameters are set
        self.check_complete()

        self.epochs = n_epochs
        self.train_loss = np.zeros(self.epochs)
        if self.validation_fn is not None:
            self.validation_loss = np.zeros(self.epochs)

        # train loop
        for epoch in tqdm(range(self.epochs)):
            self.epoch_step(epoch)
            if (epoch + 1) % self.checkpoint_freq == 0:
                self.save_checkpoint(epoch)
