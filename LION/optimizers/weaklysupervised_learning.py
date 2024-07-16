# numerical imports
import torch
import numpy as np

# Import base class
from LION.optimizers.LIONsolver import LIONsolver

# Parameter class
from LION.utils.parameter import LIONParameter

# standard imports
from tqdm import tqdm
from ts_algorithms import fdk
import wandb



class weaklysupervisedSolver(LIONsolver):
    def __init__(self, model, optimizer, loss_fn, optimizer_params=None, verbose=True):

        super().__init__(model, optimizer, loss_fn, optimizer_params, verbose)

    @staticmethod
    def default_parameters():
        param = LIONParameter()
        param.model = None
        param.optimizer = None
        param.loss_fn = None
        param.device = torch.cuda.current_device()
        param.validation_loader = None
        param.validation_fn = None #torch.nn.MSELoss()
        param.testing_fn = None #torch.nn.MSELoss()
        param.validation_freq = 10
        param.save_folder = None
        param.checkpoint_freq = 10
        param.final_result_fname = None
        param.checkpoint_fname = None
        param.validation_fname = None
        param.epoch = None
        return param

    def mini_batch_step(self, data_marginal_noisy,imagedata_marginal_real):
        """
        This function isresponsible for performing a single mini-batch step of the optimization.
        returns the loss of the mini-batch
        """
        # set model to train
        self.model.train()
        # Zero gradients
        self.optimizer.zero_grad()
        # Compute loss
        # Initialise the noisy images to be the same size as the real images
        imagedata_marginal_noisy = torch.zeros_like(imagedata_marginal_real)
        for i in range(data_marginal_noisy.shape[0]):
            imagedata_marginal_noisy[i] = fdk(self.model.op, data_marginal_noisy[i])
        # imagedata_marginal_noisy = fdk(self.model.op, data_marginal_noisy)
        self.loss = self.loss_fn(self.model, imagedata_marginal_noisy,imagedata_marginal_real)
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
        pbar = tqdm(enumerate(self.train_loader))
        for index, (data, target) in pbar:
            single_loss = self.mini_batch_step(
                data.to(self.device), target.to(self.device)
            )
            epoch_loss += single_loss
            pbar.set_postfix({'loss': single_loss})
            wandb.log({"train_loss": single_loss})
        return epoch_loss / len(self.train_loader)

    def validate(self):
        """
        This function is responsible for performing a single validation set of the optimization.
        returns the average loss of the validation set this epoch.
        """
        status = self.model.training
        # self.model.eval()
        validation_loss = 0.0
        if self.verbose:
            print("Validating...")
        for index, (data, target) in tqdm(
            enumerate(self.validation_loader), disable=True
        ):
            output = self.model.output(data.to(self.device),truth =target.to(self.device))
            val_step = self.validation_fn(output, target.to(self.device))
            validation_loss += val_step

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
            if (
                self.validation_fname is not None
                and self.validation_loss[epoch] < self.validation_loss.min()
            ):
                self.save_validation(self.validation_fname, epoch)

        elif self.verbose:
            print(f"Epoch {epoch} - Training loss: {self.train_loss[epoch]}")
        elif self.validation_freq is not None:
            self.validation_loss[epoch] = self.validate()
            wandb.log({"validation_loss": self.validation_loss[epoch]})

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
                
                
    def test(self):
        """
        This function performs a testing step
        """
        # self.model.eval()
        test_loss = np.zeros(len(self.test_loader))
        print(test_loss,test_loss.shape)
        # with torch.no_grad():
        for index, (data, target) in enumerate(self.test_loader):
            output = self.model.output(data.to(self.device))
            test_loss[index] = self.testing_fn(output, target.to(self.device))
            

        if self.verbose:
            print(
                f"Testing loss: {test_loss.mean()} - Testing loss std: {test_loss.std()}"
            )
        return test_loss
