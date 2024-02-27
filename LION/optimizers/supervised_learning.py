# standard imports
import torch

# Import base class
from LION.optimizers.LIONsolver import LIONsolver

# Parameter class
from LION.utils.parameter import LIONParameter


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
        return param

    @staticmethod
    def adam_default_parameters():
        param = supervisedSolver.default_parameters()
        param.optimizer = torch.optim.Adam
        return param

    def mini_batch_step(self, data, target):
        # Set model to training mode
        self.model.train()
        # Zero gradients
        self.optimizer.zero_grad()
        # Forward pass
        output = self.model(data)
        # Compute loss
        loss = self.loss_fn(output, target)
        # Backward pass
        loss.backward()
        # Update weights
        self.optimizer.step()
        return loss.item()
