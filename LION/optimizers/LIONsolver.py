# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================

#%% This is a base class for solvers/trainers that gives you helpful functions.
#
# It definest a bunch of auxiliary functions
#

# You will want to import LIONParameter, as all models must save and use Parameters.
from LION.utils.parameter import LIONParameter

# Lionmodels
from LION.models.LIONmodel import LIONmodel

# some numerical standard imports, e.g.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# imports related to class organization
from abc import ABC, abstractmethod, ABCMeta

# general imports
import warnings


class LIONsolver(ABC):
    def __init__(self, model, optimizer, loss_fn) -> None:
        super().__init__()
        __metaclass__ = ABCMeta  # make class abstract.

        assert isinstance(model, LIONmodel), "model must be a LIONmodel"
        assert isinstance(
            optimizer, torch.optim.Optimizer
        ), "optimizer must be a torch optimizer"
        assert isinstance(loss_fn, nn.Module), "loss_fn must be a torch loss function"
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def __check_complete(self, error=True, autofill=True):
        """
        This function checks if the solver is complete, i.e. if all the necessary parameters are set to start traning.
        """

        return_code = 0

        # Set if we need to complain
        if hasattr(self, "verbose"):
            verbose = self.verbose
        else:
            verbose = True

        # Test 1: is the device set? if not, set it if aitofill is True
        return_code = self.__check_attribute(
            "device",
            type=torch.device,
            error=error,
            autofill=autofill,
            verbose=verbose,
            default=torch.cuda.current_device(),
        )

        # Test 2: is the model set? if not, raise error or warn
        return_code = self.__check_attribute(
            "model", type=LIONmodel, error=error, autofill=False, verbose=verbose
        )

        # Test 3: is the optimizer set? if not, raise error or warn
        return_code = self.__check_attribute(
            "optimizer",
            type=torch.optim.Optimizer,
            error=error,
            autofill=False,
            verbose=verbose,
        )

        # Test 4: is the loss_fn set? if not, raise error or warn
        return_code = self.__check_attribute(
            "loss_fn", type=nn.Module, error=error, autofill=False, verbose=verbose
        )

        # Test 5: is the testing function set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "testing_fn", type=callable, error=error, autofill=autofill, verbose=verbose
        )

        if autofill and (
            not hasattr(self, "validation_loader") or self.validation_loader is None
        ):
            self.validation_loader = None
            warnings.warn(
                "Validation loader not set. It is recommended you use a validation set"
            )

        if not hasattr(self, "testing_fn") or self.testing_fn is None:
            self.testing_fn = None
            warnings.warn("Testing function not set. The testing script may fail")

        self.validation_fn = optimizer_params.validation_fn
        self.validation_freq = optimizer_params.validation_freq
        self.save_folder = optimizer_params.save_folder
        self.checkpoint_freq = optimizer_params.checkpoint_freq
        self.final_result_fname = optimizer_params.final_result_fname
        self.checkpoint_fname = optimizer_params.checkpoint_fname
        self.validation_fname = optimizer_params.validation_fname
        self.verbose = verbose

        return return_code

    def __check_attribute(
        self, attr, type=None, error=True, autofill=True, verbose=True, default=None
    ):
        """
        This function checks if an attribute exists, and sets it if needed
        """
        assert isinstance(attr, str), "attr must be a string"
        if autofill and (not hasattr(self, attr) or getattr(self, attr) is None):
            setattr(self, attr, default)
        else:
            if error:
                raise ValueError(f"Attribute {attr} not set")
            elif verbose:
                warnings.warn(f"Attribute {attr} not set")
                return 1
        if type(getattr(self, attr)) is not type:
            if error:
                raise ValueError(f"Attribute {attr} is not of type {type}")
            elif verbose:
                warnings.warn(f"Attribute {attr} is not of type {type}")
                return 2
        return 0

    @abstractmethod
    def mini_batch_step(self):
        """
        This function should perform a single step of the optimization
        """
        pass

    @abstractmethod
    def epoch_step(self):
        """
        This function should perform a single epoch of the optimization
        """
        pass

    @abstractmethod
    def validate(self):
        """
        This function should perform a validation step
        """
        pass
