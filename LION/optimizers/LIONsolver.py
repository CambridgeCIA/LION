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
from torch.utils.data import DataLoader

# imports related to class organization
from abc import ABC, abstractmethod, ABCMeta

# general imports
import warnings
import pathlib


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

    def check_complete(self, error=True, autofill=True):
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
            expected_type=torch.device,
            error=False,
            autofill=autofill,
            verbose=verbose,
            default=torch.device(torch.cuda.current_device()),
        )

        # Test 2: is the model set? if not, raise error or warn
        return_code = self.__check_attribute(
            "model",
            expected_type=LIONmodel,
            error=error,
            autofill=False,
            verbose=verbose,
        )

        # Test 3: is the optimizer set? if not, raise error or warn
        return_code = self.__check_attribute(
            "optimizer",
            expected_type=torch.optim.Optimizer,
            error=error,
            autofill=False,
            verbose=verbose,
        )

        # Test 4: is the loss_fn set? if not, raise error or warn
        return_code = self.__check_attribute(
            "loss_fn",
            expected_type=nn.Module,
            error=error,
            autofill=False,
            verbose=verbose,
        )

        # Test 5: is the testing loader set? if not, raise error or warn
        return_code = self.__check_attribute(
            "test_loader",
            expected_type=DataLoader,
            error=error,
            autofill=False,
            verbose=verbose,
        )
        # Test 6: is the testing function set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "testing_fn",
            expected_type=callable,
            error=error,
            autofill=False,
            verbose=verbose,
        )

        # Test 7: is the training loader set? if not, raise error or warn
        return_code = self.__check_attribute(
            "train_loader",
            expected_type=DataLoader,
            error=error,
            autofill=False,
            verbose=verbose,
        )
        # Test 8: is the validation loader set? if not, raise error or warn
        return_code = self.__check_attribute(
            "validation_loader",
            expected_type=DataLoader,
            error=False,
            autofill=False,
            verbose=True,
        )

        # Test 9: is the validation function set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "validation_fn",
            expected_type=callable,
            error=False,
            autofill=True,
            verbose=verbose,
            default=self.loss_fn,
        )
        # Test 10: is the validation frequency set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "validation_freq",
            expected_type=int,
            error=False,
            autofill=autofill,
            verbose=verbose,
            default=10,
        )
        # Test 11: is the save folder set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "save_folder",
            expected_type=pathlib.Path,
            error=error,
            autofill=False,
            verbose=verbose,
        )
        # Test 12: is the final result filename set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "final_result_fname",
            expected_type=pathlib.Path,
            error=error,
            autofill=True,
            verbose=verbose,
            default=self.save_folder.joinpath("final_result.pt"),
        )
        # Test 13: is the checkpoint frequency filename set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "checkpoint_fname",
            expected_type=pathlib.Path,
            error=False,
            autofill=True,
            verbose=False,
            default=self.final_result_fname.joinpath("_checkpoint_*.pt"),
        )
        # Test 14: is the checkpoint frequency set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "checkpoint_freq",
            expected_type=int,
            error=False,
            autofill=autofill,
            verbose=verbose,
            default=10,
        )
        self.verbose = verbose

        return return_code

    def __check_attribute(
        self,
        attr,
        expected_type=None,
        error=True,
        autofill=True,
        verbose=True,
        default=None,
    ):
        """
        This function checks if an attribute exists, and sets it if needed
        """
        assert isinstance(attr, str), "attr must be a string"

        if not hasattr(self, attr) or getattr(self, attr) is None:
            if autofill:
                setattr(self, attr, default)
            else:
                if error:
                    raise ValueError(f"Attribute {attr} not set")
                elif verbose:
                    warnings.warn(f"Attribute {attr} not set")
                    return 1

        # if the type we want to check against is a function, we need to treat it differently
        if expected_type is callable:
            if not callable(getattr(self, attr)):
                if error:
                    raise ValueError(f"Attribute {attr} is not callable")
                elif verbose:
                    warnings.warn(f"Attribute {attr} is not callable")
                    return 2
        # just standrad type chekcking, error or warn, depends of settings
        elif isinstance(getattr(self, attr), type):
            if error:
                raise ValueError(
                    f"Attribute {attr} is not of type {expected_type}, its {type(getattr(self, attr))}"
                )
            elif verbose:
                warnings.warn(
                    f"Attribute {attr} is not of type {expected_type}, its {type(getattr(self, attr))}"
                )
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
