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

# Some utils
from LION.utils.utils import custom_format_warning

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

        self.metadata = LIONParameter()
        self.dataset_param = LIONParameter()

    def set_training_data(self, train_loader: DataLoader):
        """
        This function sets the training data
        """
        self.train_loader = train_loader

    def set_validation_data(
        self,
        validation_loader: DataLoader,
        validation_freq: int,
        validation_fn: callable = None,
    ):
        """
        This function sets the validation data
        """
        self.validation_loader = validation_loader
        self.validation_freq = validation_freq
        self.validation_fn = validation_fn

    def set_testing_data(self, test_loader: DataLoader, testing_fn: callable):
        """
        This function sets the testing data
        """
        self.test_loader = test_loader
        self.testing_fn = testing_fn

    def set_checkpointing(
        self,
        save_folder: pathlib.Path,
        checkpoint_fname: pathlib.Path,
        checkpoint_freq: int = 10,
        load_checkpoint: bool = True,
    ):
        """
        This function sets the checkpointing
        """
        self.checkpoint_freq = checkpoint_freq
        self.save_folder = save_folder
        self.checkpoint_fname = checkpoint_fname
        self.load_checkpoint = load_checkpoint

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

        # Test 1: is the device set? if not, set it if autofill is True
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
            error=False,
            autofill=False,
            verbose=True,
        )
        # Test 12: is the final result filename set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "final_result_fname",
            expected_type=pathlib.Path,
            error=False,
            autofill=False,
            verbose=True,
        )
        # Test 13: is the checkpoint frequency filename set? if not, raise error or warn or autofill

        if self.final_result_fname is not None:
            default_checkpoint_fname = self.final_result_fname.parent / pathlib.Path(
                str(self.final_result_fname.stem) + "_checkpoint_*.pt"
            )
        else:
            default_checkpoint_fname = ""
        return_code = self.__check_attribute(
            "checkpoint_fname",
            expected_type=pathlib.Path,
            error=False,
            autofill=True,
            verbose=False,
            default=default_checkpoint_fname,
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

        # Test 15: is the load checkpoint set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "load_checkpoint",
            expected_type=bool,
            error=False,
            autofill=True,
            verbose=verbose,
            default=True,
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
        warnings.formatwarning = custom_format_warning

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

    def save_checkpoint(self, epoch):
        """
        This function saves a checkpoint of the model and the optimizer
        """
        self.model.save_checkpoint(
            self.save_folder.joinpath(
                pathlib.Path(str(self.checkpoint_fname).replace("*", f"{epoch+1:04d}"))
            ),
            epoch + 1,
            self.loss,
            self.optimizer,
            self.metadata,
            dataset=self.dataset_param,
        )

    def save_final_results(self, final_result_fname=None, epoch=None):
        """
        This function saves the final results of the optimization
        """
        if epoch is None:
            epoch = self.epochs
        if final_result_fname is not None:
            self.final_result_fname = final_result_fname
        self.model.save(
            self.final_result_fname,
            epoch=epoch,
            training=self.metadata,
            loss=self.train_loss,
            dataset=self.dataset_param,
        )

    def clean_checkpoints(self):
        """
        This function cleans the checkpoints
        """
        for f in self.save_folder.glob(str(self.checkpoint_fname).replace("*", "*")):
            f.unlink()

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
