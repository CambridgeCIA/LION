# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================

# %% This is a base class for solvers/trainers that gives you helpful functions.
#
# It defines a bunch of auxiliary functions
#

# You will want to import LIONParameter, as all models must save and use Parameters.
from enum import Enum
from typing import Callable, Optional
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.exceptions.exceptions import LIONSolverException
from LION.utils.parameter import LIONParameter

# Lionmodels
from LION.models.LIONmodel import LIONmodel, ModelInputType

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

# TODO: finish this
class SolverState(Enum):
    COMPLETE=0

class SolverParams(LIONParameter):
    def __init__(self):
        super().__init__()


class LIONsolver(ABC, metaclass=ABCMeta):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        geometry: Geometry,
        verbose: bool=True,
        device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}"),
        model_regularization=None,
        solver_params: SolverParams=SolverParams()
    ) -> None:
        super().__init__()

        assert isinstance(model, LIONmodel), "model must be a LIONmodel"
        assert isinstance(
            optimizer, torch.optim.Optimizer
        ), "optimizer must be a torch optimizer"
        assert callable(loss_fn), "loss_fn must be a function"

        # currently not used in subclasses or here, but we'll save it with the view that we'll want to serialize these at some point
        # relevant solver_params are extracted into data members on a subclass level
        self.solver_params = solver_params

        self.model = model
        self.optimizer = optimizer
        self.geo = geometry
        self.loss_fn = loss_fn
        self.device = device
        self.train_loss: np.ndarray = np.zeros(0)
        self.validation_loader: Optional[DataLoader] = None
        self.validation_fn: Optional[Callable] = None
        self.validation_freq: Optional[int] = None
        self.validation_loss: Optional[np.ndarray] = None
        self.test_loader: DataLoader
        self.testing_fn: Callable
        self.current_epoch: int = 0
        self.save_folder: Optional[pathlib.Path] = None
        self.load_folder: Optional[pathlib.Path] = None
        self.checkpoint_freq: int
        self.final_result_fname: Optional[str] = None
        self.checkpoint_fname: Optional[str] = None
        self.validation_fname: Optional[str] = None
        self.verbose = verbose
        self.model_regularization = model_regularization
        self.metadata = LIONParameter()
        self.dataset_param = LIONParameter()

    # This should return the default parameters of the solver
    @staticmethod
    @abstractmethod  # crash if not defined in derived class
    def default_parameters() -> SolverParams:
        pass

    def set_training(self, train_loader: DataLoader):
        """
        This function sets the training data
        """
        self.train_loader = train_loader

    def set_validation(
        self,
        validation_loader: DataLoader,
        validation_freq: int,
        validation_fn: Optional[Callable] = None,
        validation_fname: Optional[str] = None,
    ):
        """
        This function sets the validation data
        """
        self.validation_loader = validation_loader
        self.validation_freq = validation_freq
        self.validation_fn = validation_fn if validation_fn is not None else self.loss_fn
        self.validation_fname = validation_fname

    def set_testing(self, test_loader: DataLoader, testing_fn: Callable):
        """
        This function sets the testing data
        """
        self.test_loader = test_loader
        self.testing_fn = testing_fn

    def set_saving(self, save_folder: str | pathlib.Path, final_result_fname: str):
        if isinstance(save_folder, str):
            save_folder = pathlib.Path(save_folder)
        if not save_folder.is_dir():
            raise ValueError(f"Save folder '{save_folder}' is not a directory, failed to set saving.")
        
        self.save_folder = save_folder
        self.final_result_fname = final_result_fname
    
    def set_loading(self, load_folder: str | pathlib.Path):
        if isinstance(load_folder, str):
            load_folder = pathlib.Path(load_folder)
        if not load_folder.is_dir():
            raise ValueError(f"Save folder '{load_folder}' is not a directory, failed to set saving.")
        
        self.load_folder = load_folder

    def set_checkpointing(
        self,
        checkpoint_fname: str,
        checkpoint_freq: int = 10,
        load_checkpoint: bool = False,
    ):
        """
        This function sets the checkpointing
        """
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_fname = checkpoint_fname
        self.do_load_checkpoint = load_checkpoint

    def check_training_ready(self, error=True, autofill=True, verbose=True):
        """This should always pass, all of these things are required to initialize a LIONsolver object

        Args:
            error (bool, optional): _description_. Defaults to True.
            autofill (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        return_code = 0

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

        # Test 12: is the final result filename set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "final_result_fname",
            expected_type=str,
            error=False,
            autofill=False,
            verbose=True,
        )

        return return_code
    
    def check_validation_ready(self, autofill=True, verbose=True):
        return_code = 0

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

        # Test 14: is the validation filename set? if not, raise error or warn or autofill
        if self.final_result_fname is not None and self.save_folder is not None:
            default_validation_fname = f"{self.final_result_fname}_min_val.pt"
        else:
            default_validation_fname = None
        return_code = self.__check_attribute(
            "validation_fname",
            expected_type=str,
            error=False,
            autofill=True,
            verbose=False,
            default=default_validation_fname,
        )

        return return_code
    
    def check_testing_ready(self, error=True, verbose=True):
        return_code = 0

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

        return return_code

    def check_checkpointing_ready(self, autofill=True, verbose=True):
        return_code = 0

        # Test 13: is the checkpoint filename filename set? if not, raise error or warn or autofill
        if self.final_result_fname is not None and self.save_folder is not None:
            default_checkpoint_fname = f"{self.final_result_fname}_checkpoint_*.pt"
        else:
            default_checkpoint_fname = None
        return_code = self.__check_attribute(
            "checkpoint_fname",
            expected_type=str,
            error=False,
            autofill=True,
            verbose=False,
            default=default_checkpoint_fname,
        )

        # Test 15: is the checkpoint frequency set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "checkpoint_freq",
            expected_type=int,
            error=False,
            autofill=autofill,
            verbose=verbose,
            default=10,
        )

        # Test 16: is the load checkpoint set? if not, raise error or warn or autofill
        return_code = self.__check_attribute(
            "load_checkpoint",
            expected_type=bool,
            error=False,
            autofill=True,
            verbose=verbose,
            default=True,
        )

        return return_code

    def check_saving_ready(self):
        # Test 11: is the save folder set? if not, raise error or warn or autofill
        return self.__check_attribute(
            "save_folder",
            expected_type=pathlib.Path,
            error=False,
            autofill=False,
            verbose=True,
        )
    
    def check_regularization_ready(self):
        # might be more that needs to be done in here, don't really know much about regularization

        # Test 17: is the model regularization set?
        return self.__check_attribute(
            "model_regularization",
            expected_type=nn.Module,
            error=False,
            autofill=False,
            verbose=False,
        )

    def check_complete(self, error=True, autofill=True):
        """
        This function checks if the solver is complete, i.e. if all the necessary parameters are set to start traning.
        """

        return_code = 0

        # Set if we need to complain
        # should never have to go into the else, it's always set, but leave this here as legacy just in case...
        if hasattr(self, "verbose"):
            verbose = self.verbose
        else:
            verbose = True
            self.verbose = verbose

        return_code = self.check_training_ready(error, autofill, verbose)

        return_code = self.check_testing_ready(error, verbose)

        return_code = self.check_saving_ready()

        return_code = self.check_validation_ready(error, autofill)
        
        return_code = self.check_checkpointing_ready(autofill, verbose)
        
        if self.model_regularization is not None:
            return_code = self.check_regularization_ready() 

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
        if self.save_folder is None:
            raise LIONSolverException("Saving not set: please call set_saving")
        self.model.save_checkpoint(
            self.save_folder.joinpath(
                pathlib.Path(str(self.checkpoint_fname).replace("*", f"{epoch+1:04d}"))
            ),
            epoch + 1,
            self.train_loss,
            self.optimizer,
            self.metadata,
            dataset=self.dataset_param,
        )

    def save_validation(self, epoch):
        """
        This function saves the validation results
        """
        if self.validation_fname is None or self.validation_fn is None:
            raise LIONSolverException("No validation save filepath provided. Please call set_validation.")

        if self.validation_loss is None: 
            raise LIONSolverException("No validation losses found, failed to save.")

        if self.save_folder is None:
            raise LIONSolverException("Saving not setup: Please call set_saving.")
        
        self.model.save(
            self.save_folder.joinpath(self.validation_fname),
            epoch=epoch,
            training=self.metadata,
            loss=self.validation_loss[epoch],
            dataset=self.dataset_param,
        )

    def save_final_results(self, epoch=None):
        """
        This function saves the final results of the optimization
        """
        if self.save_folder is None or self.final_result_fname is None:
            raise LIONSolverException("Saving not setup: Please call set_saving.")
        if epoch is None:
            epoch = self.current_epoch

        self.model.save(
            self.save_folder.joinpath(self.final_result_fname),
            epoch=epoch,
            training=self.metadata,
            loss=self.train_loss,
            dataset=self.dataset_param,
        )

    def clean_checkpoints(self):
        """
        This function cleans the checkpoints
        """
        if self.save_folder is None:
            raise LIONSolverException("Saving not setup: Please call set_saving")
        if self.checkpoint_fname is None:
            raise LIONSolverException("Checkpointing not setup, can't clear checkpoint: Please call set_checkpointing")
        # TODO: This doesn't delete the .jsons only the .pt, is this intentional behaviour?
        for f in self.save_folder.glob(str(self.checkpoint_fname).replace("*", "*")):
            f.unlink()
        
    @abstractmethod
    def test(self):
        """
        This function performs a testing step
        """
        pass

    def load_checkpoint(self):
        """
        This function loads a checkpoint (if exists)
        """
        if self.load_folder is None:
            raise LIONSolverException( "Loading not set. Please call set_loading ")
        if self.checkpoint_fname is None:
            raise LIONSolverException( "Checkpointing not set, failed to load checkpoint. Please call set_checkpointing")
        (
            self.model,
            self.optimizer,
            epoch,
            self.train_loss,
            _,
        ) = self.model.load_checkpoint_if_exists(
            self.load_folder.joinpath(self.checkpoint_fname),
            self.model,
            self.optimizer,
            self.train_loss,
        )
        if (
            self.validation_fn is not None
            and epoch > 0
            and self.validation_fname is not None
            and self.validation_loss is not None
        ):
            self.validation_loss[epoch - 1] = self.model._read_min_validation(
                self.load_folder.joinpath(self.validation_fname)
            )
            if self.verbose:
                print(
                    f"Loaded checkpoint at epoch {epoch}. Current min validation loss is {self.validation_loss[epoch-1]}"
                )
        return epoch

    @abstractmethod
    def mini_batch_step(self):
        """
        This function should perform a single step of the optimization
        """
        pass

    @abstractmethod
    def epoch_step(self, epoch):
        """
        This function should perform a single epoch of the optimization
        """
        pass

    @abstractmethod
    def train(self, n_epochs: int):
        """
        This function is responsible for performing the optimization.
        """
        pass

    @abstractmethod
    def validate(self):
        """
        This function should perform a validation step
        """
        pass
