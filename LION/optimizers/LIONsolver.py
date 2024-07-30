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

from tqdm import tqdm
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
from LION.exceptions.exceptions import LIONSolverException, NoDataException
from LION.utils.parameter import LIONParameter

# Lionmodels
from LION.models.LIONmodel import LIONmodel, ModelInputType

# Some utils
from LION.utils.utils import custom_format_warning

# some numerical standard imports, e.g.
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

# imports related to class organization
from abc import ABC, abstractmethod, ABCMeta

# general imports
import warnings
import pathlib


# TODO: finish this
class SolverState(Enum):
    COMPLETE = 0


class SolverParams(LIONParameter):
    def __init__(self):
        super().__init__()


def normalize_input(func):
    def wrapper(self, *inputs):
        if self.do_normalize:
            normalized_inputs = []
            for x in inputs:
                normalized_x = (x - self.xmin) / (self.xmax - self.xmin)
                normalized_inputs.append(normalized_x)
            return func(self, *normalized_x)
        else:
            return func(self, *inputs)

    return wrapper


class LIONsolver(ABC, metaclass=ABCMeta):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn: torch.nn.Module,
        geometry: Geometry,
        verbose: bool = True,
        device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}"),
        solver_params: Optional[SolverParams] = None,
    ) -> None:
        super().__init__()
        if solver_params is None:
            self.solver_params = self.default_parameters()

        assert isinstance(model, LIONmodel), "model must be a LIONmodel"
        assert isinstance(optimizer, Optimizer), "optimizer must be a torch optimizer"

        # currently not used in subclasses or here, but we'll save it with the view that we'll want to serialize these at some point
        # relevant solver_params are extracted into data members on a subclass level
        self.solver_params = solver_params

        self.model = model
        self.optimizer = optimizer
        self.geo = geometry
        self.op = make_operator(self.geo)

        self.train_loader: Optional[DataLoader] = None
        self.train_loss: np.ndarray = np.zeros(0)

        self.loss_fn = loss_fn

        self.device = device

        self.validation_loader: Optional[DataLoader] = None
        self.validation_fn: Optional[Callable] = None
        self.validation_freq: Optional[int] = None
        self.validation_loss: Optional[np.ndarray] = None

        self.test_loader: Optional[DataLoader] = None
        self.testing_fn: Optional[Callable] = None

        self.current_epoch: int = 0

        self.save_folder: Optional[pathlib.Path] = None
        self.load_folder: Optional[pathlib.Path] = None

        self.do_load_checkpoint: bool = False
        self.checkpoint_freq: int

        self.final_result_fname: Optional[str] = None
        self.checkpoint_fname: Optional[str] = None
        self.validation_fname: Optional[str] = None

        self.verbose = verbose
        self.metadata = LIONParameter()
        self.dataset_param = LIONParameter()

        # normalization stuff
        self.do_normalize: bool = False
        self.xmin: Optional[float] = None
        self.xmax: Optional[float] = None

    # This should return the default parameters of the solver
    @staticmethod
    @abstractmethod  # crash if not defined in derived class
    def default_parameters() -> SolverParams:
        pass

    def set_training(
        self, train_loader: DataLoader, loss_fn: Optional[Callable] = None
    ):
        """
        This function sets the training data
        """
        self.train_loader = train_loader
        if loss_fn is not None:
            self.loss_fn = loss_fn

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
        self.validation_fn = (
            validation_fn if validation_fn is not None else self.loss_fn
        )
        self.validation_fname = validation_fname

    def set_testing(
        self, test_loader: DataLoader, testing_fn: Optional[Callable] = None
    ):
        """
        This function sets the testing data
        """
        self.test_loader = test_loader
        self.testing_fn = testing_fn if testing_fn is not None else self.loss_fn

    def set_saving(self, save_folder: str | pathlib.Path, final_result_fname: str):
        if isinstance(save_folder, str):
            save_folder = pathlib.Path(save_folder)
        if not save_folder.is_dir():
            raise ValueError(
                f"Save folder '{save_folder}' is not a directory, failed to set saving."
            )

        self.save_folder = save_folder
        self.final_result_fname = final_result_fname

    def set_loading(self, load_folder: str | pathlib.Path, do_load: bool = False):
        if isinstance(load_folder, str):
            load_folder = pathlib.Path(load_folder)
        if not load_folder.is_dir():
            raise ValueError(
                f"Save folder '{load_folder}' is not a directory, failed to set saving."
            )

        self.load_folder = load_folder
        self.do_load_checkpoint = do_load

    def set_checkpointing(
        self,
        checkpoint_fname: str,
        checkpoint_freq: int = 10,
    ):
        """
        This function sets the checkpointing
        """
        if self.save_folder is None:
            warnings.warn("Save folder not set. Please call set_saving")
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_fname = checkpoint_fname

    def set_normalization(self, do_normalize: bool):
        print(self.model.model_parameters.model_input_type)
        if self.model.get_input_type() == ModelInputType.SINOGRAM:
            warnings.warn(
                """Normalization will not be carried out on this model,
                as it takes inputs in the measurement domain. 
                As such inputs cannot be normalized in the image domain before being passed to the model.
                In such a case, normalization should be implemented within the model itself"""
            )
        if self.train_loader is None:
            raise NoDataException(
                "Training dataloader not set: Please call set_training"
            )
        self.do_normalize = do_normalize
        if self.do_normalize:
            xmax = -np.inf
            xmin = np.inf
            for x in self.train_loader:
                xmax = max(x[1].max(), xmax)
                xmin = min(x[1].min(), xmin)
            self.xmin = xmin
            self.xmax = xmax

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
            expected_type=Optimizer,
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
            raise LIONSolverException(
                "No validation save filepath provided. Please call set_validation."
            )

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
            raise LIONSolverException(
                "Saving not setup, unable to find save folder: Please call set_saving"
            )
        if self.checkpoint_fname is None:
            raise LIONSolverException(
                "Checkpointing not setup, can't clear checkpoints: Please call set_checkpointing"
            )
        # TODO: This doesn't delete the .jsons only the .pt, is this intentional behaviour?
        # Quick and dirty fix with fancy regex
        for f in self.save_folder.glob(self.checkpoint_fname.replace(".pt", "")):
            f.unlink()

    def normalize(self, x):
        """Normalizes input data
        returns: Normalized input data
        """
        if self.do_normalize:
            assert self.xmax is not None and self.xmin is not None
            normalized_x = (x - self.xmin) / (self.xmax - self.xmin)
        return normalized_x

    def test(self):
        self.model.eval()
        if self.check_testing_ready() != 0:
            warnings.warn("Solver not setup to test. Please call set_testing.")
            return np.array([])
        assert self.test_loader is not None
        assert self.testing_fn is not None

        with torch.no_grad():
            test_loss = np.array([])
            for data, target in tqdm(self.test_loader):
                if self.model.get_input_type() == ModelInputType.IMAGE:
                    data = fdk(data, self.op)
                output = self.model(data.to(self.device))
                test_loss = np.append(
                    test_loss, self.testing_fn(output, target.to(self.device))
                )

        if self.verbose:
            print(
                f"Testing loss: {test_loss.mean()} - Testing loss std: {test_loss.std()}"
            )

        return test_loss

    def load_checkpoint(self):
        """
        This function loads a checkpoint (if exists)
        """
        if self.load_folder is None:
            raise LIONSolverException("Loading not set. Please call set_loading ")
        if self.checkpoint_fname is None:
            raise LIONSolverException(
                "Checkpointing not set, failed to load checkpoint. Please call set_checkpointing"
            )
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
        for _, (data, target) in enumerate(tqdm(self.train_loader)):
            epoch_loss += self.mini_batch_step(
                data.to(self.device), target.to(self.device)
            )
        return epoch_loss / len(self.train_loader)

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
        """
        assert n_epochs > 0, "Number of epochs must be a positive integer"
        # Make sure all parameters are set
        self.check_training_ready()

        if self.do_load_checkpoint:
            print("Loading checkpoint...")
            self.current_epoch = self.load_checkpoint()
            self.train_loss = np.append(self.train_loss, np.zeros((n_epochs)))
        else:
            self.train_loss = np.zeros(n_epochs)

        if self.check_validation_ready() == 0:
            self.validation_loss = np.zeros((n_epochs))

        self.model.train()
        # train loop
        final_total_epochs = self.current_epoch + n_epochs
        while self.current_epoch < final_total_epochs:
            print(f"Training epoch {self.current_epoch + 1}")
            self.epoch_step(self.current_epoch)

            if (self.current_epoch + 1) % self.checkpoint_freq == 0:
                self.save_checkpoint(self.current_epoch)

            self.current_epoch += 1

    @abstractmethod
    def validate(self):
        """
        This function should perform a validation step
        """
        pass

    @abstractmethod
    def mini_batch_step(self, sino_batch, target_batch) -> float:
        """
        This function should perform a single step of the optimization
        """
        pass
