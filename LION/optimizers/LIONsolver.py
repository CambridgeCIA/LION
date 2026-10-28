# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================

# %% This is a base class for solvers/trainers that gives you helpful functions.
#
# It defines a bunch of auxiliary functions
#

# You will want to import LIONParameter, as all models must save and use Parameters.
from __future__ import annotations
from enum import Enum
from typing import Callable, Optional

from tqdm import tqdm
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
from LION.exceptions.exceptions import LIONSolverException, NoDataException
from LION.utils.parameter import LIONParameter
from LION.utils.device import default_device
from LION.utils import rng, spec
from LION.data_loaders.data_record import data_record
from LION.optimizers import training_record as records

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
import os
import warnings
import pathlib


# TODO: finish this
class SolverState(Enum):
    COMPLETE = 0


class SolverParams(LIONParameter):
    def __init__(self):
        super().__init__()


def warn_if_not_mse(loss_fn, solver_name: str):
    """Warn that ``loss_fn`` is not used by a solver whose loss must be the MSE.

    Some self-supervised losses are only valid for the mean squared error
    (e.g. their target is a conditional mean), so these solvers use the MSE
    whatever loss is passed; ``loss_fn`` is still the default validation and
    testing function.
    """
    if loss_fn is not None and not isinstance(loss_fn, torch.nn.MSELoss):
        warnings.warn(
            f"{solver_name} trains with the mean squared error, which its derivation "
            f"requires; the given loss_fn ({type(loss_fn).__name__}) is only used "
            "as the default validation and testing function."
        )


def model_forward(model: LIONmodel, data: torch.Tensor, op) -> torch.Tensor:
    """Run ``model`` on a batch of measurements, as the solvers do (and
    ``LION.reconstructors.ForwardReconstructor``).

    Image-domain models get an FDK reconstruction (with operator ``op``) as
    input. Sinogram-domain models get the measurements. With a normalisation
    set (``model.set_normalisation()``), the input is normalised (for
    measurements: as the images are, by linearity) and the output mapped
    back, so the output is in the units of the target.
    """
    normalisation = getattr(model, "normalisator", None)
    if model.get_input_type() == ModelInputType.IMAGE:
        data = fdk(data, op)
        if normalisation is not None:
            return model.unnormalise(model(model.normalise(data)), data)
        return model(data)
    if normalisation is not None:
        return model.unnormalise(model(normalisation.normalise_measurements(data, op)))
    return model(data)


class LIONsolver(ABC, metaclass=ABCMeta):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn: Callable,
        geometry: Geometry = None,
        verbose: bool = True,
        device: torch.device = None,
        solver_params: Optional[SolverParams] = None,
        save_folder: Optional[pathlib.Path] = None,
        seed: Optional[int] = 0,
        deterministic: bool = False,
    ) -> None:
        """
        ``seed``: the training seed. When training starts, torch (CPU and
        CUDA), numpy and Python's ``random`` are seeded with it, so the same
        solver, model and data give the same training, and checkpoints keep the
        random state, so a resumed run continues the same way. The randomness
        within a run (shuffling, noise a solver adds) still changes every batch
        and epoch. Two runs with the same seed give the same model: use
        different seeds for an ensemble. ``None``: unseeded.

        ``deterministic``: use only deterministic algorithms (torch's
        ``use_deterministic_algorithms``, deterministic cuDNN), so that on a GPU
        two runs with the same seed give exactly the same model, and a resumed
        run the uninterrupted one. Slower; a few operations then raise (torch
        names them). It applies to the whole program from here on. Off, GPU
        runs with the same seed differ slightly (e.g. ~1e-3 in the weights after
        a few epochs). On a GPU it needs ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
        before cuBLAS starts: it is set here if it isn't (exporting it before the
        script is surer).

        ``save_folder``: where results are saved; by default the folder in the
        ``LION_RESULTS_PATH`` environment variable, if set.
        """

        super().__init__()
        if solver_params is None:
            self.solver_params = self.default_parameters()
        else:
            self.solver_params = solver_params
        assert isinstance(model, LIONmodel), "model must be a LIONmodel"
        assert isinstance(optimizer, Optimizer), "optimizer must be a torch optimizer"

        self.model = model
        self.optimizer = optimizer

        if hasattr(model, "geometry") and model.geometry is not None:
            self.geometry = model.geometry
        else:
            assert geometry is not None, "Geometry must be provided"
            self.geometry = geometry

        self.op = make_operator(self.geometry)

        self.train_loader: Optional[DataLoader] = None
        self.train_loss: np.ndarray = np.zeros(0)

        self.loss_fn = loss_fn
        if device is None:
            device = default_device()
        self.device = device
        self.model.to(self.device)

        self.validation_loader: Optional[DataLoader] = None
        self.validation_fn: Optional[Callable] = None
        self.validation_freq: Optional[int] = None
        self.validation_loss: Optional[np.ndarray] = None
        self.validation_reconstructor = None

        self.test_loader: Optional[DataLoader] = None
        self.testing_fn: Optional[Callable] = None
        self.testing_reconstructor = None

        self.current_epoch: int = 0

        if save_folder is None:
            save_folder = os.environ.get("LION_RESULTS_PATH")
        self.save_folder: Optional[pathlib.Path] = (
            None if save_folder is None else pathlib.Path(save_folder).expanduser()
        )
        self.seed = seed
        self.deterministic = deterministic
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.n_epochs: Optional[int] = None
        self.initial_weights_sha256: Optional[str] = None
        self._data_records = {}

        self.do_load_checkpoint: bool = False
        self.checkpoint_freq: Optional[int] = None

        self.final_result_fname: Optional[str] = None
        self.checkpoint_fname: Optional[str] = None
        self.validation_fname: Optional[str] = None
        self.checkpoint_save_folder: Optional[pathlib.Path] = None
        self.verbose = verbose

    @property
    def do_normalise(self) -> bool:
        """True if the model has a normalisation set (``model.set_normalisation()``)."""
        return getattr(self.model, "normalisator", None) is not None

    def _forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run the model on a batch of measurements, as in training (see
        ``model_forward``)."""
        return model_forward(self.model, data, self.op)

    # ------------------------------------------------------------------
    # The training record (LION.optimizers.training_record)

    def spec_kwargs(self) -> dict:
        """The solver's own constructor arguments: not the model (saved as
        the model) nor the optimizer (built on the model's parameters)."""
        kwargs = spec._constructor_arguments(self)
        kwargs.pop("model", None)
        kwargs.pop("optimizer", None)
        return kwargs

    def _data_record(self, loader) -> dict:
        """The data record of a loader's dataset, made once per loader."""
        key = id(loader.dataset)
        if key not in self._data_records:
            self._data_records[key] = data_record(loader.dataset)
        return self._data_records[key]

    def training_record(self, kind: str, epoch: int, value=None) -> LIONParameter:
        """The record of this training, for a file holding the ``kind``
        ("final", "best_validation", "checkpoint") weights after ``epoch``
        epochs (see ``LION.optimizers.training_record``)."""
        solver = {spec.TARGET: spec.import_path(type(self))}
        for name, argument in self.spec_kwargs().items():
            if isinstance(argument, LIONParameter):
                # value by value, so one that can't be saved (e.g. a lambda
                # recon_fn) doesn't hide the others
                values = (
                    argument.spec_kwargs()
                    if hasattr(argument, "spec_kwargs")
                    else vars(argument)
                )
                solver[name] = {
                    spec.TARGET: spec.import_path(type(argument)),
                    **{
                        k: spec.encode_or_name(v, f"{name}.{k}")
                        for k, v in values.items()
                    },
                }
            else:
                solver[name] = spec.encode_or_name(argument, name)
        record = {
            records.RECORD_KEY: records.RECORD_VERSION,
            "solver": solver,
            "optimizer": spec.encode_or_name(self.optimizer, "the optimizer"),
            "seed": self.seed,
            "initial_weights_sha256": self.initial_weights_sha256,
        }
        if self.train_loader is not None:
            record["train"] = records.loader_settings(self.train_loader)
        if self.validation_loader is not None and self.validation_fn is not None:
            record["validation"] = {
                "data": self._data_record(self.validation_loader),
                **records.loader_settings(self.validation_loader),
                "every": self.validation_freq,
                "function": spec.encode_or_name(self.validation_fn, "it"),
            }
            if self.validation_reconstructor is not None:
                record["validation"]["reconstructor"] = spec.encode_or_name(
                    self.validation_reconstructor, "the validation reconstructor"
                )
        record["epochs"] = {"asked": self.n_epochs, "done": epoch}
        record["weights"] = {"kind": kind, "epoch": epoch}
        if value is not None:
            record["weights"]["value"] = float(value)
        record["history"] = {
            "train_loss": records.history(self.train_loss, epoch),
            "validation": records.history(self.validation_loss, epoch),
        }
        return LIONParameter(**record)

    def _training_data_record(self):
        """The training data's record: the model file's ``dataset_params``."""
        if self.train_loader is None:
            return None
        return self._data_record(self.train_loader)

    @classmethod
    def instantiate(cls, source, model: LIONmodel, *, allow=(), **arguments):
        """The solver a saved file describes, training ``model``.

        ``source``: a model file a solver saved (or the dict read from it), or
        its training record. Made from the solver's own record only: its
        class, ``solver_params``, loss and seed, and the optimizer (class and
        hyperparameters) on ``model``'s parameters. The data, validation,
        testing and saving are set as usual (``set_training``, ...).
        ``arguments`` replace recorded ones or give those that couldn't be
        recorded (e.g. ``loss_fn`` for a lambda loss) and where it runs
        (``device``, ``verbose``, ``save_folder``). Called on ``LIONsolver``
        it makes the class the record names; on a solver class, that class.
        """
        record = records.read_training_record(source)
        if spec.is_unsaved(record.optimizer):
            raise ValueError(
                f"the record only names its optimizer, "
                f"{record.optimizer.unsaved}: it can't be made again"
            )
        solver_spec = {
            k: v
            for k, v in record.solver.items()
            if k == spec.TARGET or k not in arguments
        }
        unsaved = [
            name
            for name, value in solver_spec.items()
            if isinstance(value, dict)
            and (
                "unsaved" in value
                or any(isinstance(v, dict) and "unsaved" in v for v in value.values())
            )
        ]
        if unsaved:
            raise ValueError(
                f"the record only names (part of) {', '.join(unsaved)}, which can't "
                "be made again: give " + ", ".join(f"{k}=..." for k in unsaved)
            )
        path, kwargs = spec.build_arguments(solver_spec, allow)
        target = spec.target_class(path, cls, LIONsolver, allow)
        optimizer = spec.build(record.optimizer, allow, params=model.parameters())
        return target(model=model, optimizer=optimizer, **kwargs, **arguments)

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
        reconstructor=None,
    ):
        """
        This function sets the validation data.

        The model with the best validation value is saved as ``validation_fname``
        in the solver's save folder (see ``set_saving()``). Best is the lowest
        value, unless ``validation_fn`` has ``higher_is_better = True`` (as
        LION's PSNR and SSIM metrics do; any function can set it, e.g.
        ``my_metric.higher_is_better = True``).

        ``reconstructor``: a reconstructor (``LION.reconstructors``) to
        reconstruct the validation data with, using the model being trained
        (set as its ``model``); otherwise the solver's own ``_forward``. It is
        made outside the solver, with its own parameters, and recorded.
        """
        self.validation_loader = validation_loader
        self.validation_freq = validation_freq
        self.validation_fn = (
            validation_fn if validation_fn is not None else self.loss_fn
        )
        self.validation_fname = validation_fname
        self.validation_reconstructor = reconstructor

    def set_testing(
        self,
        test_loader: DataLoader,
        testing_fn: Optional[Callable] = None,
        reconstructor=None,
    ):
        """
        This function sets the testing data. ``reconstructor``: as in
        ``set_validation``.
        """
        self.test_loader = test_loader
        self.testing_fn = testing_fn if testing_fn is not None else self.loss_fn
        self.testing_reconstructor = reconstructor

    def _reconstruct(self, data, reconstructor=None):
        """``data`` reconstructed with the model being trained: through
        ``reconstructor`` (given the model) if there is one, otherwise the
        solver's ``_forward``."""
        if reconstructor is None:
            return self._forward(data)
        reconstructor.model = self.model
        return reconstructor.reconstruct(data)

    def set_saving(self, save_folder: str | pathlib.Path, final_result_fname: str):
        """Sets save_folder and filename for saving final result and min_val result

        Args:
            save_folder (str | pathlib.Path): _description_
            final_result_fname (str): _description_

        Raises:
            ValueError: _description_
        """
        if isinstance(save_folder, str):
            save_folder = pathlib.Path(save_folder)
        if not save_folder.is_dir():
            raise ValueError(
                f"Save folder '{save_folder}' is not a directory, failed to set saving."
            )

        self.save_folder = save_folder
        self.final_result_fname = final_result_fname

    def set_checkpointing(
        self,
        checkpoint_fname: str,
        checkpoint_freq: int = 10,
        load_checkpoint_if_exists: bool = True,
        save_folder: str | pathlib.Path = None,
    ):
        """
        This function sets the checkpointing
        """
        if isinstance(save_folder, str):
            save_folder = pathlib.Path(save_folder)
        elif save_folder is None:
            save_folder = self.save_folder
        if save_folder is None:
            raise LIONSolverException(
                "No folder for the checkpoints: pass save_folder, or call set_saving() first."
            )

        if not save_folder.is_dir():
            raise ValueError(
                f"Save folder '{save_folder}' is not a directory, failed to set checkpointing."
            )

        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_fname = checkpoint_fname
        self.do_load_checkpoint = load_checkpoint_if_exists
        self.checkpoint_save_folder = save_folder

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
        return_code = max(
            return_code,
            self.__check_attribute(
                "device",
                expected_type=torch.device,
                error=False,
                autofill=autofill,
                verbose=verbose,
                default=default_device(),
            ),
        )

        # Test 2: is the model set? if not, raise error or warn
        return_code = max(
            return_code,
            self.__check_attribute(
                "model",
                expected_type=LIONmodel,
                error=error,
                autofill=False,
                verbose=verbose,
            ),
        )

        # Test 3: is the optimizer set? if not, raise error or warn
        return_code = max(
            return_code,
            self.__check_attribute(
                "optimizer",
                expected_type=Optimizer,
                error=error,
                autofill=False,
                verbose=verbose,
            ),
        )

        # Test 4: is the loss_fn set? if not, raise error or warn
        return_code = max(
            return_code,
            self.__check_attribute(
                "loss_fn",
                expected_type=callable,
                error=error,
                autofill=False,
                verbose=verbose,
            ),
        )

        # Test 7: is the training loader set? if not, raise error or warn
        return_code = max(
            return_code,
            self.__check_attribute(
                "train_loader",
                expected_type=DataLoader,
                error=error,
                autofill=False,
                verbose=verbose,
            ),
        )

        # Test 12: is the final result filename set? if not, raise error or warn or autofill
        return_code = max(
            return_code,
            self.__check_attribute(
                "final_result_fname",
                expected_type=str,
                error=False,
                autofill=False,
                verbose=True,
            ),
        )

        return return_code

    def check_validation_ready(self, autofill=True, verbose=True):
        return_code = 0

        # Test 8: is the validation loader set? if not, raise error or warn
        return_code = max(
            return_code,
            self.__check_attribute(
                "validation_loader",
                expected_type=DataLoader,
                error=False,
                autofill=False,
                verbose=True,
            ),
        )

        # Test 9: is the validation function set? if not, raise error or warn or autofill
        return_code = max(
            return_code,
            self.__check_attribute(
                "validation_fn",
                expected_type=callable,
                error=False,
                autofill=self.validation_loader is not None,
                verbose=verbose,
                default=self.loss_fn,
            ),
        )
        # Test 10: is the validation frequency set? if not, raise error or warn or autofill
        return_code = max(
            return_code,
            self.__check_attribute(
                "validation_freq",
                expected_type=int,
                error=False,
                autofill=self.validation_loader is not None,
                verbose=verbose,
                default=10,
            ),
        )

        # Test 14: file name of the min-validation model (saved in self.save_folder).
        # Not needed to validate: without it the model is just not saved.
        if (
            self.validation_fname is None
            and self.final_result_fname is not None
            and self.save_folder is not None
        ):
            self.validation_fname = f"{self.final_result_fname}_min_val.pt"

        return return_code

    def _validation_higher_is_better(self) -> bool:
        """True if the validation function is a score to maximise (e.g. PSNR, SSIM)."""
        return bool(getattr(self.validation_fn, "higher_is_better", False))

    def _is_best_validation(self, epoch: int) -> bool:
        """True if the validation value of ``epoch`` is the best so far.

        Epochs without validation have NaN in ``validation_loss`` and are
        skipped (a validation value can be 0, e.g. a loss of a perfect model).
        """
        value = self.validation_loss[epoch]
        if np.isnan(value):  # e.g. diverged: never the best
            return False
        done = self.validation_loss[~np.isnan(self.validation_loss)]
        if self._validation_higher_is_better():
            return value >= np.max(done)
        return value <= np.min(done)

    def _saves_min_validation(self) -> bool:
        """True if the model with the lowest validation loss can be saved."""
        return self.validation_fname is not None and self.save_folder is not None

    def check_testing_ready(self, error=True, verbose=True):
        return_code = 0

        # Test 5: is the testing loader set? if not, raise error or warn
        return_code = max(
            return_code,
            self.__check_attribute(
                "test_loader",
                expected_type=DataLoader,
                error=error,
                autofill=False,
                verbose=verbose,
            ),
        )
        # Test 6: is the testing function set? if not, raise error or warn or autofill
        return_code = max(
            return_code,
            self.__check_attribute(
                "testing_fn",
                expected_type=callable,
                error=error,
                autofill=False,
                verbose=verbose,
            ),
        )

        return return_code

    def check_checkpointing_ready(self, autofill=True, verbose=True):
        return_code = 0

        return_code = max(
            return_code,
            self.__check_attribute(
                "checkpoint_save_folder",
                expected_type=pathlib.Path,
                error=False,
                autofill=True,
                verbose=verbose,
                default=self.save_folder,
            ),
        )

        # Test 13: is the checkpoint filename filename set? if not, raise error or warn or autofill
        if (
            self.final_result_fname is not None
            and self.checkpoint_save_folder is not None
        ):
            default_checkpoint_fname = f"{self.final_result_fname}_checkpoint_*.pt"
        else:
            default_checkpoint_fname = None
        return_code = max(
            return_code,
            self.__check_attribute(
                "checkpoint_fname",
                expected_type=str,
                error=False,
                autofill=True,
                verbose=False,
                default=default_checkpoint_fname,
            ),
        )

        # Test 15: is the checkpoint frequency set? if not, raise error or warn or autofill
        return_code = max(
            return_code,
            self.__check_attribute(
                "checkpoint_freq",
                expected_type=int,
                error=False,
                autofill=autofill,
                verbose=verbose,
                default=10,
            ),
        )

        # Test 16: is the load checkpoint set? if not, raise error or warn or autofill
        return_code = max(
            return_code,
            self.__check_attribute(
                "do_load_checkpoint",
                expected_type=bool,
                error=False,
                autofill=True,
                verbose=verbose,
                default=True,
            ),
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

        return_code = max(
            return_code, self.check_training_ready(error, autofill, verbose)
        )

        return_code = max(return_code, self.check_testing_ready(error, verbose))

        return_code = max(return_code, self.check_saving_ready())

        return_code = max(return_code, self.check_validation_ready(autofill, verbose))

        return_code = max(
            return_code, self.check_checkpointing_ready(autofill, verbose)
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
                if verbose:
                    warnings.warn(f"Attribute {attr} not set")
                # the return code must not depend on verbosity
                return 1

        # if the type we want to check against is a function, we need to treat it differently
        if expected_type is callable:
            if not callable(getattr(self, attr)):
                if error:
                    raise ValueError(f"Attribute {attr} is not callable")
                if verbose:
                    warnings.warn(f"Attribute {attr} is not callable")
                return 2
        # just standrad type chekcking, error or warn, depends of settings
        elif not isinstance(getattr(self, attr), expected_type):
            if error:
                raise ValueError(
                    f"Attribute {attr} is not of type {expected_type}, its {type(getattr(self, attr))}"
                )
            if verbose:
                warnings.warn(
                    f"Attribute {attr} is not of type {expected_type}, its {type(getattr(self, attr))}"
                )
            return 2
        return 0

    def save_checkpoint(self, epoch):
        """
        This function saves a checkpoint of the model and the optimizer
        """
        if self.checkpoint_save_folder is None:
            raise LIONSolverException("Saving not set: please call set_saving")
        self.model.save_checkpoint(
            self.checkpoint_save_folder.joinpath(
                pathlib.Path(str(self.checkpoint_fname).replace("*", f"{epoch+1:04d}"))
            ),
            epoch + 1,
            self.train_loss,
            self.optimizer,
            self.training_record("checkpoint", epoch + 1),
            dataset=self._training_data_record(),
            random_state=rng.random_state(),
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
            raise LIONSolverException(
                "Save folder not set up, can't save the min-validation model: please call set_saving()."
            )

        self.model.save(
            self.save_folder.joinpath(self.validation_fname),
            epoch=epoch,
            training=self.training_record(
                "best_validation", epoch + 1, value=self.validation_loss[epoch]
            ),
            loss=self.validation_loss[epoch],
            dataset=self._training_data_record(),
        )

    def save_final_results(self, final_result_fname=None, save_folder=None, epoch=None):
        """
        This function saves the final results of the optimization
        """
        if save_folder is not None:
            self.save_folder = save_folder
        if final_result_fname is not None:
            self.final_result_fname = final_result_fname
        if self.save_folder is None or self.final_result_fname is None:
            raise LIONSolverException("Saving not setup: Please call set_saving.")
        if epoch is None:
            epoch = self.current_epoch

        self.model.save(
            self.save_folder.joinpath(self.final_result_fname),
            epoch=epoch,
            training=self.training_record("final", epoch),
            loss=self.train_loss,
            dataset=self._training_data_record(),
        )

    def clean_checkpoints(self):
        """
        This function cleans the checkpoints
        """
        if self.checkpoint_save_folder is None:
            raise LIONSolverException(
                "Saving not setup, unable to find save folder: Please call set_saving"
            )
        if self.checkpoint_fname is None:
            raise LIONSolverException(
                "Checkpointing not setup, can't clear checkpoints: Please call set_checkpointing"
            )
        # this ensures all files with the same extension are removed
        for f in self.checkpoint_save_folder.glob(
            self.checkpoint_fname.replace(".pt", "")
        ):
            f.unlink()

    def test(self):
        if self.check_testing_ready() != 0:
            warnings.warn("Solver not setup to test. Please call set_testing.")
            return np.array([])
        assert self.test_loader is not None
        assert self.testing_fn is not None

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            test_loss = np.array([])
            for data, target in tqdm(self.test_loader):
                output = self._reconstruct(
                    data.to(self.device), self.testing_reconstructor
                )
                test_loss = np.append(
                    test_loss,
                    self.testing_fn(output, target.to(self.device)).cpu().numpy(),
                )
        if was_training:
            self.model.train()

        if self.verbose:
            print(
                f"Testing loss: {test_loss.mean()} - Testing loss std: {test_loss.std()}"
            )

        return test_loss

    def load_checkpoint(self):
        """
        This function loads a checkpoint (if exists)
        """
        if self.checkpoint_save_folder is None:
            raise LIONSolverException("Loading not set. Please call set_loading ")
        if self.checkpoint_fname is None:
            raise LIONSolverException(
                "Checkpointing not set, failed to load checkpoint. Please call set_checkpointing"
            )
        (
            self.model,
            self.optimizer,
            self.current_epoch,
            loaded_train_loss,
            data,
        ) = self.model.load_checkpoint_if_exists(
            self.checkpoint_save_folder.joinpath(self.checkpoint_fname),
            self.model,
            self.optimizer,
            self.train_loss,
            verbose=self.verbose,
        )
        if data is not None:
            self._resume_from(data)
        # Keep the losses of the epochs already done, without shrinking the
        # array that train() allocated for this run.
        loaded_train_loss = np.asarray(loaded_train_loss, dtype=float).ravel()
        if len(self.train_loss) < len(loaded_train_loss):
            self.train_loss = np.pad(
                self.train_loss, (0, len(loaded_train_loss) - len(self.train_loss))
            )
        self.train_loss[: len(loaded_train_loss)] = loaded_train_loss

        # Restore the best validation loss so far, so that the min-validation
        # model saved before the interruption is not overwritten by a worse one.
        if (
            self.validation_fn is not None
            and self.current_epoch > 0
            and self.validation_fname is not None
            and self.validation_loss is not None
            and self.save_folder is not None
            and self.save_folder.joinpath(self.validation_fname).is_file()
        ):
            if len(self.validation_loss) < self.current_epoch:
                self.validation_loss = np.pad(
                    self.validation_loss,
                    (0, self.current_epoch - len(self.validation_loss)),
                    constant_values=np.nan,
                )
            self.validation_loss[
                self.current_epoch - 1
            ] = self.model._read_min_validation(
                self.save_folder.joinpath(self.validation_fname)
            )
            if self.verbose:
                print(
                    f"Loaded checkpoint at epoch {self.current_epoch}. Current best validation value is {self.validation_loss[self.current_epoch-1]}"
                )
        return self.current_epoch

    def _resume_from(self, data):
        """Continue the run a checkpoint was saved from: its random state, the
        hash of its initial weights, and the values the solver computed
        during the run (``solver_params`` that are ``None`` here)."""
        if self.seed is not None and "random_state" in data:
            rng.set_random_state(data["random_state"])
        try:
            record = records.read_training_record(data["file"])
        except (ValueError, OSError):
            return  # a checkpoint saved without a training record
        self.initial_weights_sha256 = getattr(record, "initial_weights_sha256", None)
        saved = record.solver.get("solver_params")
        if not isinstance(saved, dict):
            return
        saved = {k: v for k, v in saved.items() if k not in spec._KEYS}
        for name, value in saved.items():
            if isinstance(value, dict) and "unsaved" in value:
                continue
            if getattr(self.solver_params, name, 0) is None and value is not None:
                setattr(self.solver_params, name, spec.build(value))

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
            self.optimizer.zero_grad()
            batch_loss = self.mini_batch_step(
                data.to(self.device), target.to(self.device)
            )
            batch_loss.backward()
            epoch_loss += batch_loss.item()
            self.optimizer.step()
        return epoch_loss / len(self.train_loader)

    def epoch_step(self, epoch):
        """
        This function is responsible for performing a single epoch of the optimization.
        """
        self.train_loss[epoch] = self.train_step()
        # actually make sure we're doing validation
        if self.validation_loss is not None and (epoch + 1) % self.validation_freq == 0:
            # validation leaves training's random numbers as they were (a
            # DataLoader draws one when it starts), so the trained model doesn't
            # depend on how often it was validated
            cuda = [self.device] if torch.device(self.device).type == "cuda" else []
            with torch.random.fork_rng(devices=cuda):
                self.validation_loss[epoch] = self.validate()
            if self.verbose:
                print(
                    f"Epoch {epoch+1} - Training loss: {self.train_loss[epoch]} - Validation loss: {self.validation_loss[epoch]}"
                )

            if self._saves_min_validation() and self._is_best_validation(epoch):
                self.save_validation(epoch)
        elif self.verbose:
            print(f"Epoch {epoch+1} - Training loss: {self.train_loss[epoch]}")

    def train(self, n_epochs):
        """
        This function is responsible for performing the optimization.
        """
        assert n_epochs > 0, "Number of epochs must be a positive integer"
        # Make sure all parameters are set
        self.check_training_ready()
        self.n_epochs = n_epochs

        # Allocate the loss arrays before loading a checkpoint, so that the
        # loaded values (including the min validation loss) are kept. Losses of
        # epochs already trained by this solver (an earlier train() call) are kept.
        self.train_loss = self._resized(self.train_loss, n_epochs)
        if self.validation_loader is not None and self.check_validation_ready() == 0:
            # NaN: not validated (yet)
            self.validation_loss = self._resized(
                self.validation_loss, n_epochs, fill=np.nan
            )
            if not self._saves_min_validation():
                warnings.warn(
                    "Validation runs, but the model with the lowest validation loss "
                    "is not saved: call set_saving() to save it."
                )
        else:
            self.validation_loss = None

        # Fills in checkpointing defaults (frequency, file name) when possible.
        self.check_checkpointing_ready(verbose=False)
        do_checkpoint = (
            self.checkpoint_freq is not None
            and self.checkpoint_save_folder is not None
            and self.checkpoint_fname is not None
        )

        if self.do_load_checkpoint and do_checkpoint:
            print("Loading checkpoint...")
            self.current_epoch = self.load_checkpoint()

        if self.current_epoch == 0:  # a new run
            if self.seed is not None:
                rng.seed_all(self.seed)
                if self.verbose:
                    print(
                        f"Training seed: {self.seed} (the solver's seed argument; "
                        "seed=None for an unseeded run)"
                    )
            self.initial_weights_sha256 = records.weights_sha256(self.model)

        self.model.train()
        # train loop
        while self.current_epoch < n_epochs:
            print(f"Training epoch {self.current_epoch + 1}")
            self.epoch_step(self.current_epoch)

            if do_checkpoint and (self.current_epoch + 1) % self.checkpoint_freq == 0:
                self.save_checkpoint(self.current_epoch)

            self.current_epoch += 1

    def _resized(
        self, losses: Optional[np.ndarray], n_epochs: int, fill: float = 0.0
    ) -> np.ndarray:
        """``fill`` for n_epochs, keeping the losses of the epochs already trained."""
        resized = np.full(n_epochs, fill)
        if losses is not None:
            n = min(len(losses), n_epochs, self.current_epoch)
            resized[:n] = losses[:n]
        return resized

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

        # Models that need autograd in their forward pass (e.g. gradient-step
        # denoisers) enable it themselves, so no_grad is safe for all models.
        with torch.no_grad():
            validation_loss = self._validation_loop()

        if self.verbose:
            print(
                f"Validation loss: {validation_loss.mean()} - Validation loss std: {validation_loss.std()}"
            )

        # return to train if it was in train
        if status:
            self.model.train()

        return np.mean(validation_loss)

    def _validation_loop(self):
        """
        This function performs the validation loop.
        """
        validation_loss = np.array([])
        for data, targets in tqdm(self.validation_loader):
            outputs = self._reconstruct(
                data.to(self.device), self.validation_reconstructor
            )
            # same argument order as the training loss: (output, target)
            validation_loss = np.append(
                validation_loss,
                self.validation_fn(outputs, targets.to(self.device))
                .detach()
                .cpu()
                .numpy(),
            )
        return validation_loss

    def get_model(self) -> LIONmodel:
        """
        Returns the model used by the solver.
        """
        return self.model

    @abstractmethod
    def mini_batch_step(self, sino_batch, target_batch) -> torch.Tensor:
        """
        This function should perform a single step of the optimization and return the loss
        """
        pass
