# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


# %% This is a base class for LION models.
#
# All classes must derive from this one.
# It definest a bunch of auxiliary functions
#

# %% Imports

# You will want to import LIONParameter, as all models must save and use Parameters.
from enum import Enum
from typing import Optional
from LION.utils.parameter import LIONParameter
import LION.utils.spec as spec
from LION.data_loaders.data_record import is_data_record
from LION.optimizers.training_record import is_training_record
from LION.utils import environment

# We will need utilities
import LION.utils.utils as ai_utils
from LION.utils.normaliser import Normalisation
from LION.utils.device import default_device
from LION.utils.tomosipo_import import import_tomosipo
from LION.exceptions.exceptions import NoDataException

# (optional) Given this is a tomography library, it is likely that you will want to load geometries of the tomogprahic problem you are solving, e.g. a ct_geometry
import LION.CTtools.ct_geometry as ct
import LION.CTtools.ct_utils as ct_utils

# some numerical standard imports, e.g.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# imports related to class
from abc import ABC, abstractmethod, ABCMeta

# Some other imports
import json
import pickle
import warnings
from pathlib import Path
import inspect


class ModelInputType(int, Enum):
    SINOGRAM = 0
    IMAGE = 1


# Class for Model parameters that should be true for all models
class LIONModelParameter(LIONParameter):
    def __init__(self, **kwargs):
        # Set defaults before applying kwargs, so values passed in (e.g. when
        # loading from file) are not overwritten.
        self.model_input_type = None
        self.normalisator = None
        super().__init__(**kwargs)


def as_model_parameter(params: LIONParameter) -> LIONModelParameter:
    """The LIONModelParameter for ``params``, with ``model_input_type`` as a
    ModelInputType.

    JSON files have no classes: a saved parameter loads back as a plain
    LIONParameter, nested ones included (e.g. the backbone parameters of GSD),
    and ``model_input_type`` as an int. Models call this on their parameters,
    so they accept them however they were made.
    """
    if not isinstance(params, LIONModelParameter):
        params = LIONModelParameter(**vars(params))
    if params.model_input_type is not None:
        params.model_input_type = ModelInputType(params.model_input_type)
    return params


def _plain_numbers(value):
    """A loss or epoch as Python numbers (a float, an int or a list of them),
    which torch's safe loader reads; other values as they are."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, (np.ndarray, np.generic, list, tuple)):
        try:
            return np.asarray(value).tolist()
        except (TypeError, ValueError):
            return value
    return value


def _safe_globals():
    """What torch's safe loader may rebuild from a model's ``.pt``, besides
    tensors and Python values: LION's parameter objects, paths and numpy
    arrays, which some files hold. Rebuilding these runs no code from the file.
    """
    from numpy import dtypes

    try:
        from numpy._core import multiarray
    except ImportError:  # numpy 1
        from numpy.core import multiarray
    import pathlib

    allowed = [
        LIONParameter,
        LIONModelParameter,
        ModelInputType,
        ct.Geometry,
        pathlib.Path,
        pathlib.PosixPath,
        pathlib.WindowsPath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
        np.ndarray,
        np.dtype,
        multiarray._reconstruct,
        multiarray.scalar,
    ]
    allowed += [getattr(dtypes, n) for n in dir(dtypes) if n.endswith("DType")]
    # numpy's functions as files written with numpy 1 name them
    allowed += [
        (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        (multiarray.scalar, "numpy.core.multiarray.scalar"),
    ]
    return allowed


class LIONmodel(nn.Module, ABC):
    """
    Base class for all models in the toolbox,
    """

    # Initialization of the models should have only "LIONParameter()" classes. These should be "topic-wise", with at minimum 1 parameter object being passed.
    # e.g. a Unet will have only 1 parameter (model_parameters), but the Learned Primal Dual will have 2, one for the model parameters and another one
    # for the geometry parameters of the inverse problem.
    def __init__(
        self,
        model_parameters: Optional[LIONModelParameter],  # model parameters
        geometry: Optional[
            ct.Geometry
        ] = None,  # (optional) if your model uses an operator, you may need its parameters. e.g. ct geometry parameters for tomosipo operators
    ):
        super().__init__()  # Initialize parent classes.
        __metaclass__ = ABCMeta  # make class abstract.

        if geometry is not None and not isinstance(geometry, ct.Geometry):
            raise TypeError(
                f"Expected geometry to be of type Geometry or None, but got {type(geometry).__name__}. "
                "If you passed positional arguments to the model, please verify their order matches the model's __init__ signature."
            )

        # A Geometry is a LIONParameter too: catch it passed in the wrong place.
        if isinstance(model_parameters, ct.Geometry) or not (
            model_parameters is None or isinstance(model_parameters, LIONParameter)
        ):
            raise TypeError(
                f"Expected model_parameters to be of type LIONModelParameter or None, but got {type(model_parameters).__name__}. "
                "Ensure you are not accidentally passing a Geometry object as model_parameters."
            )

        if model_parameters is None:
            model_parameters = self.default_parameters()
        else:
            # A name the model doesn't know is not used: likely a typo
            spec.warn_unknown_names(
                type(self).__name__, model_parameters, self.default_parameters()
            )
        model_parameters = as_model_parameter(model_parameters)
        # Pass all relevant parameters to internal storage.
        self.geometry = geometry
        self.model_parameters = model_parameters
        # Set by set_normalisation(); None means inputs are used as they are.
        self.normalisator = None

    # This should return the parameters from the paper the model is from
    @staticmethod
    @abstractmethod  # crash if not defined in derived class
    def default_parameters(mode="ct") -> LIONModelParameter:
        pass

    # makes operator and make it pytorch compatible.
    def _make_operator(self):
        # if self.model_parameters.mode.lower() != "ct":
        #     raise NotImplementedError("Only CT operators supported")
        if hasattr(self, "geometry") and self.geometry is not None:
            to_autograd = import_tomosipo(
                "tomosipo.torch_support", f"{type(self).__name__}._make_operator"
            ).to_autograd

            self.op = ct_utils.make_operator(self.geometry)
            self.A = to_autograd(self.op, num_extra_dims=1)
            self.AT = to_autograd(self.op.T, num_extra_dims=1)
        else:
            raise AttributeError("Can't make operator without geometry parameters.")

    # All classes should have this method, just change the amount of Parameters it returns of you have more/less
    def get_parameters(self):
        if self.geometry is not None:
            return self.model_parameters, self.geometry
        else:
            return self.model_parameters

    def get_input_type(self) -> ModelInputType:
        return self.model_parameters.model_input_type

    def set_normalisation(
        self, normalisation_type="sample", dataset=None, data_range=None
    ):
        if self.get_input_type() == ModelInputType.SINOGRAM:
            warnings.warn(
                """Normalization will not be carried out on this model,
                as it takes inputs in the measurement domain. 
                As such inputs cannot be normalized in the image domain before being passed to the model.
                In such a case, normalization should be implemented within the model itself"""
            )
        if normalisation_type == "dataset" and dataset is None:
            raise NoDataException("dataloader not set: Please call set_training")

        self.normalisator = Normalisation(normalisation_type, dataset, data_range)

    def normalise(self, x):
        """Normalise ``x`` as set by ``set_normalisation()`` (identity if unset)."""
        if getattr(self, "normalisator", None) is None:
            return x
        return self.normalisator.normalise(x)

    def unnormalise(self, x, target=None):
        """Undo ``normalise()`` (identity if no normalisation is set)."""
        if getattr(self, "normalisator", None) is None:
            return x
        return self.normalisator.unnormalise(x, target)

    # All classes should have this method. This is the example for Learned Primal Dual.
    # You can obtain this exact text from Google Scholar's page of the paper.
    @staticmethod
    def cite(cite_format="MLA"):
        print("cite not implemented for selected method")
        pass

    #     if cite_format == "MLA":
    #         print("Adler, Jonas, and Ozan Öktem.")
    #         print('"Learned primal-dual reconstruction."')
    #         print("\x1B[3mIEEE transactions on medical imaging \x1B[0m")
    #         print("37.6 (2018): 1322-1332.")
    #     elif cite_format == "bib":
    #         string = """
    #         @article{adler2018learned,
    #         title={Learned primal-dual reconstruction},
    #         author={Adler, Jonas and {\"O}ktem, Ozan},
    #         journal={IEEE transactions on medical imaging},
    #         volume={37},
    #         number={6},
    #         pages={1322--1332},
    #         year={2018},
    #         publisher={IEEE}
    #         }"""
    #         print(string)
    #     else:
    #         raise AttributeError(
    #             'cite_format not understood, only "MLA" and "bib" supported'
    #         )

    # This shoudl save all relevant information to completely reproduce models
    def save(self, fname, **kwargs):
        """
        Saves model given a filename.
        While its not enforced, the following Parameters are expected from kwargs:
        - 'dataset' : the record of the data (``LION.data_loaders.data_record``,
          e.g. ``experiment.data_record("train")``), saved and read back as it is;
          or a LIONParameter describing the dataset creation and handling.
        - 'training': the record of the training (``LION.optimizers.training_record``,
          written by the solvers), saved and read back as it is; or a
          LIONParameter describing the training algorithm and procedures
        - 'geometry': If the model itself has no scan geometry parameter, but the dataset was created with some geometry

        If you want to save the model for training later (i.e. checkpoiting), use save_checkpoint()
        """
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)

        # Create dict of saved data:
        ##########################
        dic = {"model_state_dict": self.state_dict()}

        # Parse kwargs
        ################
        dataset_params = None
        if "dataset" in kwargs:
            dataset_params = kwargs.pop("dataset")

        else:
            warnings.warn(
                "\nExpected 'dataset' parameter! Only ignore if you really don't have it.\n"
            )
        training = None
        if "training" in kwargs:
            training = kwargs.pop("training")

        else:
            warnings.warn(
                "\nExpected 'training' parameter! Only ignore if there has been no training.\n"
            )

        # Plain numbers, so the file loads with torch.load(weights_only=True)
        if "loss" in kwargs:
            dic["loss"] = _plain_numbers(kwargs.pop("loss"))
        if "epoch" in kwargs:
            dic["epoch"] = _plain_numbers(kwargs.pop("epoch"))
        if "optimizer" in kwargs:
            dic["optimizer_state_dict"] = kwargs.pop("optimizer")
        if "random_state" in kwargs:  # LION.utils.rng.random_state(), for resuming
            dic["random_state"] = kwargs.pop("random_state")

        # (optional)
        geometry = None
        given_geometry = None
        if "geometry" in kwargs:
            geometry = given_geometry = kwargs.pop("geometry")
        elif hasattr(self, "geometry") and self.geometry is not None:
            geometry = self.geometry
        else:
            warnings.warn(
                "Expected 'geometry' parameter! Only ignore if tomographic reconstruction was not part of the model."
            )

        if kwargs:  # if not empty yet
            raise ValueError(
                "The following parameters are not understood: "
                + str(list(kwargs.keys()))
            )
        # The model's metadata: the spec that builds it again (see
        # LION.utils.spec), and what is known of its data and training.
        metadata = {"lion_format": spec.FORMAT_VERSION}
        # We should always save models with the git hash they were created. Models may change, and if loading at some point breaks
        # we need to at least know exactly when the model was saved, to at least be able to reproduce.
        try:
            metadata["commit_hash"] = ai_utils.get_git_revision_hash()
        except Exception:
            warnings.warn("\nCould not get git hash.\n")
        metadata["model"] = spec.to_spec(self)
        # LION's code, the main packages and the machine (LION.utils.environment)
        first = next(self.parameters(), None)
        metadata["environment"] = environment.environment_record(
            None if first is None else first.device
        ).serialize()
        # The normalisation it was trained with: reconstructing needs the same
        if getattr(self, "normalisator", None) is not None:
            metadata["normalisation"] = self.normalisator.record()
        # A geometry given here describes the data (e.g. for a model without one)
        if given_geometry is not None:
            metadata["geometry"] = spec.encode(given_geometry)
        if is_data_record(dataset_params):
            # already JSON (LION.data_loaders.data_record), kept as it is
            metadata["dataset_params"] = spec.read(dataset_params)  # as JSON
        elif dataset_params is not None:
            metadata["dataset_params"] = spec.encode(dataset_params)
        if is_training_record(training):
            metadata["training"] = spec.read(training)  # as JSON
        elif training is not None:
            metadata["training"] = spec.encode(training)

        # Do the save:
        with open(fname.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        torch.save(dic, fname.with_suffix(".pt"))

    # Mandatory function, saves model for training
    def save_checkpoint(self, fname, epoch, loss, optimizer, training_param, **kwargs):
        """
        This is like save, but saves a checkpoint of the model.
        Its essentailly a wrapper of save() with mandatory values
        """

        assert isinstance(optimizer, torch.optim.Optimizer)
        self.save(
            fname,
            epoch=epoch,
            loss=loss,
            optimizer=optimizer.state_dict(),
            training=training_param,
            **kwargs,
        )

    @staticmethod
    def _load_data(fname, *, weights_only=True, supress_warnings=False):
        """The contents of a model's ``.pt``.

        ``weights_only=True`` (default) reads it with torch's safe loader, which
        runs no code from the file: tensors, numbers, strings and containers,
        plus the LION parameter objects and numpy arrays some files hold (see
        ``_safe_globals``). ``weights_only=False`` reads any pickle: only for
        files you trust.
        """
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)
        # Check compatible suffixes:
        if fname.with_suffix(".pt").is_file():
            fname = fname.with_suffix(".pt")
        elif fname.with_suffix(".pth").is_file():
            fname = fname.with_suffix(".pth")
        # Load the actual pythorch saved data
        if weights_only:
            try:
                with torch.serialization.safe_globals(_safe_globals()):
                    data = torch.load(
                        fname, map_location=default_device(), weights_only=True
                    )
            except pickle.UnpicklingError as error:
                raise pickle.UnpicklingError(
                    f"{fname} holds objects LION does not load safely. If you trust "
                    "the file, load it with weights_only=False (it can run code)."
                ) from error
        else:
            data = torch.load(fname, map_location=default_device(), weights_only=False)
        if isinstance(data.get("loss"), list):
            data["loss"] = np.asarray(data["loss"], dtype=float)
        if len(data) > 1 and not supress_warnings:
            # this should be only 1 thing, but you may be loading a checkpoint or may have saved more data.  Its OK, but we warn.
            warnings.warn(
                "\nSaved file contains more than 1 object, but only model_state_dict is being loaded.\n Call load_checkpoint() to load checkpointed model.\n"
            )
        return data

    @classmethod
    def _read_model_file(cls, fname, allow=()):
        """The class to build, its keyword arguments and the file's metadata
        (``options``), from a model's ``.json`` (its path, or the dict read
        from it, or the model's spec alone).

        The model is built as ``cls``, unless ``cls`` is ``LIONmodel``: then as
        the class the file names, which must be in a module ``LION.utils.spec``
        allows (LION's own, or one given in ``allow``).
        """
        if isinstance(fname, dict):
            saved, where = fname, "the metadata given"
            if spec.TARGET in saved:  # a model's spec alone
                saved = {"lion_format": spec.FORMAT_VERSION, "model": saved}
            elif "lion_format" not in saved:
                raise ValueError(
                    "metadata without 'lion_format' (an older file): "
                    "give its file instead"
                )
        else:
            fname = Path(fname)
            where = fname.with_suffix(".json")
            saved = spec.read(fname)
            if "lion_format" not in saved:
                return cls._read_unversioned_model_file(fname)
        if saved["lion_format"] > spec.FORMAT_VERSION:
            raise ValueError(
                f"{where} has metadata format {saved['lion_format']}; "
                f"this version of LION reads up to {spec.FORMAT_VERSION}. Update LION."
            )

        path, kwargs = spec.build_arguments(saved["model"], allow)
        target = spec.target_class(path, cls, LIONmodel, allow)

        options = LIONParameter()
        options.lion_format = saved["lion_format"]
        if "commit_hash" in saved:
            options.commit_hash = saved["commit_hash"]
        if "environment" in saved:
            options.environment = LIONParameter(**saved["environment"])
            difference = environment.code_differences(saved["environment"])
            if difference is not None:
                warnings.warn(
                    f"\n{where} was saved with other LION code: {difference}. "
                    "Loading may fail, or the model may behave differently.\n"
                )
        options.model_name = path.rsplit(".", 1)[-1]
        if saved.get("normalisation") is not None:
            options.normalisation = saved["normalisation"]
        options.model_parameters = kwargs.get("model_parameters")
        if "geometry" in saved:
            options.geometry = spec.build(saved["geometry"], allow)
        elif kwargs.get("geometry") is not None:
            options.geometry = kwargs["geometry"]
        for key in ("dataset_params", "training"):
            if is_data_record(saved.get(key)) or is_training_record(saved.get(key)):
                # a record, whose specs stay dicts: making its components
                # again (e.g. the dataset, which reads the data) is on request
                setattr(options, key, LIONParameter(**saved[key]))
            elif key in saved:
                setattr(options, key, spec.build(saved[key], allow))
        return target, kwargs, options

    @classmethod
    def _read_unversioned_model_file(cls, fname):
        """``_read_model_file`` for a ``.json`` without ``lion_format``: the model
        parameters and geometry as LIONParameters, and the class only by name."""
        if cls is LIONmodel:
            raise ValueError(
                f"{fname.with_suffix('.json')} names its model class only by name: load "
                "it with that class, e.g. FBPConvNet.load(...)"
            )
        options = LIONParameter()
        options.load(fname.with_suffix(".json"))
        # JSON loads nested parameters back as plain LIONParameter
        if isinstance(getattr(options, "model_parameters", None), LIONParameter):
            options.model_parameters = as_model_parameter(options.model_parameters)
        if hasattr(options, "geometry"):
            options.geometry = ct.Geometry.init_from_parameter(options.geometry)
        # Error check
        ################################
        # Check if model has been changed since save.
        # if not hasattr(options, "commit_hash") and not supress_warnings:
        #     warnings.warn(
        #         "\nNo commit hash found. This model was not saved with the standard AItomotools function and it will likely fail to load.\n"
        #     )
        # else:
        #     curr_commit_hash = ai_utils.get_git_revision_hash()
        #     curr_class_path = cls.current_file()
        #     curr_aitomomodel_path = Path(__file__)
        #     if (
        #         ai_utils.check_if_file_changed_git(
        #             curr_class_path, options.commit_hash, curr_commit_hash
        #         )
        #         or ai_utils.check_if_file_changed_git(
        #             curr_aitomomodel_path, options.commit_hash, curr_commit_hash
        #         )
        #         and not supress_warnings
        #     ):
        #         warnings.warn(
        #             f"\nThe code for the model has changed since it was saved, loading it may fail. This model was saved in {options.commit_hash}\n"
        #         )
        # Check if model name matches the one that is loading it
        if options.model_name != cls.__name__:
            warnings.warn(
                f"\nSaved model is from a class with a different name than current, likely load will fail. \nCurrent class name: {cls.__name__}, Saved model class name: {options.model_name}\n"
            )
        # Some models need geometry, some others not.
        kwargs = {"model_parameters": options.model_parameters}
        sig = inspect.signature(cls.__init__)
        if hasattr(options, "geometry") and "geometry" in sig.parameters:
            kwargs["geometry"] = options.geometry
        return cls, kwargs, options

    @classmethod
    def instantiate(cls, source, *, allow=()):
        """The model a saved file describes, with its saved parameters and new
        (untrained) weights. ``load()`` is this plus the saved weights.

        ``source``: a model's file (``"model"``, ``"model.json"`` or
        ``"model.pt"``: the ``.json`` is read), the dict read from it, or the
        model's spec. Called on a model class it makes that class; called on
        ``LIONmodel``, the class the file names (``allow`` as in ``load``).
        """
        target, kwargs, _ = cls._read_model_file(source, allow)
        return target(**kwargs)

    @classmethod
    def load(cls, fname, *, weights_only=True, supress_warnings=False, allow=()):
        """
        Function that loads a model from memory: built from its ``.json``, with
        the weights of its ``.pt``.

        Called on a model class (``FBPConvNet.load(f)``) it builds that class;
        called on ``LIONmodel`` (``LIONmodel.load(f)``) it builds the class the
        file names. ``allow``: module prefixes of your own model classes (see
        ``LION.utils.spec.build``). ``weights_only=False`` reads a ``.pt``
        that holds other objects, which can run code: only for files you
        trust. Returns ``(model, options, data)``; the model keeps its file
        as ``model.source_file`` (without suffix).
        """
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)

        target, kwargs, options = cls._read_model_file(fname, allow)

        # load data
        data = LIONmodel._load_data(
            fname, weights_only=weights_only, supress_warnings=supress_warnings
        )
        model = target(**kwargs)

        # Load the data into the model we created.
        model.to(default_device())
        model.load_state_dict(data.pop("model_state_dict"))
        _restore_normalisation(model, options)
        # where it came from, so records of methods using it can name the file
        model.source_file = fname.with_suffix("")

        return model, options, data

    @classmethod
    def load_checkpoint(cls, fname, *, weights_only=True, allow=()):
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)

        target, kwargs, options = cls._read_model_file(fname, allow)
        # load data
        data = LIONmodel._load_data(
            fname, weights_only=weights_only, supress_warnings=True
        )
        model = target(**kwargs)
        # Load the data into the model we created.
        model.to(default_device())
        model.load_state_dict(data.pop("model_state_dict"))
        _restore_normalisation(model, options)

        return model, options.unpack(), data

    @classmethod
    def _current_file(cls):
        import sys

        module = sys.modules[cls.__module__]
        fname = Path(module.__file__)
        # This will be in the install path, so lets get a relative path. Assuming user here will be working on AItomotools folder, which they may not be.
        parts = fname.resolve().parts[fname.resolve().parts.index("models") - 1 :]
        return Path(*parts)

    @staticmethod
    def final_file_exists(fname, stop_code=False):
        if isinstance(fname, str):
            fname = Path(fname)
        exists = fname.is_file()
        if stop_code and exists:
            print("Final version found, no need to loop further, exiting")
            exit()
        return exists

    @classmethod
    def load_checkpoint_if_exists(
        cls, fname, model, optimiser, total_loss, verbose=True
    ):
        """Load the latest checkpoint matching ``fname`` (a glob pattern) into
        ``model`` and ``optimiser``, in place.

        The weights are loaded into the *given* model (rather than into a new
        instance) so that ``optimiser`` keeps optimising the parameters of the
        model that is returned. Returns ``(model, optimiser, start_epoch,
        total_loss, data)``, ``data["file"]`` being the checkpoint read; if no checkpoint exists the inputs are returned
        unchanged with ``start_epoch=0`` and ``data=None``.
        """
        if isinstance(fname, str):
            fname = Path(fname)
        # glob() already returns paths that include fname.parent
        checkpoints = sorted(fname.parent.glob(fname.name))
        if not checkpoints:
            if verbose:
                print(f"checkpoint {fname} not found, failed to load.")
            return model, optimiser, 0, total_loss, None

        data = cls._load_data(checkpoints[-1], supress_warnings=True)
        data["file"] = checkpoints[-1]
        model.load_state_dict(data.pop("model_state_dict"))
        optimiser.load_state_dict(data["optimizer_state_dict"])
        model.train()
        return model, optimiser, data["epoch"], data["loss"], data

    @staticmethod
    def _read_min_validation(filename):
        """
        Given a filename for saved models, reads the validaton loss and returns it.
        This is useful when both checkpointing and validation are done in the same training, and the training partially stops.
        One can load the checkpoint, but the minimum validation loss is not saved in the checkpoint, so this function reads it from the filename,
        which should be the minimum validation model.
        """
        if isinstance(filename, str):
            filename = Path(filename)
        data = LIONmodel._load_data(filename, supress_warnings=True)
        loss = data["loss"]
        if type(loss) is np.ndarray:
            if len(loss) > 1:
                warnings.warn(
                    "More than one loss found in file, which suggests that it was not a minimum validation file. Returning last loss."
                )
                loss = loss[-1]
            else:
                loss = loss[0]
        return loss


def _restore_normalisation(model, options):
    """The normalisation a model file records (the one it was trained with)."""
    recorded = getattr(options, "normalisation", None)
    if recorded is not None:
        if not isinstance(recorded, dict):
            recorded = vars(recorded)
        model.normalisator = Normalisation.from_record(recorded)
