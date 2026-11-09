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
from LION.utils.parameter import (
    DEFAULT_RECONSTRUCTION,
    LIONParameter,
    OperatorParameters,
    resolve_reconstruction,
)
import LION.utils.spec as spec
from LION.data_loaders.data_record import is_data_record
from LION.optimizers.training_record import is_training_record, weights_sha256
from LION.utils import environment, rng

from LION.utils.normaliser import Normalisation
from LION.utils.device import default_device

# (optional) Given this is a tomography library, it is likely that you will want to load geometries of the tomogprahic problem you are solving, e.g. a ct_geometry

# some numerical standard imports, e.g.
import numpy as np
import torch
import torch.nn as nn

# imports related to class
from abc import ABC, abstractmethod

# Some other imports
import json
import pickle
import warnings
from pathlib import Path


class ModelInputType(int, Enum):
    SINOGRAM = 0
    IMAGE = 1


# Class for Model parameters that should be true for all models
class LIONModelParameter(LIONParameter):
    def __init__(self, **kwargs):
        # Set defaults before applying kwargs, so values passed in (e.g. when
        # loading from file) are not overwritten.
        self.model_input_type = None
        # How the model's initial reconstruction is made from measurements,
        # f(sinogram, operator) (set_initial_reconstruction). "default": the
        # operator's default reconstruction (FDK for CT), replaced by that
        # function once the model has its operator.
        self.initial_reconstruction = "default"
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
    tensors and Python values: paths and numpy arrays (e.g. the random state
    a checkpoint holds). Rebuilding these runs no code from the file.
    """
    from numpy import dtypes

    try:
        from numpy._core import multiarray
    except ImportError:  # numpy 1
        from numpy.core import multiarray
    import pathlib

    allowed = [
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
    Base class for all models in the toolbox.

    A model that uses an operator takes ``(operator_params,
    model_parameters=None)``; one that doesn't takes only
    ``(model_parameters=None)``.

    ``operator_params``: the parameters of the operator the model uses (an
    ``OperatorParameters``). **For CT, it's the geometry**: ``LPD(geometry)``.
    The model makes its operator from them (``self.op``): its domain is the
    image, its range the measurements (the sinogram, in CT).

    ``model_parameters``: the model's own parameters; None gives
    ``default_parameters()``.
    """

    def __init__(
        self,
        model_parameters: Optional[LIONModelParameter],  # model parameters
        # (optional) the parameters of the operator the model uses (an
        # OperatorParameters, e.g. a CT Geometry), which make it
        operator_params: Optional[OperatorParameters] = None,
    ):
        super().__init__()  # Initialize parent classes.
        # torch's random state before the model's layers draw their initial
        # weights: made again from it, the model has the same initial weights
        # (instantiate(..., initial_weights=True)); saved with the model
        self.initial_random_state = torch.get_rng_state()

        if operator_params is not None and not isinstance(
            operator_params, OperatorParameters
        ):
            raise TypeError(
                "Expected operator_params to be operator parameters "
                f"(OperatorParameters, e.g. a CT Geometry) or None, but got {type(operator_params).__name__}. "
                "If you passed positional arguments to the model, please verify their order matches the model's __init__ signature."
            )

        # Operator parameters are LIONParameters too: catch them passed in the
        # wrong place.
        if isinstance(model_parameters, OperatorParameters) or not (
            model_parameters is None or isinstance(model_parameters, LIONParameter)
        ):
            raise TypeError(
                f"Expected model_parameters to be of type LIONModelParameter or None, but got {type(model_parameters).__name__}. "
                "Ensure you are not accidentally passing operator parameters (e.g. a Geometry) as model_parameters."
            )

        if model_parameters is None:
            model_parameters = self.default_parameters()
        else:
            # A name the model doesn't know is not used: likely a typo
            spec.warn_unknown_names(
                type(self).__name__,
                model_parameters,
                self.default_parameters(),
                # set by set_normalisation(), not a default
                ignore=("normalisation",),
            )
        model_parameters = as_model_parameter(model_parameters)
        # Pass all relevant parameters to internal storage.
        self.operator_params = operator_params
        self.model_parameters = model_parameters
        # The normalisation the model is trained and used with (a parameter,
        # written by set_normalisation()); None: inputs are used as they are.
        recorded = getattr(model_parameters, "normalisation", None)
        self.normalisator = (
            None if recorded is None else Normalisation.from_record(recorded)
        )

    # This should return the parameters from the paper the model is from
    @staticmethod
    @abstractmethod  # crash if not defined in derived class
    def default_parameters(mode="ct") -> LIONModelParameter:
        pass

    # makes operator and make it pytorch compatible.
    def _make_operator(self, autograd: bool = False):
        """Make the model's operator, ``self.op``, from its parameters. With
        ``autograd=True``, for models whose gradients go through the operator
        (e.g. LPD), also ``self.A`` and ``self.AT``: the operator and its
        adjoint as torch autograd functions (``Operator.autograd``). An
        operator without them can't be used by such a model: it raises here."""
        if getattr(self, "operator_params", None) is None:
            raise AttributeError("Can't make operator without operator parameters.")
        self.op = self.operator_params.make_operator()
        if autograd:
            try:
                self.A, self.AT = self.op.autograd()
            except NotImplementedError as error:
                raise NotImplementedError(
                    f"{type(self).__name__}'s gradients go through its operator: {error}"
                ) from None
        self._resolve_initial_reconstruction(self.op)

    # All classes should have this method, just change the amount of Parameters it returns of you have more/less
    def get_parameters(self):
        if self.operator_params is not None:
            return self.model_parameters, self.operator_params
        else:
            return self.model_parameters

    def get_input_type(self) -> ModelInputType:
        return self.model_parameters.model_input_type

    def takes_noise_level(self) -> bool:
        """Whether the model is called with the noise level of its input,
        ``model(x, noise_level=...)``: its parameters' ``use_noise_level``
        (e.g. DRUNet's). A model that decides otherwise (e.g. GSD, from its
        denoiser) says so by overriding this."""
        return bool(getattr(self.model_parameters, "use_noise_level", False))

    def set_normalisation(
        self, normalisation_type="sample", dataset=None, data_range=None
    ):
        """Normalise the model's inputs (and map its outputs back), in
        training and whenever it is used: "sample" (each image's own range),
        "dataset" (the targets' range over ``dataset``, computed now),
        "custom" (``data_range``) or "none". It is one of the model's
        parameters (``model_parameters.normalisation``: its type, and range
        for "dataset" and "custom"), so it is saved and made again with the
        model.

        A model that takes measurements (sinograms) gets them mapped as the
        images are, by linearity (``Normalisation.normalise_measurements``),
        and its output is mapped back; its operator is unchanged. "sample"
        isn't defined for such models."""
        if (
            self.get_input_type() == ModelInputType.SINOGRAM
            and normalisation_type == "sample"
        ):
            raise ValueError(
                f"{type(self).__name__} takes measurements: a 'sample' "
                "normalisation isn't defined for it (an image's range isn't "
                "known before reconstructing); use 'dataset' or 'custom'"
            )
        if normalisation_type == "dataset" and dataset is None:
            raise ValueError('a dataset is needed for the "dataset" normalisation')

        self.normalisator = Normalisation(normalisation_type, dataset, data_range)
        self.model_parameters.normalisation = LIONParameter(
            **self.normalisator.record()
        )

    def set_initial_reconstruction(self, reconstruction):
        """How the model's initial reconstruction is made from the
        measurements, ``reconstruction(sinogram, operator)``: the input of a
        model that takes images (``LION.optimizers.LIONsolver.model_forward``),
        or where a model that takes measurements and starts from a
        reconstruction (e.g. LPD) starts, in training and whenever it is used.
        It is one of the model's parameters
        (``model_parameters.initial_reconstruction``), so it is saved and made
        again with the model: a LION function or a ``functools.partial`` of
        one, e.g. ``functools.partial(sirt, num_iterations=50)``.
        ``"default"`` (or None) is the operator's default reconstruction
        (``Operator.default_reconstruction_fn``, e.g. ``fdk`` for CT), replaced
        by that function once the model has its operator, so it is saved as
        the function that was used."""
        if reconstruction is None or reconstruction == DEFAULT_RECONSTRUCTION:
            self.model_parameters.initial_reconstruction = DEFAULT_RECONSTRUCTION
            if getattr(self, "op", None) is not None:
                self._resolve_initial_reconstruction(self.op)
            return
        spec.encode(reconstruction)  # raises if it can't be saved
        self.model_parameters.initial_reconstruction = reconstruction

    def _resolve_initial_reconstruction(self, op):
        """Replace a ``"default"`` initial reconstruction by ``op``'s default
        reconstruction function. An operator without one leaves it "default",
        until the model is used with one that has one."""
        value = self.model_parameters.initial_reconstruction
        if value != DEFAULT_RECONSTRUCTION:
            return
        try:
            value = resolve_reconstruction(value, op, "initial_reconstruction")
        except NotImplementedError:
            return
        self.model_parameters.initial_reconstruction = value

    def initial_reconstruction(self, sino, op):
        """The model's initial reconstruction of the measurements ``sino``
        (``model_parameters.initial_reconstruction``; "default": ``op``'s
        default reconstruction, e.g. FDK for CT)."""
        self._resolve_initial_reconstruction(op)
        return resolve_reconstruction(
            self.model_parameters.initial_reconstruction, op, "initial_reconstruction"
        )(sino, op)

    def normalise(self, x):
        """Normalise ``x`` as set by ``set_normalisation()`` (identity if unset)."""
        if getattr(self, "normalisator", None) is None:
            return x
        return self.normalisator.normalise(x)

    def unnormalise(self, x, reference=None):
        """Undo ``normalise()`` (identity if no normalisation is set).
        ``reference``: the input that was normalised, whose own range a
        "sample" normalisation maps back to."""
        if getattr(self, "normalisator", None) is None:
            return x
        return self.normalisator.unnormalise(x, reference)

    @staticmethod
    def cite(cite_format="bib"):
        """Where the model comes from: its paper's citation, BibTeX ("bib") or
        "MLA" (``LION.utils.citation``); None if it has none. Models from a
        paper define it."""
        return None

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
        - 'operator_params' (optional): the operator's parameters (e.g. the CT
          geometry) of the data, for a model without them

        If you want to save the model for training later (i.e. checkpoiting), use save_checkpoint()
        """
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)

        # Create dict of saved data:
        ##########################
        dic = {"model_state_dict": self.state_dict()}
        if getattr(self, "initial_random_state", None) is not None:
            dic["initial_random_state"] = self.initial_random_state.cpu()

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
        if kwargs.get("scheduler") is not None:  # its state_dict(), for resuming
            dic["scheduler_state_dict"] = kwargs.pop("scheduler")
        kwargs.pop("scheduler", None)

        # (optional) the operator's parameters of the data, for a model
        # without them (the data record has them too)
        given_operator_params = kwargs.pop("operator_params", None)

        if kwargs:  # if not empty yet
            raise ValueError(
                "The following parameters are not understood: "
                + str(list(kwargs.keys()))
            )
        # The model's metadata: the spec that builds it again (see
        # LION.utils.spec), and what is known of its data and training.
        metadata = {"lion_format": spec.FORMAT_VERSION}
        metadata["model"] = spec.to_spec(self)
        # LION's code (version, commit, uncommitted changes), the main
        # packages and the machine (LION.utils.environment)
        first = next(self.parameters(), None)
        metadata["environment"] = environment.environment_record(
            None if first is None else first.device
        ).serialize()
        # Operator parameters given here describe the data (e.g. for a model
        # without them)
        if given_operator_params is not None:
            metadata["operator_params"] = spec.encode(given_operator_params)
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

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"optimizer must be a torch optimizer, not {type(optimizer)}"
            )
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
        plus paths and numpy arrays (see
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
        # the weights and the random state they were made from are the model
        if len(set(data) - {"initial_random_state"}) > 1 and not supress_warnings:
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
                raise ValueError(
                    f"{where} has no 'lion_format': it was saved before LION's "
                    "metadata (specs), which LION doesn't read"
                )
        if saved["lion_format"] > spec.FORMAT_VERSION:
            raise ValueError(
                f"{where} has metadata format {saved['lion_format']}; "
                f"this version of LION reads up to {spec.FORMAT_VERSION}. Update LION."
            )

        path, kwargs = spec.build_arguments(saved["model"], allow)
        target = spec.target_class(path, cls, LIONmodel, allow)

        options = LIONParameter()
        options.lion_format = saved["lion_format"]
        if "commit_hash" in saved:  # files saved before 30 Oct 2026
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
        options.model_parameters = kwargs.get("model_parameters")
        if "operator_params" in saved:
            options.operator_params = spec.build(saved["operator_params"], allow)
        elif kwargs.get("operator_params") is not None:
            options.operator_params = kwargs["operator_params"]
        for key in ("dataset_params", "training"):
            if is_data_record(saved.get(key)) or is_training_record(saved.get(key)):
                # a record, whose specs stay dicts: making its components
                # again (e.g. the dataset, which reads the data) is on request
                setattr(options, key, LIONParameter(**saved[key]))
            elif key in saved:
                setattr(options, key, spec.build(saved[key], allow))
        return target, kwargs, options

    @classmethod
    def instantiate(cls, source, *, allow=(), initial_weights=False, **arguments):
        """The model a saved file describes, with its saved parameters and new
        (untrained) weights. ``load()`` is this plus the saved weights.

        ``source``: a model's file (``"model"``, ``"model.json"`` or
        ``"model.pt"``: the ``.json`` is read), the dict read from it, or the
        model's spec. Called on a model class it makes that class; called on
        ``LIONmodel``, the class the file names (``allow`` as in ``load``).

        ``arguments`` replace recorded ones, e.g. ``model_parameters`` from
        ``parameters_from_file``, changed, or ``operator_params``.

        ``initial_weights=True`` (``source`` a file a solver saved): the
        weights its training started from, as its training record says: made
        again from the random state saved with the model, or read from the
        initial weights the solver saved; checked against the recorded hash.
        Not with ``arguments``: a changed model is trained from new weights.
        """
        target, kwargs, _ = cls._read_model_file(source, allow)
        if initial_weights and arguments:
            raise ValueError(
                "initial_weights=True makes the recorded model's initial weights: "
                "a model with other arguments starts from new ones"
            )
        kwargs.update(arguments)
        if not initial_weights:
            return target(**kwargs)
        return _with_initial_weights(target, kwargs, source)

    @classmethod
    def parameters_from_file(cls, source, *, allow=()):
        """What ``default_parameters()`` gives, as ``source`` (as in
        ``instantiate``) recorded it: the model's parameters, to change and
        make a model with, e.g.

            parameters = LPD.parameters_from_file("model")
            parameters.n_iters = 5
            model = LPD.instantiate("model", model_parameters=parameters)

        ``instantiate`` makes the class the file records, with its other
        arguments (e.g. the operator's parameters); ``LPD(operator_params,
        parameters)`` works too.
        Called on a model class, a file of another class gives a warning.
        """
        target, kwargs, _ = cls._read_model_file(source, allow)
        if kwargs.get("model_parameters") is None:
            return target.default_parameters()
        return kwargs["model_parameters"]

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
        model.initial_random_state = _saved_random_state(data)
        # where it came from, so records of methods using it can name the file
        model.source_file = fname.with_suffix("")
        # the loaded weights' hash: training that starts from them records
        # this file as where they come from (fine-tuning)
        model.source_weights_sha256 = weights_sha256(model)

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
        model.initial_random_state = _saved_random_state(data)

        return model, options.unpack(), data

    @classmethod
    def load_checkpoint_if_exists(
        cls, fname, model, optimiser, total_loss, verbose=False
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
            if verbose >= 2:
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


def made_from_random_state(target, kwargs, state):
    """``target(**kwargs)`` made with torch's random state ``state`` (so with
    the initial weights a model made from that state had), leaving the
    program's random numbers as they were."""
    saved = rng.random_state()
    try:
        torch.set_rng_state(state.cpu())
        return target(**kwargs)
    finally:
        rng.set_random_state(saved)


def _saved_random_state(data):
    """The random state a model file saved (None for files saved before)."""
    state = data.pop("initial_random_state", None)
    return None if state is None else state.cpu()


def _with_initial_weights(target, kwargs, source):
    """The model ``source`` describes, with the initial weights of its
    training (see ``LIONmodel.instantiate``)."""
    from LION.optimizers.training_record import read_training_record

    training = read_training_record(source)
    how = getattr(training, "initial_weights", None)
    how = vars(how) if isinstance(how, LIONParameter) else how
    made_from = (how or {}).get("from")
    if made_from == "random_state":
        state = LIONmodel._load_data(source, supress_warnings=True).get(
            "initial_random_state"
        )
        if state is None:
            raise ValueError(f"{source} has no saved random state")
        model = made_from_random_state(target, kwargs, state)
    elif made_from == "file":
        model = target(**kwargs)
        path = Path(source).parent / how["file"]
        model.load_state_dict(
            torch.load(path, map_location=default_device(), weights_only=True)
        )
    else:
        raise ValueError(
            f"{source}'s training record doesn't say how to make its initial "
            "weights again (saved before LION recorded it, or without a save "
            "folder)"
        )
    if weights_sha256(model) != getattr(training, "initial_weights_sha256", None):
        warnings.warn(
            f"the initial weights made again from {source} aren't the recorded "
            "ones (other code, or another machine)",
            stacklevel=3,
        )
    return model
