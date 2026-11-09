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
from typing import Callable, Optional

from tqdm import tqdm
from LION.utils.parameter import (
    LIONParameter,
    OperatorParameters,
    resolve_reconstruction,
)
from LION.utils.device import default_device
from LION.utils import rng, spec
from LION.data_loaders.data_record import data_record
from LION.optimizers import training_record as records

# Lionmodels
from LION.models.LIONmodel import LIONmodel, ModelInputType

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

    Image-domain models get a reconstruction (with operator ``op``) as input:
    the operator's default (FDK for CT), or the model's own
    (``model.initial_reconstruction``). Sinogram-domain models
    get the measurements. With a normalisation
    set (``model.set_normalisation()``), the input is normalised (for
    measurements: as the images are, by linearity) and the output mapped
    back, so the output is in the units of the target.
    """
    normalisation = getattr(model, "normalisator", None)
    if model.get_input_type() == ModelInputType.IMAGE:
        data = model.initial_reconstruction(data, op)
        if normalisation is not None:
            return model.unnormalise(model(model.normalise(data)), data)
        return model(data)
    if normalisation is not None:
        return model.unnormalise(model(normalisation.normalise_measurements(data, op)))
    return model(data)


class LIONsolver(ABC, metaclass=ABCMeta):
    """Base class for the solvers, which train a ``model``.

    A solver is made with the model, the optimizer, the loss and its settings
    (``__init__``), then told about the data and what to do around training,
    then trained:

    - ``set_training(loader)``: the training data (required);
    - ``set_validation(loader, freq, fn)``: validate every ``freq`` epochs,
      keeping the best model;
    - ``set_saving(folder, name)``: where the final and best models go;
    - ``set_checkpointing(name, freq)``: checkpoints, to resume from;
    - ``set_testing(loader, fn)``: data for ``test()``;
    - ``train(n_epochs)``, then ``save_final_results()``.

    Every saved file has the training record (``training_record``): enough
    to train the same model again (``LIONsolver.instantiate``,
    ``LION.package``).

    A new solver derives from this class and defines
    ``mini_batch_step(y, target)`` (the loss of one mini-batch) and
    ``default_parameters()`` (its ``solver_params``); see
    ``SupervisedSolver`` for the simplest.

    ``operator_params``: the parameters of the operator the measurements come
    from (an ``OperatorParameters``; **for CT, the geometry**). Not needed when
    the model has them: the solver takes the model's.
    """

    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn: Callable,
        operator_params: OperatorParameters = None,
        solver_params: Optional[LIONParameter] = None,
        model_regularization=None,
        scheduler=None,
        scheduler_step: str = "epoch",
        verbose: int = 1,
        device: torch.device = None,
        seed: Optional[int] = 0,
        deterministic: bool = False,
    ) -> None:
        """
        Every solver takes ``(model, optimizer, loss_fn, operator_params,
        solver_params, model_regularization, scheduler, scheduler_step,
        verbose, device, seed, deterministic)``, in that order (``ARSolver``
        has no ``loss_fn``).

        ``operator_params``: the parameters of the operator (for CT, the
        geometry); the model's, if it has them.

        ``solver_params``: the solver's settings, from
        ``default_parameters()``.

        ``model_regularization``: None, or a term added to the training loss
        of every mini-batch, ``model_regularization(model, step)``: a penalty
        on the model's weights (``step`` unused) or on what it does with the
        mini-batch (e.g. ``LION.losses.jacobian_penalty``). ``step`` has
        ``input`` (what the model was given), ``output``, ``target`` (what
        the output is compared with; None for solvers without one) and
        ``forward`` (the model as the solver calls it, e.g. with the noise
        level). Recorded as ``loss_fn`` is.

        ``scheduler``: a torch learning-rate scheduler made on ``optimizer``
        (e.g. ``torch.optim.lr_scheduler.StepLR(optimizer, 30)``), or None. It
        steps after every epoch (``scheduler_step="epoch"``, torch's
        convention) or every mini-batch (``"batch"``, e.g. for
        ``OneCycleLR``). ``ReduceLROnPlateau`` steps with the validation value
        of the epochs that validate, or with the training loss when there is
        no validation. Its state is saved in checkpoints, and the training
        record has it and the learning rate of every epoch.

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

        ``verbose``: how much progress is shown. 0: nothing; 1 (default): one
        progress bar for training (every mini-batch of every epoch, with the
        epoch, the loss, the last validation value and the time left) and one
        for testing; 2: also the
        validation bars and messages (each epoch's losses, checkpoints, the
        seed). True is 1, False 0.
        """

        super().__init__()
        if solver_params is None:
            self.solver_params = self.default_parameters()
        else:
            self.solver_params = solver_params
        if not isinstance(model, LIONmodel):
            raise TypeError(f"model must be a LIONmodel, not {type(model)}")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                f"optimizer must be a torch optimizer, not {type(optimizer)}"
            )

        self.model = model
        self.optimizer = optimizer

        if operator_params is not None and not isinstance(
            operator_params, OperatorParameters
        ):
            raise TypeError(
                "operator_params must be the operator's parameters (an "
                f"OperatorParameters, e.g. a CT Geometry), not "
                f"{type(operator_params).__name__}: solvers take (model, "
                "optimizer, loss_fn, operator_params, solver_params, ...)"
            )
        if getattr(model, "operator_params", None) is not None:
            self.operator_params = model.operator_params
        else:
            if operator_params is None:
                raise ValueError("operator_params must be provided")
            self.operator_params = operator_params

        if scheduler_step not in ("epoch", "batch"):
            raise ValueError(
                f'scheduler_step must be "epoch" or "batch", not {scheduler_step!r}'
            )
        if scheduler is not None and scheduler.optimizer is not optimizer:
            raise ValueError("the scheduler must be made on the solver's optimizer")
        self.model_regularization = model_regularization
        self.scheduler = scheduler
        self.scheduler_step = scheduler_step
        # the learning rate of each epoch (of each parameter group), recorded
        self.learning_rates: list = []

        self.op = self.operator_params.make_operator()
        # "default" (the operator's default reconstruction) is replaced by the
        # function, so the training record says which was used
        if hasattr(self.solver_params, "recon_fn"):
            self.solver_params.recon_fn = resolve_reconstruction(
                self.solver_params.recon_fn, self.op, "recon_fn"
            )

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

        # where results are saved (set_saving); by default the folder in the
        # LION_RESULTS_PATH environment variable, if set
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
        # how the initial weights can be made again (see _initial_weights)
        self.initial_weights: Optional[dict] = None
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

    def _regularized(self, loss, input=None, output=None, target=None, forward=None):
        """``loss`` plus ``model_regularization(model, step)``, if the solver
        has one (see ``__init__``): ``step`` is this mini-batch's ``input``,
        ``output``, ``target`` and ``forward`` (the model as called here;
        the model itself if None)."""
        if self.model_regularization is None:
            return loss
        step = LIONParameter()
        step.input, step.output, step.target = input, output, target
        step.forward = self.model if forward is None else forward
        return loss + self.model_regularization(self.model, step)

    def _forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run the model on a batch of measurements, as in training (see
        ``model_forward``)."""
        return model_forward(self.model, data, self.op)

    def reconstruct_measurements(self, y: torch.Tensor, op) -> torch.Tensor:
        """Reconstruct the measurements ``y`` of the operator ``op`` with the
        solver parameters' ``recon_fn(y, op)``, for solvers whose method starts
        from reconstructions (Noise2Inverse, Proj2Proj, ...). Its default,
        ``"default"``, is the operator's default reconstruction (FDK for CT),
        replaced by that function when the solver is made."""
        return self.solver_params.recon_fn(y, op)

    # ------------------------------------------------------------------
    # The training record (LION.optimizers.training_record)

    def spec_kwargs(self) -> dict:
        """The solver's own constructor arguments: not the model (saved as
        the model) nor the optimizer (built on the model's parameters)."""
        kwargs = spec._constructor_arguments(self)
        kwargs.pop("model", None)
        kwargs.pop("optimizer", None)
        kwargs.pop("scheduler", None)  # recorded as the training's
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
            "parameter_groups": self._parameter_groups_record(),
            "scheduler": (
                None
                if self.scheduler is None
                else spec.encode_or_name(self.scheduler, "the scheduler")
            ),
            "seed": self.seed,
            "initial_weights_sha256": self.initial_weights_sha256,
            "initial_weights": self.initial_weights,
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
            "learning_rate": [
                lrs[0] if len(lrs) == 1 else lrs for lrs in self.learning_rates[:epoch]
            ],
        }
        return LIONParameter(**record)

    def _parameter_groups(self):
        """The optimizer's parameter groups, if it has other than one group of
        all the model's parameters with the optimizer's defaults: each group's
        parameters (their names in the model) and settings, with the learning
        rate it started with (a scheduler changes it). None otherwise."""
        names = {id(p): n for n, p in self.model.named_parameters()}
        groups = []
        for group in self.optimizer.param_groups:
            settings = {
                k: v for k, v in group.items() if k not in ("params", "initial_lr")
            }
            settings["lr"] = group.get("initial_lr", group["lr"])
            parameters = [names.get(id(p)) for p in group["params"]]
            if None in parameters:
                raise spec.SpecError(
                    "the optimizer has parameters that aren't the model's"
                )
            groups.append({"parameters": parameters, **settings})
        if (
            len(groups) == 1
            and sorted(groups[0]["parameters"]) == sorted(names.values())
            and all(groups[0].get(k) == v for k, v in self.optimizer.defaults.items())
        ):
            return None
        return groups

    def _parameter_groups_record(self):
        """``_parameter_groups()`` for the record, or what a record can only
        name (with a warning)."""
        try:
            return spec.encode(self._parameter_groups())
        except spec.SpecError as error:
            warnings.warn(
                f"the record only names the optimizer's parameter groups ({error})"
            )
            return {"unsaved": "the optimizer's parameter groups", "reason": str(error)}

    def _learning_rates(self) -> list:
        """The learning rate of each parameter group now."""
        return [float(group["lr"]) for group in self.optimizer.param_groups]

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
        class, ``solver_params``, loss and seed, the optimizer (class,
        hyperparameters and parameter groups) on ``model``'s parameters, and
        the learning-rate scheduler on that optimizer. The data, validation,
        testing and saving are set as usual (``set_training``, ...).
        ``arguments`` replace recorded ones or give those that couldn't be
        recorded (e.g. ``loss_fn`` for a lambda loss) and where it runs
        (``device``, ``verbose``). Called on ``LIONsolver``
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
        groups = getattr(record, "parameter_groups", None)
        if spec.is_unsaved(groups):
            raise ValueError(
                "the record only names the optimizer's parameter groups: "
                "they can't be made again"
            )
        params = model.parameters()
        if groups:
            named = dict(model.named_parameters())
            params = [
                {
                    "params": [named[n] for n in group["parameters"]],
                    **{k: v for k, v in group.items() if k != "parameters"},
                }
                for group in groups
            ]
        optimizer = spec.build(record.optimizer, allow, params=params)
        if "scheduler" not in arguments and getattr(record, "scheduler", None):
            if spec.is_unsaved(record.scheduler):
                raise ValueError(
                    f"the record only names its scheduler, "
                    f"{vars(record.scheduler)['unsaved']}: give scheduler=... (made on "
                    "solver.optimizer, so make the solver without it first)"
                )
            arguments["scheduler"] = spec.build(
                record.scheduler, allow, optimizer=optimizer
            )
        return target(model=model, optimizer=optimizer, **kwargs, **arguments)

    @classmethod
    def parameters_from_file(cls, source, *, allow=()):
        """What ``default_parameters()`` gives, as ``source`` (a model file a
        solver saved, or its training record) recorded it: the solver's
        ``solver_params``, to change and make the solver with
        (``instantiate(source, model, solver_params=parameters)``). The
        defaults for a solver that takes none (e.g. ``SupervisedSolver``)."""
        record = records.read_training_record(source)
        return spec.parameters(record.solver, "solver_params", cls, LIONsolver, allow)

    # This should return the default parameters of the solver
    @staticmethod
    @abstractmethod  # crash if not defined in derived class
    def default_parameters() -> LIONParameter:
        """The solver's settings (``solver_params``), each with its default:
        every setting the solver has, so this is also their list."""

    def set_training(
        self, train_loader: DataLoader, loss_fn: Optional[Callable] = None
    ):
        """The training data: a DataLoader of (measurements, target) batches.
        ``loss_fn``: replaces the solver's loss, if given."""
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
        """Validate every ``validation_freq`` epochs on ``validation_loader``,
        with ``validation_fn(output, target)`` (the training loss if None).

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
        """The data ``test()`` uses: ``test_loader``, scored with
        ``testing_fn(output, target)`` (the training loss if None).
        ``reconstructor``: as in ``set_validation``."""
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
        # inside the solver's own progress bar: the reconstructor's only at
        # verbose >= 2
        return reconstructor.reconstruct(data, verbose=max(int(self.verbose) - 1, 0))

    def set_saving(self, save_folder: str | pathlib.Path, final_result_fname: str):
        """Where results are saved: the final model as
        ``save_folder/final_result_fname`` (``save_final_results()``), and the
        best validation model next to it. ``save_folder`` must be a folder
        that exists (ValueError otherwise).
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
        """Save a checkpoint every ``checkpoint_freq`` epochs, as
        ``checkpoint_fname`` (its ``*`` replaced by the epoch) in
        ``save_folder`` (the solver's, from ``set_saving()``, if None). A
        checkpoint has the model, the optimizer's and scheduler's state, the
        losses and the random state: with ``load_checkpoint_if_exists``,
        ``train()`` resumes from the latest one, as if never stopped."""
        if isinstance(save_folder, str):
            save_folder = pathlib.Path(save_folder)
        elif save_folder is None:
            save_folder = self.save_folder
        if save_folder is None:
            raise RuntimeError(
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

    def check_training_ready(self, error=True, autofill=True):
        """Checks what training needs (all set when the solver is made).

        ``error``: raise when something required is missing (otherwise warn);
        ``autofill``: fill in what has a default (the device). Returns 0 when
        ready, 1 when something is missing, 2 when something has the wrong
        type.
        """
        return_code = 0

        # is the device set? if not, set it if autofill is True
        return_code = max(
            return_code,
            self.__check_attribute(
                "device",
                expected_type=torch.device,
                error=False,
                autofill=autofill,
                default=default_device(),
            ),
        )

        # is the model set? if not, raise error or warn
        return_code = max(
            return_code,
            self.__check_attribute(
                "model",
                expected_type=LIONmodel,
                error=error,
                autofill=False,
            ),
        )

        # is the optimizer set? if not, raise error or warn
        return_code = max(
            return_code,
            self.__check_attribute(
                "optimizer",
                expected_type=Optimizer,
                error=error,
                autofill=False,
            ),
        )

        # is the loss_fn set? if not, raise error or warn
        return_code = max(
            return_code,
            self.__check_attribute(
                "loss_fn",
                expected_type=callable,
                error=error,
                autofill=False,
            ),
        )

        # is the training loader set? if not, raise error or warn
        return_code = max(
            return_code,
            self.__check_attribute(
                "train_loader",
                expected_type=DataLoader,
                error=error,
                autofill=False,
            ),
        )

        # is the final result filename set? if not, raise error or warn or autofill
        return_code = max(
            return_code,
            self.__check_attribute(
                "final_result_fname",
                expected_type=str,
                error=False,
                autofill=False,
            ),
        )

        return return_code

    def check_validation_ready(self, autofill=True):
        """Checks what validation needs (``set_validation()``), filling in the
        validation function (the loss) and frequency (10) with ``autofill``,
        and the best model's file name. Returns 0 when ready, 1 when
        something is missing, 2 when something has the wrong type."""
        return_code = 0

        # is the validation loader set? if not, raise error or warn
        return_code = max(
            return_code,
            self.__check_attribute(
                "validation_loader",
                expected_type=DataLoader,
                error=False,
                autofill=False,
            ),
        )

        # is the validation function set? if not, raise error or warn or autofill
        return_code = max(
            return_code,
            self.__check_attribute(
                "validation_fn",
                expected_type=callable,
                error=False,
                autofill=self.validation_loader is not None,
                default=self.loss_fn,
            ),
        )
        # is the validation frequency set? if not, raise error or warn or autofill
        return_code = max(
            return_code,
            self.__check_attribute(
                "validation_freq",
                expected_type=int,
                error=False,
                autofill=self.validation_loader is not None,
                default=10,
            ),
        )

        # file name of the min-validation model (saved in self.save_folder).
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

    def check_testing_ready(self, error=True):
        """Checks what ``test()`` needs (``set_testing()``); ``error``: raise
        when something is missing (otherwise warn). Returns as
        ``check_validation_ready``."""
        return_code = 0

        # is the testing loader set? if not, raise error or warn
        return_code = max(
            return_code,
            self.__check_attribute(
                "test_loader",
                expected_type=DataLoader,
                error=error,
                autofill=False,
            ),
        )
        # is the testing function set? if not, raise error or warn or autofill
        return_code = max(
            return_code,
            self.__check_attribute(
                "testing_fn",
                expected_type=callable,
                error=error,
                autofill=False,
            ),
        )

        return return_code

    def check_checkpointing_ready(self, autofill=True):
        """Fills in what checkpointing needs; checkpointing is optional, so
        nothing missing here is warned about."""
        return_code = 0

        return_code = max(
            return_code,
            self.__check_attribute(
                "checkpoint_save_folder",
                expected_type=pathlib.Path,
                error=False,
                autofill=True,
                warn=False,
                default=self.save_folder,
            ),
        )

        # is the checkpoint filename filename set? if not, raise error or warn or autofill
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
                warn=False,
                default=default_checkpoint_fname,
            ),
        )

        # is the checkpoint frequency set? if not, raise error or warn or autofill
        return_code = max(
            return_code,
            self.__check_attribute(
                "checkpoint_freq",
                expected_type=int,
                error=False,
                autofill=autofill,
                warn=False,
                default=10,
            ),
        )

        # is the load checkpoint set? if not, raise error or warn or autofill
        return_code = max(
            return_code,
            self.__check_attribute(
                "do_load_checkpoint",
                expected_type=bool,
                error=False,
                autofill=True,
                warn=False,
                default=True,
            ),
        )

        return return_code

    def check_saving_ready(self):
        """Checks that there is a folder to save in (``set_saving()``):
        0 if so, 1 if not set, 2 if not a folder."""
        # is the save folder set? if not, raise error or warn or autofill
        return self.__check_attribute(
            "save_folder",
            expected_type=pathlib.Path,
            error=False,
            autofill=False,
        )

    def check_complete(self, error=True, autofill=True):
        """Every check (training, testing, saving, validation,
        checkpointing): the worst of their return codes."""

        return_code = max(
            self.check_training_ready(error, autofill),
            self.check_testing_ready(error),
            self.check_saving_ready(),
            self.check_validation_ready(autofill),
            self.check_checkpointing_ready(autofill),
        )
        return return_code

    def __check_attribute(
        self,
        attr,
        expected_type=None,
        error=True,
        autofill=True,
        warn=True,
        default=None,
    ):
        """Checks that the attribute ``attr`` is set, with ``expected_type``.

        When it isn't: fill it with ``default`` (``autofill``), or raise
        (``error``), or warn (``warn``: whether a missing or wrong ``attr`` is
        worth a warning; it doesn't depend on ``verbose``). Returns 0 when it
        is set, 1 when it is missing, 2 when it has the wrong type.
        """

        if not hasattr(self, attr) or getattr(self, attr) is None:
            if autofill:
                setattr(self, attr, default)
            else:
                if error:
                    raise ValueError(f"Attribute {attr} not set")
                if warn:
                    warnings.warn(f"Attribute {attr} not set")
                return 1

        # if the type we want to check against is a function, we need to treat it differently
        if expected_type is callable:
            if not callable(getattr(self, attr)):
                if error:
                    raise ValueError(f"Attribute {attr} is not callable")
                if warn:
                    warnings.warn(f"Attribute {attr} is not callable")
                return 2
        # just standrad type chekcking, error or warn, depends of settings
        elif not isinstance(getattr(self, attr), expected_type):
            if error:
                raise ValueError(
                    f"Attribute {attr} is not of type {expected_type}, its {type(getattr(self, attr))}"
                )
            if warn:
                warnings.warn(
                    f"Attribute {attr} is not of type {expected_type}, its {type(getattr(self, attr))}"
                )
            return 2
        return 0

    def save_checkpoint(self, epoch):
        """Saves a checkpoint after ``epoch`` (0-based), as
        ``set_checkpointing()`` says: the model, the optimizer's and
        scheduler's state, the losses, the random state and the training
        record."""
        if self.checkpoint_save_folder is None:
            raise RuntimeError("Saving not set: please call set_saving")
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
            scheduler=None if self.scheduler is None else self.scheduler.state_dict(),
        )

    def save_validation(self, epoch):
        """Saves the model as it is after ``epoch``, its best validation so
        far, as ``validation_fname`` in the save folder, with the training
        record and the validation value."""
        if self.validation_fname is None or self.validation_fn is None:
            raise RuntimeError(
                "No validation save filepath provided. Please call set_validation."
            )

        if self.validation_loss is None:
            raise RuntimeError("No validation losses found, failed to save.")

        if self.save_folder is None:
            raise RuntimeError(
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
        """Saves the trained model as ``final_result_fname`` in
        ``save_folder`` (those of ``set_saving()`` if None), with the training
        record and the losses. ``epoch``: the epochs trained (the current
        one if None)."""
        if save_folder is not None:
            self.save_folder = save_folder
        if final_result_fname is not None:
            self.final_result_fname = final_result_fname
        if self.save_folder is None or self.final_result_fname is None:
            raise RuntimeError("Saving not setup: Please call set_saving.")
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
        """Deletes the checkpoint files (``set_checkpointing()``), e.g. after
        a run that finished."""
        if self.checkpoint_save_folder is None:
            raise RuntimeError(
                "Saving not setup, unable to find save folder: Please call set_saving"
            )
        if self.checkpoint_fname is None:
            raise RuntimeError(
                "Checkpointing not setup, can't clear checkpoints: Please call set_checkpointing"
            )
        # this ensures all files with the same extension are removed
        for f in self.checkpoint_save_folder.glob(
            self.checkpoint_fname.replace(".pt", "")
        ):
            f.unlink()

    def test(self):
        """Scores the model on the test data (``set_testing()``): an array of
        ``testing_fn`` values, one per batch (or per sample, if
        ``testing_fn`` gives one per sample). Empty, with a warning, without
        test data."""
        if self.check_testing_ready() != 0:
            warnings.warn("Solver not setup to test. Please call set_testing.")
            return np.array([])
        assert self.test_loader is not None
        assert self.testing_fn is not None

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            test_loss = np.array([])
            for data, target in tqdm(
                self.test_loader, desc="Testing", disable=self.verbose < 1
            ):
                output = self._reconstruct(
                    data.to(self.device), self.testing_reconstructor
                )
                test_loss = np.append(
                    test_loss,
                    self.testing_fn(output, target.to(self.device)).cpu().numpy(),
                )
        if was_training:
            self.model.train()

        if self.verbose >= 2:
            tqdm.write(
                f"Testing loss: {test_loss.mean()} - Testing loss std: {test_loss.std()}"
            )

        return test_loss

    def load_checkpoint(self):
        """Loads the latest checkpoint, if there is one, and resumes from it:
        the model, the optimizer's and scheduler's state, the epoch, the
        losses, the random state. ``train()`` calls it."""
        if self.checkpoint_save_folder is None:
            raise RuntimeError("Loading not set. Please call set_loading ")
        if self.checkpoint_fname is None:
            raise RuntimeError(
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
            if self.verbose >= 2:
                tqdm.write(
                    f"Loaded checkpoint at epoch {self.current_epoch}. Current best validation value is {self.validation_loss[self.current_epoch-1]}"
                )
        return self.current_epoch

    def _initial_weights(self) -> dict:
        """How the weights training starts from can be made again, for the
        training record: ``{"from": "random_state"}`` if the model made again
        from the random state saved with it (``LIONmodel.__init__``) has
        exactly these weights (checked here, once); otherwise (a custom or
        pretrained start, layers made on a GPU or with numpy's random numbers)
        the weights are saved next to the results, ``{"from": "file", "file":
        "<name>_initial.pt"}``; ``{"from": None}`` if there is no save folder
        to save them in. With saved or unsaved weights, ``origin`` is the
        model file they were loaded from (``LIONmodel.load``) and its weights
        hash: a fine-tuned model's original. Weights that didn't come from a
        LION model file (e.g. ``load_state_dict``) have ``origin: None``, with
        a warning."""
        from LION.models.LIONmodel import made_from_random_state

        state = getattr(self.model, "initial_random_state", None)
        if state is not None:
            try:
                written = spec.to_spec(self.model)
                module = type(self.model).__module__.split(".")[0] + "."
                path, kwargs = spec.build_arguments(written, allow=(module,))
                target = spec.resolve(path, allow=(module,))
                made = made_from_random_state(target, kwargs, state)
                if records.weights_sha256(made) == self.initial_weights_sha256:
                    return {"from": "random_state"}
            except Exception:  # can't be made again from its spec
                pass
        origin = self._origin()
        if self.save_folder is None or self.final_result_fname is None:
            warnings.warn(
                "the initial weights can't be made again from the model's random "
                "state, and there is no save folder (set_saving) to save them in: "
                "this training can't be repeated exactly",
                stacklevel=3,
            )
            return {"from": None, "origin": origin}
        name = f"{pathlib.Path(self.final_result_fname).stem}_initial.pt"
        torch.save(self.model.state_dict(), self.save_folder / name)
        return {"from": "file", "file": name, "origin": origin}

    def _origin(self):
        """The LION model file the model's weights were loaded from, if they
        still are that file's (see ``_initial_weights``)."""
        source = getattr(self.model, "source_file", None)
        loaded = getattr(self.model, "source_weights_sha256", None)
        if source is not None and loaded == self.initial_weights_sha256:
            return {
                "file": spec.encode(pathlib.Path(source)),
                "weights_sha256": loaded,
            }
        warnings.warn(
            "the weights training starts from aren't a model made from its random "
            "state, nor loaded from a LION model file (LIONmodel.load): they are "
            "saved, but where they come from isn't recorded",
            stacklevel=4,
        )
        return None

    def _resume_from(self, data):
        """Continue the run a checkpoint was saved from: its random state, the
        model's initial random state and how its initial weights are made
        (those of the run, not of the model made to resume it), and the values
        the solver computed during the run (``solver_params`` that are
        ``None`` here)."""
        if self.seed is not None and "random_state" in data:
            rng.set_random_state(data["random_state"])
        if self.scheduler is not None and "scheduler_state_dict" in data:
            self.scheduler.load_state_dict(data["scheduler_state_dict"])
        if data.get("initial_random_state") is not None:
            self.model.initial_random_state = data["initial_random_state"].cpu()
        try:
            record = records.read_training_record(data["file"])
        except (ValueError, OSError):
            return  # a checkpoint saved without a training record
        self.initial_weights_sha256 = getattr(record, "initial_weights_sha256", None)
        history = getattr(record, "history", None)
        self.learning_rates = [
            lrs if isinstance(lrs, list) else [lrs]
            for lrs in getattr(history, "learning_rate", [])
        ]
        initial = getattr(record, "initial_weights", None)
        self.initial_weights = (
            vars(initial) if isinstance(initial, LIONParameter) else initial
        )
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
        """One pass over the training data: for each mini-batch, the loss
        (``mini_batch_step``), a gradient step (and a scheduler step, with
        ``scheduler_step="batch"``). Returns the mean loss of the epoch."""
        if self.train_loader is None:
            raise RuntimeError("Training dataloader not set: Please call set_training")
        self.model.train()
        epoch_loss = 0.0
        progress = getattr(self, "_progress", None)  # train()'s progress bar
        for batch, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            batch_loss = self.mini_batch_step(
                data.to(self.device), target.to(self.device)
            )
            batch_loss.backward()
            epoch_loss += batch_loss.item()
            self.optimizer.step()
            if self.scheduler is not None and self.scheduler_step == "batch":
                self.scheduler.step()
            if progress is not None:
                shown = {
                    "epoch": f"{self.current_epoch + 1}/{self.n_epochs}",
                    "loss": f"{epoch_loss / (batch + 1):.4g}",
                }
                if getattr(self, "_last_validation", None) is not None:
                    shown["validation"] = f"{self._last_validation:.4g}"
                progress.set_postfix(shown, refresh=False)
                progress.update()
        return epoch_loss / len(self.train_loader)

    def epoch_step(self, epoch):
        """Epoch ``epoch`` (0-based): training (``train_step``), validation
        when due (saving the best model), and a scheduler step."""
        # the learning rates this epoch starts with
        self.learning_rates = self.learning_rates[:epoch] + [self._learning_rates()]
        self.train_loss[epoch] = self.train_step()
        # actually make sure we're doing validation
        if self.validation_loss is not None and (epoch + 1) % self.validation_freq == 0:
            # validation leaves training's random numbers as they were (a
            # DataLoader draws one when it starts), so the trained model doesn't
            # depend on how often it was validated
            cuda = [self.device] if torch.device(self.device).type == "cuda" else []
            with torch.random.fork_rng(devices=cuda):
                self.validation_loss[epoch] = self.validate()
            self._last_validation = self.validation_loss[epoch]  # for the bar
            if self.verbose >= 2:
                tqdm.write(
                    f"Epoch {epoch+1} - Training loss: {self.train_loss[epoch]} - Validation loss: {self.validation_loss[epoch]}"
                )

            if self._saves_min_validation() and self._is_best_validation(epoch):
                self.save_validation(epoch)
        elif self.verbose >= 2:
            tqdm.write(f"Epoch {epoch+1} - Training loss: {self.train_loss[epoch]}")
        if self.scheduler is not None and self.scheduler_step == "epoch":
            self._step_scheduler(epoch)

    def _step_scheduler(self, epoch):
        """Steps the scheduler after ``epoch``. ``ReduceLROnPlateau`` steps
        with the validation value of the epochs that validate, or with the
        training loss when there is no validation."""
        if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        elif self.validation_loss is None:
            self.scheduler.step(self.train_loss[epoch])
        elif not np.isnan(self.validation_loss[epoch]):
            self.scheduler.step(self.validation_loss[epoch])

    def train(self, n_epochs):
        """Trains until ``n_epochs`` epochs are done in all: resuming from a
        checkpoint (``set_checkpointing()``) or an earlier ``train()``, with
        one progress bar (``verbose``). The model is left in training mode;
        ``save_final_results()`` saves it."""
        if n_epochs <= 0:
            raise ValueError(
                f"Number of epochs must be a positive integer, not {n_epochs}"
            )
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
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            higher = self.validation_loss is not None and (
                self._validation_higher_is_better()
            )
            if self.scheduler.mode != ("max" if higher else "min"):
                warnings.warn(
                    f"ReduceLROnPlateau has mode={self.scheduler.mode!r}, but it "
                    f"steps with {'the validation value' if self.validation_loss is not None else 'the training loss'}, "
                    f"which is better when {'higher' if higher else 'lower'}"
                )

        # Fills in checkpointing defaults (frequency, file name) when possible.
        self.check_checkpointing_ready()
        do_checkpoint = (
            self.checkpoint_freq is not None
            and self.checkpoint_save_folder is not None
            and self.checkpoint_fname is not None
        )

        if self.do_load_checkpoint and do_checkpoint:
            if self.verbose >= 2:
                tqdm.write("Loading checkpoint...")
            self.current_epoch = self.load_checkpoint()

        if self.current_epoch == 0:  # a new run
            if self.seed is not None:
                rng.seed_all(self.seed)
                if self.verbose >= 2:
                    tqdm.write(
                        f"Training seed: {self.seed} (the solver's seed argument; "
                        "seed=None for an unseeded run)"
                    )
            self.initial_weights_sha256 = records.weights_sha256(self.model)
            self.initial_weights = self._initial_weights()

        self.model.train()
        # one progress bar over every mini-batch left, so its ETA is the
        # training's (verbose >= 1)
        self._progress = tqdm(
            total=max(n_epochs - self.current_epoch, 0) * len(self.train_loader),
            desc="Training",
            unit="batch",
            disable=self.verbose < 1,
        )
        try:
            while self.current_epoch < n_epochs:
                self.epoch_step(self.current_epoch)

                if (
                    do_checkpoint
                    and (self.current_epoch + 1) % self.checkpoint_freq == 0
                ):
                    self.save_checkpoint(self.current_epoch)

                self.current_epoch += 1
        finally:
            self._progress.close()
            self._progress = None
            self._last_validation = None

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
        """The validation value of the model now: the mean of
        ``validation_fn`` over the validation data, with the model in
        evaluation mode, the random state left as it was."""
        if self.check_validation_ready() != 0:
            raise RuntimeError(
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

        if self.verbose >= 2:
            tqdm.write(
                f"Validation loss: {validation_loss.mean()} - Validation loss std: {validation_loss.std()}"
            )

        # return to train if it was in train
        if status:
            self.model.train()

        return np.mean(validation_loss)

    def _validation_loop(self):
        """``validation_fn`` of the model's output on each validation batch
        (through the validation reconstructor, if any): one value per batch,
        or per sample."""
        validation_loss = np.array([])
        for data, targets in tqdm(
            self.validation_loader, disable=self.verbose < 2, leave=False
        ):
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
        """The model being trained."""
        return self.model

    @abstractmethod
    def mini_batch_step(self, y, target_batch) -> torch.Tensor:
        """The loss of one mini-batch: measurements ``y`` and their targets
        (the solver adds ``model_regularization`` with ``_regularized``).
        What a new solver defines."""

    @staticmethod
    def cite(cite_format="bib"):
        """Where the solver comes from: its paper's citation, BibTeX ("bib") or
        "MLA" (``LION.utils.citation``); None if it has none. Solvers
        from a paper define it."""
        return None
