# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""The record of a training (LION.optimizers.training_record): what the solvers
write in every model file, the training seed, resuming the same run,
LIONsolver.instantiate, LION_RESULTS_PATH and the seeding helpers.

Solvers run on CPU without tomosipo: the ``no_ct`` fixture (autouse) replaces
the CT operator by ``None`` and FDK by the identity, as in test_solvers.py.
"""

import functools
import importlib
import io
import json
import os
import pathlib
import random
import tempfile
import warnings

# the datasets' default folders are in LION_DATA_PATH
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset, TensorDataset  # noqa: E402

import LION.optimizers.LIONsolver as lion_solver  # noqa: E402
import LION.optimizers.SupervisedSolver as supervised_solver  # noqa: E402
from LION.operators.ct.ct_geometry import Geometry  # noqa: E402
from LION.models.LIONmodel import LIONmodel  # noqa: E402
from LION.optimizers.GaussianDenoiserSolver import (  # noqa: E402
    GaussianDenoiserSolver,
)
from LION.optimizers.AdversarialRegularizerSolver import ARSolver  # noqa: E402
from LION.optimizers.LIONsolver import LIONsolver  # noqa: E402
from LION.metrics.psnr import PSNR  # noqa: E402
from LION.optimizers.training_record import (  # noqa: E402
    is_training_record,
    read_training_record,
    weights_sha256,
)
from LION.utils import rng  # noqa: E402
from LION.utils.spec import encode  # noqa: E402
from pytests.helpers import CTStandIn, replace_fdk, small_dncnn  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def no_ct(monkeypatch):
    for module in (lion_solver, supervised_solver):
        monkeypatch.setattr(Geometry, "make_operator", lambda self: CTStandIn())
    replace_fdk(monkeypatch, lambda sino, op: sino)


def data(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(
        torch.rand(n, 1, 8, 8, generator=g), torch.rand(n, 1, 8, 8, generator=g)
    )


def model_from(init_seed=0):
    torch.manual_seed(init_seed)
    return small_dncnn()


def denoiser_solver(
    folder=None, init_seed=0, seed=0, lr=1e-2, validation=True, model=None
):
    """A GaussianDenoiserSolver (random noise in every batch) on shuffled data,
    checkpointing every epoch: its training uses torch's random numbers.
    ``model``: the model to train (default: a new one)."""
    model = model_from(init_seed) if model is None else model
    solver_params = GaussianDenoiserSolver.default_parameters()
    solver_params.noise_level = [0.05, 0.2]
    solver = GaussianDenoiserSolver(
        model,
        torch.optim.Adam(model.parameters(), lr=lr),
        torch.nn.MSELoss(),
        operator_params=Geometry.default_parameters(),
        solver_params=solver_params,
        device=torch.device("cpu"),
        seed=seed,
    )
    solver.set_training(DataLoader(data(), batch_size=2, shuffle=True))
    if validation:
        solver.set_validation(
            DataLoader(data(4, seed=1), batch_size=2), 1, validation_fname="best"
        )
    if folder is not None:
        solver.set_saving(folder, "final")
        solver.set_checkpointing("check_*.pt", 1, save_folder=folder)
    return solver


def weights(model):
    return [p.detach().clone() for p in model.parameters()]


def same_weights(a, b):
    """Equal weights, wherever each model is (``LIONmodel.load`` puts a model
    on the default device, the GPU if there is one)."""
    return all(torch.equal(x.cpu(), y.cpu()) for x, y in zip(weights(a), weights(b)))


def saved(folder, name):
    return json.loads((pathlib.Path(folder) / f"{name}.json").read_text())


# --------------------------------------------------------------------------
# The record in every file a solver writes


def test_record_in_every_saved_file(tmp_path):
    solver = denoiser_solver(tmp_path)
    solver.train(3)
    solver.save_final_results()
    final = saved(tmp_path, "final")["training"]
    assert is_training_record(final)
    assert final["solver"]["_target_"].endswith("GaussianDenoiserSolver")
    assert final["solver"]["seed"] == 0 and final["seed"] == 0
    assert final["optimizer"]["_target_"] == "torch.optim.adam.Adam"
    assert final["optimizer"]["lr"] == 1e-2
    assert final["train"] == {"batch_size": 2, "drop_last": False, "shuffle": True}
    assert final["validation"]["every"] == 1
    assert final["validation"]["function"]["_target_"].endswith("MSELoss")
    assert final["epochs"] == {"asked": 3, "done": 3}
    assert final["weights"] == {"kind": "final", "epoch": 3}
    assert final["history"]["train_loss"] == pytest.approx(list(solver.train_loss))
    assert final["history"]["validation"] == pytest.approx(list(solver.validation_loss))

    best = saved(tmp_path, "best")["training"]
    assert best["weights"]["kind"] == "best_validation"
    k = best["weights"]["epoch"]
    assert best["weights"]["value"] == pytest.approx(np.nanmin(solver.validation_loss))
    assert best["weights"]["value"] == pytest.approx(solver.validation_loss[k - 1])

    check = saved(tmp_path, "check_0002")["training"]
    assert check["weights"] == {"kind": "checkpoint", "epoch": 2}
    assert len(check["history"]["train_loss"]) == 2

    solver = denoiser_solver(tmp_path)
    solver.set_validation(
        DataLoader(Subset(data(4, seed=1), [0, 2]), batch_size=2),
        1,
        validation_fname="best",
    )
    solver.train(1)
    solver.save_final_results()
    file = saved(tmp_path, "final")
    assert file["dataset_params"]["lion_data_record"] == 1  # the training data
    assert file["training"]["validation"]["data"]["subset"] == {"indices": [0, 2]}


# --------------------------------------------------------------------------
# The seed: the same run every time, and resuming continues it


def test_same_seed_same_model(tmp_path):
    first = denoiser_solver(seed=0)
    first.train(3)
    torch.manual_seed(123)  # whatever the global state before training
    second = denoiser_solver(seed=0)
    second.train(3)
    assert same_weights(first.model, second.model)
    np.testing.assert_array_equal(first.train_loss, second.train_loss)
    other = denoiser_solver(seed=1)
    other.train(3)
    assert not same_weights(first.model, other.model)

    rng.seed_all(3)
    state = rng.random_state()
    expected = (torch.rand(3), np.random.rand(2), random.random())
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    torch.rand(5), np.random.rand(5), random.random()
    rng.set_random_state(torch.load(buffer, weights_only=True))
    again = (torch.rand(3), np.random.rand(2), random.random())
    assert torch.equal(expected[0], again[0])
    np.testing.assert_array_equal(expected[1], again[1])
    assert expected[2] == again[2]


def test_resuming_continues_the_same_run(tmp_path):
    whole = tmp_path / "whole"
    whole.mkdir()
    uninterrupted = denoiser_solver(whole)
    uninterrupted.train(4)

    stopped = tmp_path / "stopped"
    stopped.mkdir()
    denoiser_solver(stopped).train(2)
    resumed = denoiser_solver(stopped, init_seed=7)  # other weights until loaded
    resumed.train(4)

    assert same_weights(uninterrupted.model, resumed.model)
    np.testing.assert_array_equal(uninterrupted.train_loss, resumed.train_loss)
    assert resumed.initial_weights_sha256 == uninterrupted.initial_weights_sha256

    whole = tmp_path / "scheduled_whole"
    whole.mkdir()
    uninterrupted = scheduled_solver(whole)
    uninterrupted.train(4)

    stopped = tmp_path / "scheduled_stopped"
    stopped.mkdir()
    scheduled_solver(stopped).train(2)
    resumed = scheduled_solver(stopped, init_seed=7)
    resumed.train(4)
    assert resumed.learning_rates == uninterrupted.learning_rates
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 / 16)
    assert same_weights(uninterrupted.model, resumed.model)


# --------------------------------------------------------------------------
# Validation noise: the same in every validation


def test_denoiser_validation_noise_is_the_same_each_time():
    solver = denoiser_solver()
    solver.train(1)
    assert solver.validate() == solver.validate()
    unseeded = denoiser_solver(seed=None)
    unseeded.train(1)
    assert unseeded.validate() != unseeded.validate()


def test_validation_does_not_change_the_training():
    validated = denoiser_solver(validation=True)
    validated.train(2)
    not_validated = denoiser_solver(validation=False)
    not_validated.train(2)
    assert same_weights(validated.model, not_validated.model)


# --------------------------------------------------------------------------
# instantiate: the solver from its own record


def test_instantiate_makes_the_recorded_solver(tmp_path):
    solver = denoiser_solver(tmp_path, seed=4)
    solver.train(1)
    solver.save_final_results()
    model = model_from(9)
    for cls in (LIONsolver, GaussianDenoiserSolver):
        made = cls.instantiate(tmp_path / "final", model, device=torch.device("cpu"))
        assert type(made) is GaussianDenoiserSolver
        assert made.model is model and made.seed == 4
        assert type(made.optimizer) is torch.optim.Adam
        assert made.optimizer.defaults["lr"] == 1e-2
        assert {id(p) for p in made.optimizer.param_groups[0]["params"]} == {
            id(p) for p in model.parameters()
        }
        assert isinstance(made.loss_fn, torch.nn.MSELoss)
        assert made.solver_params.noise_level == [0.05, 0.2]
        assert made.solver_params.patches is None
        assert made.train_loader is None  # the data are set by the user

    model = small_dncnn()
    solver = supervised_solver.SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        lambda output, target: ((output - target) ** 2).mean(),  # can't be saved
        operator_params=Geometry.default_parameters(),
        device=torch.device("cpu"),
    )
    solver.set_training(DataLoader(data(), batch_size=4))
    solver.set_saving(tmp_path, "final")
    solver.train(1)
    solver.save_final_results()
    with pytest.raises(ValueError, match="loss_fn=..."):
        LIONsolver.instantiate(tmp_path / "final", small_dncnn())
    made = LIONsolver.instantiate(
        tmp_path / "final", small_dncnn(), loss_fn=torch.nn.L1Loss(), seed=3
    )
    assert isinstance(made.loss_fn, torch.nn.L1Loss) and made.seed == 3


# --------------------------------------------------------------------------
# Pieces


# --------------------------------------------------------------------------
# The initial weights: made again from the random state saved with the model,
# or saved by the solver when they can't be


def test_initial_weights_from_the_random_state(tmp_path):
    torch.rand(7)  # whatever the script drew before making the model
    solver = denoiser_solver(tmp_path)
    initial = weights_sha256(solver.model)
    solver.train(1)
    solver.save_final_results()
    record = read_training_record(tmp_path / "final")
    assert vars(record.initial_weights) == {"from": "random_state"}
    assert record.initial_weights_sha256 == initial
    assert not (tmp_path / "final_initial.pt").exists()  # nothing to save
    # made again: the same initial weights; the program's random numbers untouched
    before = torch.get_rng_state()
    again = LIONmodel.instantiate(tmp_path / "final", initial_weights=True)
    assert weights_sha256(again) == initial
    assert torch.equal(torch.get_rng_state(), before)

    torch.manual_seed(5)
    state = torch.get_rng_state()
    model = small_dncnn()
    torch.testing.assert_close(model.initial_random_state, state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.save(tmp_path / "model")
    loaded = LIONmodel.load(tmp_path / "model", supress_warnings=True)[0]
    torch.testing.assert_close(loaded.initial_random_state, state)


def test_initial_weights_saved_when_they_cant_be_made_again(tmp_path):
    """A custom (or pretrained) start: the solver saves the initial weights."""
    solver = denoiser_solver(tmp_path)
    with torch.no_grad():
        for p in solver.model.parameters():
            p.mul_(0.5)
    initial = weights_sha256(solver.model)
    with pytest.warns(UserWarning, match="where they come from isn't recorded"):
        solver.train(1)
    solver.save_final_results()
    record = read_training_record(tmp_path / "final")
    assert vars(record.initial_weights) == {
        "from": "file",
        "file": "final_initial.pt",
        "origin": None,
    }
    again = LIONmodel.instantiate(tmp_path / "final", initial_weights=True)
    assert weights_sha256(again) == initial


def test_a_run_is_repeated_from_its_file_alone(tmp_path):
    """Everything a training needs is in its file: the data (a LION dataset),
    the model's initial weights, the solver, its seed and settings."""
    from LION.data_loaders.LIONDataset import LIONDataset
    from pytests.test_evaluation import data as pairs

    torch.rand(11)  # the script's own random numbers, before the model
    model = small_dncnn()
    solver = supervised_solver.SupervisedSolver(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-2),
        torch.nn.MSELoss(),
        operator_params=Geometry.default_parameters(),
        device=torch.device("cpu"),
    )
    solver.set_training(DataLoader(pairs(), batch_size=2, shuffle=True))
    (tmp_path / "a").mkdir()
    solver.set_saving(tmp_path / "a", "final")
    solver.train(2)
    solver.save_final_results()

    file = tmp_path / "a" / "final"
    record = read_training_record(file)
    again = LIONmodel.instantiate(file, initial_weights=True)
    # where it runs isn't recorded: given, as when the run was made
    repeat = LIONsolver.instantiate(
        file, again, allow=["pytests."], device=torch.device("cpu")
    )
    repeat.set_training(
        DataLoader(
            LIONDataset.instantiate(file, allow=["pytests."]),
            batch_size=record.train.batch_size,
            shuffle=record.train.shuffle,
        )
    )
    repeat.train(record.epochs.asked)
    assert same_weights(repeat.model, LIONmodel.load(file, supress_warnings=True)[0])


def test_a_fine_tuned_model_records_its_original(tmp_path):
    """Trained from a model loaded from its file: the training record names
    that file and its weights' hash."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = denoiser_solver(tmp_path / "a", validation=False)
    first.train(1)
    first.save_final_results()
    loaded = LIONmodel.load(tmp_path / "a" / "final", supress_warnings=True)[0]
    tuned = denoiser_solver(tmp_path / "b", validation=False, model=loaded)
    tuned.train(1)
    tuned.save_final_results()
    how = read_training_record(tmp_path / "b" / "final").initial_weights
    assert how.serialize() == {
        "from": "file",
        "file": "final_initial.pt",
        "origin": {
            "file": encode(tmp_path / "a" / "final"),
            "weights_sha256": weights_sha256(
                LIONmodel.load(tmp_path / "a" / "final", supress_warnings=True)[0]
            ),
        },
    }
    # weights changed after loading: they aren't that file's any more
    again = LIONmodel.load(tmp_path / "a" / "final", supress_warnings=True)[0]
    with torch.no_grad():
        next(again.parameters()).mul_(0.5)
    (tmp_path / "c").mkdir()
    changed = denoiser_solver(tmp_path / "c", validation=False, model=again)
    with pytest.warns(UserWarning, match="where they come from isn't recorded"):
        changed.train(1)
    assert changed.initial_weights["origin"] is None


# --------------------------------------------------------------------------
# Solver arguments, learning-rate schedulers and the optimizer's groups


def _solver_classes():
    import inspect
    import pkgutil

    import LION.optimizers

    for info in pkgutil.walk_packages(LION.optimizers.__path__, "LION.optimizers."):
        module = importlib.import_module(info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, LIONsolver) and cls.__module__ == info.name:
                yield cls


@pytest.mark.parametrize("cls", sorted(set(_solver_classes()), key=str), ids=str)
def test_solvers_take_their_arguments_in_one_order(cls):
    import inspect

    order = [
        "model",
        "optimizer",
        "loss_fn",
        "operator_params",
        "solver_params",
        "model_regularization",
        "scheduler",
        "scheduler_step",
        "verbose",
        "device",
        "seed",
        "deterministic",
    ]
    names = list(inspect.signature(cls.__init__).parameters)[1:]
    expected = [n for n in order if not (n == "loss_fn" and cls is ARSolver)]
    assert names == expected
    assert "save_folder" not in names  # set_saving()


def scheduled_solver(folder=None, scheduler=None, step="epoch", init_seed=0, **sgd):
    model = model_from(init_seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, **sgd)
    if scheduler is None:
        scheduler = functools.partial(
            torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5
        )
    solver = supervised_solver.SupervisedSolver(
        model,
        optimizer,
        torch.nn.MSELoss(),
        operator_params=Geometry.default_parameters(),
        scheduler=scheduler(optimizer),
        scheduler_step=step,
        device=torch.device("cpu"),
    )
    solver.set_training(DataLoader(data(), batch_size=4))  # 2 batches per epoch
    if folder is not None:
        solver.set_saving(folder, "final")
        solver.set_checkpointing("check_*.pt", 1, save_folder=folder)
    return solver


def test_the_scheduler_steps_every_epoch_and_is_recorded(tmp_path):
    solver = scheduled_solver(tmp_path)
    solver.train(3)
    solver.save_final_results()
    record = saved(tmp_path, "final")["training"]
    assert record["history"]["learning_rate"] == [0.1, 0.05, 0.025]
    assert record["scheduler"] == {
        "_target_": "torch.optim.lr_scheduler.StepLR",
        "step_size": 1,
        "gamma": 0.5,
    }
    assert record["parameter_groups"] is None  # one group, the defaults
    assert "scheduler" not in record["solver"]
    assert record["solver"]["scheduler_step"] == "epoch"

    made = LIONsolver.instantiate(tmp_path / "final", model_from(1))
    assert type(made.scheduler) is torch.optim.lr_scheduler.StepLR
    assert made.scheduler.optimizer is made.optimizer
    assert (made.scheduler.step_size, made.scheduler.gamma) == (1, 0.5)
    assert made.optimizer.param_groups[0]["lr"] == 0.1  # starts again

    solver = scheduled_solver(step="batch")
    solver.train(2)
    assert solver.learning_rates == [[0.1], [0.025]]  # two batches per epoch


def test_reduce_on_plateau_steps_with_the_validation_value():
    plateau = functools.partial(
        torch.optim.lr_scheduler.ReduceLROnPlateau, mode="max", patience=0
    )
    solver = scheduled_solver(scheduler=plateau)
    stepped = []
    solver.scheduler.step = lambda value: stepped.append(value)
    solver.set_validation(DataLoader(data(4, seed=1), batch_size=2), 2, PSNR())
    solver.train(4)  # validates after epochs 2 and 4
    np.testing.assert_array_equal(stepped, solver.validation_loss[[1, 3]])


def test_parameter_groups_are_recorded_and_made_again(tmp_path):
    model = model_from()
    first, *rest = list(model.parameters())
    optimizer = torch.optim.SGD(
        [{"params": [first], "lr": 0.01}, {"params": rest}], lr=0.1, momentum=0.9
    )
    solver = supervised_solver.SupervisedSolver(
        model,
        optimizer,
        torch.nn.MSELoss(),
        Geometry.default_parameters(),
        device=torch.device("cpu"),
    )
    solver.set_training(DataLoader(data(), batch_size=4))
    solver.set_saving(tmp_path, "final")
    solver.train(1)
    solver.save_final_results()
    record = saved(tmp_path, "final")["training"]
    groups = record["parameter_groups"]
    names = [n for n, _ in model.named_parameters()]
    assert [g["parameters"] for g in groups] == [names[:1], names[1:]]
    assert [g["lr"] for g in groups] == [0.01, 0.1]
    assert record["history"]["learning_rate"] == [[0.01, 0.1]]

    again = model_from(1)
    made = LIONsolver.instantiate(tmp_path / "final", again)
    named = dict(again.named_parameters())
    assert [g["lr"] for g in made.optimizer.param_groups] == [0.01, 0.1]
    assert made.optimizer.param_groups[0]["params"] == [named[names[0]]]
    assert made.optimizer.param_groups[1]["momentum"] == 0.9


def test_the_model_regularization_is_recorded_and_made_again(tmp_path):
    from LION.losses.jacobian_penalty import ProximalDenoiserPenalty

    model = model_from()
    solver = supervised_solver.SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        Geometry.default_parameters(),
        model_regularization=ProximalDenoiserPenalty(1e-2, power_iterations=2),
        device=torch.device("cpu"),
    )
    solver.set_training(DataLoader(data(), batch_size=4))
    solver.set_saving(tmp_path, "final")
    solver.train(1)
    solver.save_final_results()
    record = saved(tmp_path, "final")["training"]
    assert record["solver"]["model_regularization"] == {
        "_target_": "LION.losses.jacobian_penalty.ProximalDenoiserPenalty",
        "weight": 1e-2,
        "epsilon": 0.1,
        "power_iterations": 2,
    }
    made = LIONsolver.instantiate(tmp_path / "final", model_from(1))
    assert isinstance(made.model_regularization, ProximalDenoiserPenalty)
    assert made.model_regularization.power_iterations == 2
