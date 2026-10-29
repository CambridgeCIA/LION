"""The record of a training (LION.optimizers.training_record): what the solvers
write in every model file, the training seed, resuming the same run,
LIONsolver.instantiate, LION_RESULTS_PATH and the seeding helpers.

Solvers run on CPU without tomosipo: the ``no_ct`` fixture (autouse) replaces
the CT operator by ``None`` and FDK by the identity, as in test_solvers.py.
"""

import io
import json
import os
import pathlib
import random
import tempfile
import types
import warnings

# LION.utils.paths requires LION_DATA_PATH at import time
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset, TensorDataset  # noqa: E402

import LION.optimizers.LIONsolver as lion_solver  # noqa: E402
import LION.optimizers.SupervisedSolver as supervised_solver  # noqa: E402
from LION.CTtools.ct_geometry import Geometry  # noqa: E402
from LION.data_loaders.data_record import data_record  # noqa: E402
from LION.models.LIONmodel import LIONmodel  # noqa: E402
from LION.optimizers.EquivariantSolver import (  # noqa: E402
    EquivariantSolver,
    RotationGroup,
)
from LION.optimizers.GaussianDenoiserSolver import (  # noqa: E402
    GaussianDenoiserSolver,
)
from LION.optimizers.LIONsolver import LIONsolver  # noqa: E402
from LION.optimizers.training_record import (  # noqa: E402
    is_training_record,
    loader_settings,
    read_training_record,
    weights_sha256,
)
from LION.utils import paths, rng  # noqa: E402
from LION.utils.parameter import LIONParameter  # noqa: E402
from LION.utils.image_transforms import rotate  # noqa: E402
from LION.utils.spec import build, encode, to_spec  # noqa: E402
from pytests.helpers import small_dncnn  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def no_ct(monkeypatch):
    for module in (lion_solver, supervised_solver):
        monkeypatch.setattr(module, "make_operator", lambda geometry: None)
        monkeypatch.setattr(module, "fdk", lambda sino, op: sino)


def data(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(
        torch.rand(n, 1, 8, 8, generator=g), torch.rand(n, 1, 8, 8, generator=g)
    )


def model_from(init_seed=0):
    torch.manual_seed(init_seed)
    return small_dncnn()


def denoiser_solver(folder=None, init_seed=0, seed=0, lr=1e-2, validation=True):
    """A GaussianDenoiserSolver (random noise in every batch) on shuffled data,
    checkpointing every epoch: its training uses torch's random numbers."""
    model = model_from(init_seed)
    solver = GaussianDenoiserSolver(
        model,
        torch.optim.Adam(model.parameters(), lr=lr),
        torch.nn.MSELoss(),
        geometry=Geometry.default_parameters(),
        device=torch.device("cpu"),
        noise_level=[0.05, 0.2],
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


def test_record_is_read_back_as_a_lion_parameter(tmp_path):
    solver = denoiser_solver(tmp_path)
    solver.train(2)
    solver.save_final_results()
    _, options, _ = LIONmodel.load(tmp_path / "final", supress_warnings=True)
    record = options.training
    assert isinstance(record, LIONParameter)
    assert record.serialize() == saved(tmp_path, "final")["training"]
    assert record.epochs.done == 2 and record.weights.kind == "final"
    assert record.train.shuffle is True and record.validation.every == 1
    assert isinstance(record.solver, dict) and isinstance(record.optimizer, dict)
    assert isinstance(options.dataset_params, LIONParameter)
    assert read_training_record(tmp_path / "final").serialize() == record.serialize()


def test_validation_data_recorded_training_data_as_dataset_params(tmp_path):
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


def test_save_folder_is_not_recorded(tmp_path):
    solver = denoiser_solver(tmp_path)
    solver.train(1)
    solver.save_final_results()
    assert "save_folder" not in saved(tmp_path, "final")["training"]["solver"]


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


def test_no_seed_leaves_the_random_state_alone():
    torch.manual_seed(5)
    first = denoiser_solver(seed=None)
    torch.manual_seed(5)
    first.train(2)
    torch.manual_seed(6)
    second = denoiser_solver(seed=None)
    torch.manual_seed(6)
    second.train(2)
    assert not same_weights(first.model, second.model)


def test_training_randomness_changes_within_the_run():
    """The seed makes the run repeatable, not each batch's noise the same."""
    solver = denoiser_solver(validation=False)
    solver.train(1)
    rng.seed_all(0)
    a, b = solver._noise_level(), solver._noise_level()
    assert a != b


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


def test_initial_weights_hash(tmp_path):
    solver = denoiser_solver(tmp_path)
    start = weights_sha256(solver.model)
    solver.train(2)
    solver.save_final_results()
    assert saved(tmp_path, "final")["training"]["initial_weights_sha256"] == start
    assert start != weights_sha256(solver.model)
    assert denoiser_solver(tmp_path).initial_weights_sha256 is None  # not trained


def test_checkpoint_random_state_loads_safely(tmp_path):
    denoiser_solver(tmp_path).train(1)
    data_ = torch.load(tmp_path / "check_0001.pt", weights_only=True)
    assert set(data_["random_state"]) == {"torch", "cuda", "numpy", "random"}


def test_e2i_noise_std_comes_back_on_resume(tmp_path, monkeypatch):
    import LION.optimizers.Equivariance2InverseSolver as e2i

    identity = types.SimpleNamespace(
        to_autograd=lambda op, num_extra_dims: (lambda x: x)
    )
    monkeypatch.setattr(e2i, "import_tomosipo", lambda *args: identity)

    def e2i_solver():
        params = e2i.Equivariance2InverseSolver.default_parameters()
        params.recon_fn = lambda sino, op: sino  # can't be saved: named
        model = model_from()
        solver = e2i.Equivariance2InverseSolver(
            model,
            torch.optim.SGD(model.parameters(), lr=1e-3),
            torch.nn.MSELoss(),
            solver_params=params,
            geometry=Geometry.default_parameters(),
            device=torch.device("cpu"),
            verbose=False,
        )
        sinos = torch.rand(4, 1, 16, 16, generator=torch.Generator().manual_seed(0))
        solver.set_training(DataLoader(TensorDataset(sinos, sinos), batch_size=2))
        solver.set_saving(tmp_path, "e2i")
        solver.set_checkpointing("e2i_check_*.pt", 1, save_folder=tmp_path)
        return solver

    first = e2i_solver()
    first.train(1)
    estimated = first.solver_params.noise_std
    assert estimated is not None
    record = saved(tmp_path, "e2i_check_0001")["training"]["solver"]["solver_params"]
    assert "unsaved" in record["recon_fn"] and record["noise_std"] == estimated
    resumed = e2i_solver()
    resumed.train(2)
    assert resumed.solver_params.noise_std == estimated


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
        np.testing.assert_array_equal(made.noise_level, [0.05, 0.2])
        assert made.train_loader is None  # the data are set by the user


def test_instantiate_repeats_the_run(tmp_path):
    first = denoiser_solver(tmp_path)
    first.train(2)
    first.save_final_results()
    again = LIONsolver.instantiate(
        tmp_path / "final", model_from(), device=torch.device("cpu")
    )
    again.set_training(DataLoader(data(), batch_size=2, shuffle=True))
    again.train(2)
    assert same_weights(first.model, again.model)


def test_instantiate_with_arguments(tmp_path):
    model = small_dncnn()
    solver = supervised_solver.SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        lambda output, target: ((output - target) ** 2).mean(),  # can't be saved
        geometry=Geometry.default_parameters(),
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
# LION_RESULTS_PATH


def test_results_path_is_the_default_save_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("LION_RESULTS_PATH", str(tmp_path))
    solver = denoiser_solver()
    assert solver.save_folder == tmp_path
    monkeypatch.delenv("LION_RESULTS_PATH")
    assert denoiser_solver().save_folder is None


def test_paths_inside_results_path_are_relative(tmp_path, monkeypatch):
    monkeypatch.setenv("LION_RESULTS_PATH", str(tmp_path))
    path = tmp_path / "LPD" / "model.pt"
    written = json.loads(json.dumps(encode(path)))
    assert written == {
        "_target_": "LION.utils.paths.results_path",
        "_args_": ["LPD/model.pt"],
    }
    assert build(written) == path.resolve()
    monkeypatch.delenv("LION_RESULTS_PATH")
    with pytest.raises(RuntimeError, match="LION_RESULTS_PATH"):
        paths.results_path("LPD/model.pt")


def test_innermost_lion_folder_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("LION_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("LION_RESULTS_PATH", str(tmp_path / "results"))
    assert encode(tmp_path / "results" / "a")["_target_"].endswith("results_path")
    assert encode(tmp_path / "processed")["_target_"].endswith("data_path")


# --------------------------------------------------------------------------
# Pieces


def test_loader_settings():
    assert loader_settings(DataLoader(data(), batch_size=3, shuffle=True)) == {
        "batch_size": 3,
        "drop_last": False,
        "shuffle": True,
    }
    assert loader_settings(DataLoader(data(), batch_size=3))["shuffle"] is False
    sampler = torch.utils.data.WeightedRandomSampler([1.0] * 8, 8)
    custom = loader_settings(DataLoader(data(), sampler=sampler))
    assert custom["shuffle"] is None and "WeightedRandomSampler" in custom["sampler"]


def test_subset_record():
    record = data_record(Subset(data(), [3, 1]))
    assert record.subset.indices == [3, 1]


def test_random_state_round_trip():
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


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_random_state_loaded_onto_the_gpu():
    """Checkpoints are loaded with map_location=default_device(), so on a GPU
    machine the saved states come back as CUDA tensors."""
    rng.seed_all(3)
    state = rng.random_state()
    expected = torch.rand(3), torch.rand(3, device="cuda")
    on_gpu = dict(state, torch=state["torch"].cuda())
    on_gpu["cuda"] = [s.cuda() for s in state["cuda"]]
    torch.rand(5), torch.rand(5, device="cuda")
    rng.set_random_state(on_gpu)
    assert torch.equal(torch.rand(3), expected[0])
    assert torch.equal(torch.rand(3, device="cuda"), expected[1])


def test_rotation_group_is_the_same_as_before():
    group = EquivariantSolver.default_parameters().transformation_group
    assert isinstance(group, RotationGroup) and len(group) == 360
    x = torch.rand(1, 1, 16, 16)
    for i in (0, 1, 45, 359):
        assert torch.equal(group[i](x), rotate(x, i * 1.0))
    written = json.loads(json.dumps(to_spec(EquivariantSolver.default_parameters())))
    assert written["transformation_group"]["cardinality"] == 360
    assert build(written).transformation_group.cardinality == 360
    random.seed(0)
    picked = [random.choice(group) for _ in range(3)]
    random.seed(0)
    assert [random.choice(range(360)) for _ in range(3)] == [
        round(p.keywords["angle"]) for p in picked
    ]


# --------------------------------------------------------------------------
# The initial weights: made again from the random state saved with the model,
# or saved by the solver when they can't be


def test_the_model_keeps_the_random_state_it_was_made_with(tmp_path):
    torch.manual_seed(5)
    state = torch.get_rng_state()
    model = small_dncnn()
    torch.testing.assert_close(model.initial_random_state, state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.save(tmp_path / "model")
    loaded = LIONmodel.load(tmp_path / "model", supress_warnings=True)[0]
    torch.testing.assert_close(loaded.initial_random_state, state)


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


def test_initial_weights_saved_when_they_cant_be_made_again(tmp_path):
    """A custom (or pretrained) start: the solver saves the initial weights."""
    solver = denoiser_solver(tmp_path)
    with torch.no_grad():
        for p in solver.model.parameters():
            p.mul_(0.5)
    initial = weights_sha256(solver.model)
    solver.train(1)
    solver.save_final_results()
    record = read_training_record(tmp_path / "final")
    assert vars(record.initial_weights) == {"from": "file", "file": "final_initial.pt"}
    again = LIONmodel.instantiate(tmp_path / "final", initial_weights=True)
    assert weights_sha256(again) == initial


def test_without_a_save_folder_they_cant_be_saved():
    solver = denoiser_solver(validation=False)
    with torch.no_grad():
        next(solver.model.parameters()).mul_(0.5)
    with pytest.warns(UserWarning, match="can't be repeated exactly"):
        solver.train(1)
    assert solver.initial_weights == {"from": None}


def test_a_resumed_run_keeps_how_its_initial_weights_are_made(tmp_path):
    first = denoiser_solver(tmp_path)
    first.train(1)
    resumed = denoiser_solver(tmp_path, init_seed=3)  # another model: resumed
    resumed.train(2)
    assert resumed.initial_weights == {"from": "random_state"}
    assert resumed.initial_weights_sha256 == first.initial_weights_sha256


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
        geometry=Geometry.default_parameters(),
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
