"""Training must resume correctly from a checkpoint."""

import pathlib

import numpy as np
import pytest
import torch

from pytests.optimizers._helpers import loader, make_solver, small_dncnn

pytestmark = pytest.mark.filterwarnings("ignore")


def _params(model):
    return [p.detach().clone() for p in model.parameters()]


def _first_run(folder, n_epochs, validation=False):
    torch.manual_seed(0)
    solver = make_solver()
    solver.set_training(loader())
    solver.set_saving(folder, "model")
    solver.set_checkpointing("model_check_*.pt", checkpoint_freq=1, save_folder=folder)
    if validation:
        solver.set_validation(loader(seed=1), 1, validation_fname="model_min_val.pt")
    solver.train(n_epochs)
    return solver


def _resumed_solver(folder, validation=False):
    torch.manual_seed(1)  # different init, so we can tell whether loading happened
    solver = make_solver()
    solver.set_training(loader())
    solver.set_saving(folder, "model")
    solver.set_checkpointing("model_check_*.pt", checkpoint_freq=1, save_folder=folder)
    if validation:
        solver.set_validation(loader(seed=1), 1, validation_fname="model_min_val.pt")
    return solver


def test_resumed_model_keeps_training(tmp_path):
    first = _first_run(tmp_path, n_epochs=2)
    saved = _params(first.model)

    solver = _resumed_solver(tmp_path)
    solver.train(4)

    assert solver.current_epoch == 4
    # the optimiser must track the parameters of the model being trained
    model_params = {id(p) for p in solver.model.parameters()}
    for group in solver.optimizer.param_groups:
        assert all(id(p) in model_params for p in group["params"])
    # and training after the resume must actually change them
    assert any(not torch.equal(a, b) for a, b in zip(saved, _params(solver.model)))


def test_resume_starts_from_checkpoint_weights(tmp_path):
    first = _first_run(tmp_path, n_epochs=2)
    solver = _resumed_solver(tmp_path)
    # n_epochs equal to the checkpointed epoch: load, but don't train further
    solver.train(2)
    for a, b in zip(_params(first.model), _params(solver.model)):
        torch.testing.assert_close(a, b)


def test_resume_keeps_train_loss_history(tmp_path):
    first = _first_run(tmp_path, n_epochs=2)
    solver = _resumed_solver(tmp_path)
    solver.train(3)
    assert solver.train_loss.shape == (3,)
    np.testing.assert_allclose(solver.train_loss[:2], first.train_loss[:2])
    assert solver.train_loss[2] > 0


def test_resume_restores_min_validation_loss(tmp_path):
    first = _first_run(tmp_path, n_epochs=2, validation=True)
    best = np.min(first.validation_loss)

    solver = _resumed_solver(tmp_path, validation=True)
    solver.train(2)  # only loads
    assert solver.validation_loss[1] == pytest.approx(best)


def test_resume_with_relative_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = pathlib.Path("runs")
    folder.mkdir()
    _first_run(folder, n_epochs=1)
    solver = _resumed_solver(folder)
    solver.train(1)
    assert solver.current_epoch == 1


def test_train_without_checkpointing(tmp_path):
    solver = make_solver(small_dncnn())
    solver.set_training(loader())
    solver.train(2)
    assert solver.current_epoch == 2
    assert not list(tmp_path.iterdir())


def test_min_validation_model_lives_in_save_folder(tmp_path):
    # checkpoints may go elsewhere; the min-validation model is in save_folder
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()

    def solver_for_run(seed):
        torch.manual_seed(seed)
        solver = make_solver()
        solver.set_training(loader())
        solver.set_saving(tmp_path, "model")
        solver.set_checkpointing("model_check_*.pt", 1, save_folder=checkpoints)
        solver.set_validation(loader(seed=1), 1, validation_fname="model_min_val.pt")
        return solver

    first = solver_for_run(0)
    first.train(2)
    assert (tmp_path / "model_min_val.pt").with_suffix(".pt").is_file()
    assert not list(checkpoints.glob("model_min_val*"))

    resumed = solver_for_run(1)
    resumed.train(2)  # only loads
    assert resumed.validation_loss[1] == pytest.approx(np.min(first.validation_loss))
