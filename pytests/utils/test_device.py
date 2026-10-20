"""LION's default device follows torch.set_default_device / torch.cuda.set_device."""

import pytest
import torch

import LION.optimizers.LIONsolver as lion_solver
import LION.optimizers.SupervisedSolver as supervised_solver
from LION.CTtools.ct_geometry import Geometry
from LION.utils.device import default_device
from pytests.optimizers._helpers import small_dncnn


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def restore_default_device():
    yield
    torch.set_default_device("cpu")


def test_cpu_without_cuda(no_cuda):
    assert default_device() == torch.device("cpu")


def test_follows_torch_set_default_device(no_cuda, restore_default_device):
    torch.set_default_device("meta")  # any non-CPU device; meta needs no GPU
    assert default_device() == torch.device("meta")


def test_follows_torch_cuda_set_device(monkeypatch):
    # what torch.cuda.set_device(2) does, without needing three GPUs
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert default_device() == torch.device("cuda", 2)


def test_solver_uses_default_device(monkeypatch, restore_default_device):
    monkeypatch.setattr(lion_solver, "make_operator", lambda geometry: None)
    monkeypatch.setattr(supervised_solver, "make_operator", lambda geometry: None)
    torch.set_default_device("meta")
    model = small_dncnn()
    solver = supervised_solver.SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        geometry=Geometry.default_parameters(),
    )
    assert solver.device == torch.device("meta")
    assert next(solver.model.parameters()).device == torch.device("meta")


def test_explicit_device_wins(monkeypatch):
    monkeypatch.setattr(lion_solver, "make_operator", lambda geometry: None)
    monkeypatch.setattr(supervised_solver, "make_operator", lambda geometry: None)
    model = small_dncnn()
    solver = supervised_solver.SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        geometry=Geometry.default_parameters(),
        device=torch.device("cpu"),
    )
    assert solver.device == torch.device("cpu")
