"""ARSolver evaluation logic, on CPU with an identity "CT" operator."""

import sys
import types

import numpy as np
import pytest
import torch

import LION.optimizers.AdversarialRegularizerSolver as ar_module
import LION.optimizers.LIONsolver as lion_solver
from LION.CTtools.ct_geometry import Geometry
from LION.utils.parameter import LIONParameter
from pytests.optimizers._helpers import loader, small_dncnn

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def identity_ct(monkeypatch):
    """A = A^T = identity, FDK = identity."""
    fake = types.ModuleType("tomosipo.torch_support")
    fake.to_autograd = lambda op, **kwargs: (lambda x: x)
    monkeypatch.setitem(sys.modules, "tomosipo.torch_support", fake)
    monkeypatch.setitem(sys.modules, "tomosipo", types.ModuleType("tomosipo"))
    monkeypatch.setattr(
        lion_solver, "make_operator", lambda g: types.SimpleNamespace(T=None)
    )
    monkeypatch.setattr(ar_module, "fdk", lambda sino, op: sino)


class RecordingSGD(torch.optim.SGD):
    kwargs = []

    def __init__(self, params, **kwargs):
        RecordingSGD.kwargs.append(kwargs)
        super().__init__(params, **kwargs)


def make_ar_solver(no_steps=3):
    params = ar_module.ARParams(
        optimizer_params=LIONParameter(momentum=0.5), no_steps=no_steps
    )
    model = small_dncnn()
    return ar_module.ARSolver(
        model,
        torch.optim.Adam(model.parameters()),
        RecordingSGD,
        geometry=Geometry.default_parameters(),
        device=torch.device("cpu"),
        solver_params=params,
        verbose=False,
    )


def test_estimate_alpha_is_mean_over_batches(identity_ct):
    solver = make_ar_solver()
    data = loader(n=6, batch_size=3)
    expected = np.mean(
        [torch.linalg.vector_norm(y - x, dim=(2, 3)).mean().item() for x, y in data]
    )
    solver.estimate_alpha(data)
    assert float(solver.alpha) == pytest.approx(expected, rel=1e-5)


def test_test_reports_one_value_per_batch(identity_ct):
    solver = make_ar_solver(no_steps=4)
    solver.set_testing(loader(n=6, batch_size=2), testing_fn=torch.nn.MSELoss())
    losses = solver.test()
    assert losses.shape == (3,)


def test_reconstruction_optimizer_gets_keyword_options(identity_ct):
    RecordingSGD.kwargs.clear()
    solver = make_ar_solver()
    solver.set_testing(loader(n=2, batch_size=2), testing_fn=torch.nn.MSELoss())
    solver.test()
    assert RecordingSGD.kwargs == [
        {"momentum": 0.5, "lr": solver.solver_params.step_size}
    ]
    # the parameter object must not be emptied by being used
    assert solver.solver_params.optimizer_params.momentum == 0.5
