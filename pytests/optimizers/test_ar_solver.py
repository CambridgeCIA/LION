"""ARSolver trains the regulariser; validation and testing reconstruct with
LearnedRegularizerReconstructor. CPU only, with an identity "CT" operator."""

import pytest
import torch

import LION.optimizers.AdversarialRegularizerSolver as ar_module
import LION.optimizers.LIONsolver as lion_solver
from LION.CTtools.ct_geometry import Geometry
from LION.operators import Operator
from pytests.optimizers._helpers import loader, small_dncnn

pytestmark = pytest.mark.filterwarnings("ignore")


class IdentityOp(Operator):
    def __call__(self, x, out=None):
        return x

    def forward(self, x):
        return x

    def adjoint(self, y):
        return y

    @property
    def domain_shape(self):
        return (1, 8, 8)

    @property
    def range_shape(self):
        return (1, 8, 8)


@pytest.fixture
def identity_ct(monkeypatch):
    monkeypatch.setattr(lion_solver, "make_operator", lambda g: IdentityOp())
    monkeypatch.setattr(ar_module, "fdk", lambda sino, op: sino)


def make_ar_solver(**params):
    model = small_dncnn()
    return ar_module.ARSolver(
        model,
        torch.optim.Adam(model.parameters()),
        geometry=Geometry.default_parameters(),
        device=torch.device("cpu"),
        solver_params=ar_module.ARParams(**params),
        verbose=False,
    )


def test_default_parameters(identity_ct):
    model = small_dncnn()
    solver = ar_module.ARSolver(
        model,
        torch.optim.Adam(model.parameters()),
        geometry=Geometry.default_parameters(),
    )
    assert solver.reconstructor.n_steps == ar_module.ARParams().n_steps


def test_training_step(identity_ct):
    solver = make_ar_solver()
    solver.set_training(loader(n=4, batch_size=2))
    solver.train_loss = None
    loss = solver.mini_batch_step(*next(iter(loader(n=2, batch_size=2))))
    assert torch.isfinite(loss)


def test_test_reports_one_value_per_batch_and_estimates_alpha(identity_ct):
    solver = make_ar_solver(n_steps=4, step_size=1e-2)
    data = loader(n=6, batch_size=2)
    solver.set_testing(data, testing_fn=torch.nn.MSELoss())
    losses = solver.test()
    assert losses.shape == (3,)
    expected_alpha = torch.stack(
        [torch.linalg.vector_norm(y - x) for xb, yb in data for x, y in zip(xb, yb)]
    ).mean()
    assert solver.reconstructor.alpha == pytest.approx(expected_alpha.item(), rel=1e-5)


def test_validation_uses_the_reconstructor(identity_ct, tmp_path):
    solver = make_ar_solver(n_steps=2, step_size=1e-2)
    calls = []
    original = solver.reconstructor.reconstruct
    solver.reconstructor.reconstruct = lambda d: calls.append(d.shape) or original(d)
    solver.set_saving(tmp_path, "model")
    solver.set_validation(loader(n=4, batch_size=2), validation_freq=1)
    assert solver.validate() > 0
    assert calls == [torch.Size([2, 1, 8, 8])] * 2


def test_default_metrics_are_reconstruction_mse(identity_ct, tmp_path):
    solver = make_ar_solver()
    solver.set_saving(tmp_path, "model")
    solver.set_validation(loader(), validation_freq=1)
    solver.set_testing(loader())
    assert isinstance(solver.validation_fn, torch.nn.MSELoss)
    assert isinstance(solver.testing_fn, torch.nn.MSELoss)
