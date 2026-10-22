"""Solvers on CPU without tomosipo: checkpointing and resuming, validation and
testing, normalisation, the adversarial regulariser and Sparse2Inverse.

The solvers always build a CT operator from a geometry and run FDK for
image-domain models. Neither matters for these tests, so the ``no_ct`` fixture
(autouse, this file only) replaces the operator by ``None`` and FDK by the identity.
"""

import itertools
import pathlib
import sys
import types

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import LION.CTtools.ct_utils as ct_utils
import LION.optimizers.AdversarialRegularizerSolver as ar_module
import LION.optimizers.LIONsolver as lion_solver
import LION.optimizers.SupervisedSolver as supervised_solver
from LION.CTtools.ct_geometry import Geometry
from LION.exceptions.exceptions import LIONSolverException
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.operators import Operator
from LION.optimizers.Sparse2InverseSolver import Sparse2InverseSolver
from pytests.helpers import small_dncnn, small_gsd, small_gsdrunet

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def no_ct(monkeypatch):
    for module in (lion_solver, supervised_solver):
        monkeypatch.setattr(module, "make_operator", lambda geometry: None)
        monkeypatch.setattr(module, "fdk", lambda sino, op: sino)


def loader(n=4, batch_size=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, 8, 8, generator=g)
    y = torch.rand(n, 1, 8, 8, generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def make_solver(model=None, lr=1e-2, **kwargs):
    model = small_dncnn() if model is None else model
    return supervised_solver.SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=lr),
        torch.nn.MSELoss(),
        geometry=Geometry.default_parameters(),
        device=torch.device("cpu"),
        **kwargs,
    )


def test_supervised_solver_builds_one_operator(monkeypatch):
    built = []
    monkeypatch.setattr(lion_solver, "make_operator", lambda g: built.append(g) or "op")
    solver = make_solver()
    assert built and len(built) == 1
    assert solver.op == "op"


# --------------------------------------------------------------------------
# Checkpointing: training must resume correctly from a checkpoint


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


def test_train_twice_keeps_the_loss_history():
    solver = make_solver()
    solver.set_training(loader())
    solver.train(2)
    first = solver.train_loss.copy()
    solver.train(3)  # one more epoch
    assert solver.current_epoch == 3
    np.testing.assert_array_equal(solver.train_loss[:2], first)
    assert solver.train_loss[2] > 0


def test_checkpointing_needs_a_folder():
    solver = make_solver()
    with pytest.raises(LIONSolverException, match="set_saving"):
        solver.set_checkpointing("model_check_*.pt")


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


# --------------------------------------------------------------------------
# Validation and testing run for every model type, with the same
# (output, target) argument order as the training loss


@pytest.mark.parametrize("input_type", [ModelInputType.IMAGE, ModelInputType.SINOGRAM])
def test_validation_runs_for_all_input_types(tmp_path, input_type):
    solver = make_solver(small_dncnn(input_type))
    solver.set_saving(tmp_path, "model")
    solver.set_validation(loader(n=6), validation_freq=1)
    losses = solver._validation_loop()
    assert losses.shape == (3,)  # one value per batch
    assert solver.validate() > 0


def test_validation_runs_without_set_saving():
    solver = make_solver()
    solver.set_training(loader())
    solver.set_validation(loader(seed=1), validation_freq=1)
    with pytest.warns(UserWarning, match="not saved"):
        solver.train(2)
    assert solver.validation_loss is not None
    assert (solver.validation_loss > 0).all()


def test_test_restores_training_mode():
    solver = make_solver()
    solver.set_testing(loader())
    solver.model.train()
    solver.test()
    assert solver.model.training
    solver.model.eval()
    solver.test()
    assert not solver.model.training


def _order_check_fn(targets):
    seen = []

    def fn(first, second):
        # targets come straight from the loader, so they are leaf tensors with the
        # exact values of the dataset; the model output is not.
        seen.append(any(torch.equal(second, t) for t in targets))
        return torch.mean((first - second) ** 2)

    return fn, seen


@pytest.mark.parametrize("input_type", [ModelInputType.IMAGE, ModelInputType.SINOGRAM])
def test_validation_and_test_take_output_then_target(tmp_path, input_type):
    data = loader(n=4, batch_size=2)
    targets = [y for _, y in data]
    solver = make_solver(small_dncnn(input_type))
    solver.set_saving(tmp_path, "model")

    fn, seen = _order_check_fn(targets)
    solver.set_validation(data, validation_freq=1, validation_fn=fn)
    solver._validation_loop()
    assert seen == [True, True]

    fn, seen = _order_check_fn(targets)
    solver.set_testing(data, testing_fn=fn)
    solver.test()
    assert seen == [True, True]


@pytest.mark.parametrize("make_model", [small_gsd, small_gsdrunet])
def test_gradient_step_denoisers_validate_and_test(tmp_path, make_model):
    """Validation/testing run under torch.no_grad(); gradient-step denoisers
    enable autograd themselves, so they work too."""
    solver = make_solver(make_model())
    solver.set_saving(tmp_path, "model")
    data = loader(n=2, batch_size=2)
    solver.set_validation(data, validation_freq=1)
    solver.set_testing(data)
    assert solver.validate() > 0
    assert solver.test().shape == (1,)


# --------------------------------------------------------------------------
# model.set_normalisation() must actually be used by the solvers


def _capture_inputs(model):
    seen = []
    model.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    return seen


def test_do_normalise_follows_model():
    solver = make_solver()
    assert not solver.do_normalise
    solver.model.set_normalisation("sample")
    assert solver.do_normalise


def test_training_step_normalises_input_and_returns_target_units():
    solver = make_solver()
    solver.model.set_normalisation("sample")
    seen = _capture_inputs(solver.model)

    data = 100 + 50 * torch.rand(2, 1, 8, 8)  # far from [0, 1]
    target = data.clone()
    loss = solver.mini_batch_step(data, target)

    assert torch.isfinite(loss)
    model_input = seen[-1]
    assert model_input.min() >= 0 and model_input.max() <= 1
    # output is mapped back to the data range before the loss
    output = solver._forward(data)
    assert output.mean() > 50


def test_validation_uses_normalisation(tmp_path):
    solver = make_solver()
    solver.model.set_normalisation("sample")
    solver.set_saving(tmp_path, "model")
    solver.set_validation(loader(), validation_freq=1)
    seen = _capture_inputs(solver.model)
    solver._validation_loop()
    assert all(x.min() >= 0 and x.max() <= 1 for x in seen)


@pytest.mark.parametrize("normalisation_type", ["sample", "custom"])
def test_normalising_a_constant_image_gives_no_nan(normalisation_type):
    model = small_dncnn()
    model.set_normalisation(normalisation_type, data_range=(2.0, 2.0))
    x = torch.full((2, 1, 8, 8), 2.0)
    normalised = model.normalise(x)
    assert torch.isfinite(normalised).all()
    torch.testing.assert_close(model.unnormalise(normalised), x)


def test_model_normalise_is_identity_without_normaliser():
    model = small_dncnn()
    x = torch.rand(2, 1, 8, 8) * 10
    assert model.normalise(x) is x
    assert model.unnormalise(x) is x


def test_model_normalise_does_not_hide_errors():
    model = small_dncnn()
    model.set_normalisation("sample")
    with pytest.raises(Exception):
        model.normalise(torch.rand(8))  # 'sample' needs (B, C, H, W)


@pytest.mark.parametrize("noise_level", [0.05, [0.05], [0.01, 0.1]])
def test_gaussian_denoiser_noise_level(noise_level):
    from LION.optimizers.GaussianDenoiserSolver import GaussianDenoiserSolver

    model = small_dncnn()
    solver = GaussianDenoiserSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        geometry=Geometry.default_parameters(),
        device=torch.device("cpu"),
        noise_level=noise_level,
    )
    assert solver.noise_level.shape == (2,)
    assert solver.noise_level[0] == np.atleast_1d(noise_level)[0]


# --------------------------------------------------------------------------
# ARSolver trains the regulariser; validation and testing reconstruct with
# LearnedRegularizerReconstructor (here with an identity "CT" operator)


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


# --------------------------------------------------------------------------
# Sparse2Inverse sub-operators for any number of angular splits


@pytest.fixture
def fake_ct(monkeypatch):
    """Operators are replaced by the angles they were built with."""
    fake = types.ModuleType("tomosipo.torch_support")
    fake.to_autograd = lambda op, **kwargs: op
    monkeypatch.setitem(sys.modules, "tomosipo.torch_support", fake)
    monkeypatch.setitem(sys.modules, "tomosipo", types.ModuleType("tomosipo"))
    monkeypatch.setattr(ct_utils, "make_operator", lambda g: np.asarray(g.angles))
    monkeypatch.setattr(LIONmodel, "_make_operator", lambda self: None)


def make_s2i_solver(splits, geometry, inputs=None, recon_fn=None):
    params = Sparse2InverseSolver.default_parameters()
    params.sino_split_count = splits
    if inputs is None:
        del params.input_split_count  # parameter files from before it existed
    else:
        params.input_split_count = inputs
    if recon_fn is not None:
        params.recon_fn = recon_fn
    model = small_dncnn()
    return Sparse2InverseSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        solver_params=params,
        geometry=geometry,
        device=torch.device("cpu"),
        verbose=False,
    )


def test_defaults_match_the_paper_setup():
    params = Sparse2InverseSolver.default_parameters()
    assert (params.sino_split_count, params.input_split_count) == (4, 2)


@pytest.mark.parametrize(
    "splits, inputs", [(3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (5, 2), (5, 3)]
)
def test_inputs_and_target_projectors(fake_ct, splits, inputs):
    geometry = Geometry.default_parameters()  # 360 angles
    solver = make_s2i_solver(splits, geometry, inputs)
    angles = np.asarray(geometry.angles)

    # the single input sinogram is split into interleaved angle subsets
    assert len(solver.sub_ops) == splits
    for i, sub_op in enumerate(solver.sub_ops):
        np.testing.assert_array_equal(sub_op, angles[i::splits])

    # every combination of `inputs` splits is a network input...
    assert solver.split_combinations == list(
        itertools.combinations(range(splits), inputs)
    )
    # ...and the target is the projection onto all the remaining angles
    for combo in solver.split_combinations:
        target = tuple(s for s in range(splits) if s not in combo)
        expected = sorted(i for s in target for i in range(s, len(angles), splits))
        np.testing.assert_array_equal(
            solver.combo_ops_autograd[target], angles[expected]
        )
    # the solver's own geometry is not modified
    assert len(solver.geometry.angles) == len(angles)


def test_old_parameter_files_use_pairs(fake_ct):
    solver = make_s2i_solver(4, Geometry.default_parameters(), inputs=None)
    assert solver.input_split_count == 2
    assert solver.split_combinations == solver.two_two_strategy(4)


@pytest.mark.parametrize("splits, inputs", [(2, 2), (4, 4), (4, 0), (1, 1)])
def test_at_least_one_split_left_as_target(fake_ct, splits, inputs):
    with pytest.raises(ValueError, match="input_split_count"):
        make_s2i_solver(splits, Geometry.default_parameters(), inputs)


@pytest.mark.parametrize("splits, inputs", [(4, 2), (5, 3)])
def test_training_step_uses_each_combination(fake_ct, monkeypatch, splits, inputs):
    """One mini-batch: p sub-reconstructions per combination, averaged, then
    compared with the held-out projections."""
    # projectors return a constant sinogram over their angles, depending on the image
    monkeypatch.setitem(
        sys.modules["tomosipo.torch_support"].__dict__,
        "to_autograd",
        lambda op, **kw: (lambda img: img.mean() * torch.ones(1, 1, len(op), 8)),
    )
    calls = []

    def recon_fn(sub_sino, op):
        calls.append(sub_sino.shape[2])
        return sub_sino.mean() * torch.ones(sub_sino.shape[0], 1, 8, 8)

    geometry = Geometry.default_parameters()  # 360 angles
    solver = make_s2i_solver(splits, geometry, inputs, recon_fn=recon_fn)
    sinos = torch.rand(2, 1, 360, 8)
    loss = solver.mini_batch_step(sinos, None)
    assert torch.isfinite(loss) and loss > 0
    n_combinations = len(list(itertools.combinations(range(splits), inputs)))
    assert len(calls) == n_combinations * inputs
    assert all(n == 360 // splits for n in calls)


def test_validation_and_test_use_the_reconstruction(fake_ct, tmp_path):
    """Self-supervised solvers validate and test their reconstruct() output
    (they used to return 0 from validate(), which crashed train())."""
    geometry = Geometry.default_parameters()
    geometry.image_shape = [1, 8, 8]
    calls = []

    def recon_fn(sub_sino, op):
        calls.append(sub_sino.shape)
        return sub_sino.mean() * torch.ones(sub_sino.shape[0], 1, 8, 8)

    solver = make_s2i_solver(4, geometry, 2, recon_fn=recon_fn)
    sinos, images = torch.rand(4, 1, 360, 8), torch.rand(4, 1, 8, 8)
    data = DataLoader(TensorDataset(sinos, images), batch_size=2)
    solver.set_saving(tmp_path, "model")
    solver.set_validation(data, validation_freq=1)
    solver.set_testing(data)

    assert solver.validate() > 0
    torch.testing.assert_close(
        solver._forward(sinos[:2]), solver.reconstruct(sinos[:2])
    )
    assert solver.test().shape == (2,)
    assert calls  # the sub-reconstructions were computed
