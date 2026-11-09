# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Solvers on CPU without tomosipo: checkpointing and resuming, validation and
testing, normalisation, the adversarial regulariser and Sparse2Inverse.

The solvers always build a CT operator from a geometry and run FDK for
image-domain models. Neither matters for these tests, so the ``no_ct`` fixture
(autouse, this file only) replaces the operator by ``None`` and FDK by the identity.
"""

import functools
import itertools

import numpy as np

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import LION.optimizers.AdversarialRegularizerSolver as ar_module
import LION.optimizers.LIONsolver as lion_solver
import LION.optimizers.SupervisedSolver as supervised_solver
from LION.operators.ct.ct_geometry import Geometry
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.optimizers.Sparse2InverseSolver import Sparse2InverseSolver
from LION.utils import spec
from pytests.helpers import IdentityStandIn
from pytests.helpers import adjoint_reconstruction as helpers_adjoint
from pytests.helpers import (
    CTStandIn,
    replace_fdk,
    small_dncnn,
    small_gsd_noise_channel,
)

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def no_ct(monkeypatch):
    for module in (lion_solver, supervised_solver):
        monkeypatch.setattr(Geometry, "make_operator", lambda self: CTStandIn())
    replace_fdk(monkeypatch, lambda sino, op: sino)


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
        operator_params=Geometry.default_parameters(),
        device=torch.device("cpu"),
        **kwargs,
    )


# --------------------------------------------------------------------------
# Checkpointing: training must resume correctly from a checkpoint


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


def test_resume_keeps_train_loss_history(tmp_path):
    first = _first_run(tmp_path, n_epochs=2)
    solver = _resumed_solver(tmp_path)
    solver.train(3)
    assert solver.train_loss.shape == (3,)
    np.testing.assert_allclose(solver.train_loss[:2], first.train_loss[:2])
    assert solver.train_loss[2] > 0

    (tmp_path / "validated").mkdir()
    first = _first_run(tmp_path / "validated", n_epochs=2, validation=True)
    best = np.min(first.validation_loss)

    solver = _resumed_solver(tmp_path / "validated", validation=True)
    solver.train(2)  # only loads
    assert solver.validation_loss[1] == pytest.approx(best)


@pytest.mark.parametrize("verbose", [0, 1, 2])
def test_solvers_show_progress_by_level(tmp_path, capfd, verbose):
    solver = make_solver(verbose=verbose)
    solver.set_training(loader())
    solver.set_saving(tmp_path, "model")
    solver.set_checkpointing(
        "model_check_*.pt", checkpoint_freq=1, save_folder=tmp_path
    )
    solver.set_validation(loader(seed=1), 1, validation_fname="model_min_val.pt")
    solver.train(2)
    out, err = capfd.readouterr()
    if verbose == 0:
        assert out == "" and err == ""
        return
    # one bar over every mini-batch of every epoch (2 epochs x 2 batches)
    assert "Training" in err and "4/4" in err and "epoch=2/2" in err
    assert ("Epoch 1" in out) == (verbose == 2)  # messages only at 2


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


class Scripted:
    """A validation function returning the given values, one per validation."""

    def __init__(self, values, higher_is_better=None):
        self.values = iter(values)
        if higher_is_better is not None:
            self.higher_is_better = higher_is_better

    def __call__(self, output, target):
        return torch.tensor(next(self.values))


@pytest.mark.parametrize(
    "validation_fn, best",
    [
        (Scripted([0.5, 0.3, 0.7]), 0.3),  # a loss: lowest is best
        (Scripted([0.5, 0.9, 0.7], higher_is_better=True), 0.9),  # e.g. SSIM
        (Scripted([0.0, 0.3, 0.7]), 0.0),  # a validation value can be 0
        (Scripted([0.4, 0.0, 0.7]), 0.0),
        (Scripted([float("nan"), 0.3, 0.7]), 0.3),  # e.g. diverged: never best
        (Scripted([0.4, float("nan"), 0.7]), 0.4),
    ],
    ids=["loss", "score", "zero-first", "zero-later", "nan-first", "nan-later"],
)
def test_best_validation_model_follows_the_metric_direction(
    tmp_path, validation_fn, best
):
    solver = make_solver()
    solver.set_training(loader())
    solver.set_saving(tmp_path, "model")
    data = loader(n=2, batch_size=2)  # one batch: one value per validation
    solver.set_validation(data, 1, validation_fn, validation_fname="best.pt")
    solver.train(3)
    assert solver.model._read_min_validation(tmp_path / "best.pt") == pytest.approx(
        best
    )


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


# --------------------------------------------------------------------------
# model.set_normalisation() must actually be used by the solvers


def _capture_inputs(model):
    seen = []
    model.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    return seen


def test_training_step_normalises_input_and_returns_target_units(tmp_path):
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

    solver = make_solver()
    solver.model.set_normalisation("sample")
    solver.set_saving(tmp_path, "model")
    solver.set_validation(loader(), validation_freq=1)
    seen = _capture_inputs(solver.model)
    solver._validation_loop()
    assert all(x.min() >= 0 and x.max() <= 1 for x in seen)


def gaussian_denoiser_solver(**params):
    from LION.optimizers.GaussianDenoiserSolver import GaussianDenoiserSolver

    model = small_dncnn()
    solver_params = GaussianDenoiserSolver.default_parameters()
    for name, value in params.items():
        setattr(solver_params, name, value)
    return GaussianDenoiserSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        operator_params=Geometry.default_parameters(),
        solver_params=solver_params,
        device=torch.device("cpu"),
    )


def test_gaussian_denoiser_gives_each_image_its_noise_level():
    solver = gaussian_denoiser_solver(noise_level=[0.01, 0.2])
    solver.model = small_gsd_noise_channel()
    given = []
    convolution = solver.model._backbone_model.module
    original = convolution.forward
    convolution.forward = lambda x, noise_level: given.append(noise_level) or original(
        x, noise_level
    )
    solver.mini_batch_step(None, torch.rand(4, 1, 8, 8))
    (levels,) = given
    assert levels.shape == (4,) and levels.unique().numel() == 4

    for noise_level, expected in [
        (0.05, (0.05, 0.05)),
        ([0.05], (0.05, 0.05)),
        ([0.01, 0.1], (0.01, 0.1)),
    ]:
        solver = gaussian_denoiser_solver(noise_level=noise_level)
        assert solver._noise_range() == expected
        low, high = expected
        levels = solver._noise_levels(100, "cpu")
        assert levels.shape == (100,) and low <= levels.min() <= levels.max() <= high
        assert (levels.unique().numel() > 1) == (low < high)  # one per image


def test_gaussian_denoiser_augmentations_are_options(monkeypatch):
    for augment, erasing in [(False, False), (True, False), (False, True)]:
        import LION.optimizers.GaussianDenoiserSolver as module

        called = []
        for name in ("random_flip_rotate", "random_erasing"):
            monkeypatch.setattr(
                module, name, lambda x, name=name: called.append(name) or x
            )
        solver = gaussian_denoiser_solver(augment=augment, random_erasing=erasing)
        solver.mini_batch_step(None, torch.rand(2, 1, 8, 8))
        assert called == ["random_flip_rotate"] * augment + ["random_erasing"] * erasing

    from LION.optimizers.GaussianDenoiserSolver import GaussianDenoiserSolver

    solver = gaussian_denoiser_solver(patches=GaussianDenoiserSolver.patches(3, 4))
    seen = []
    solver.loss_fn = (
        lambda output, target: seen.append(target.shape)
        or ((output - target) ** 2).mean()
    )
    solver.mini_batch_step(None, torch.rand(2, 1, 8, 8))
    assert seen == [torch.Size([6, 1, 4, 4])]  # 3 patches of each of 2 images


class _WeightPenalty(torch.nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, model, step):
        return self.weight * sum(p.pow(2).sum() for p in model.parameters())


def test_a_model_regularization_is_added_to_the_loss():
    solver = make_solver()
    y, target = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    plain = solver.mini_batch_step(y, target)
    seen = []

    def regularization(model, step):
        seen.append(step)
        return _WeightPenalty()(model, step)

    solver.model_regularization = regularization
    regularized = solver.mini_batch_step(y, target)
    weights = sum(p.pow(2).sum() for p in solver.model.parameters())
    torch.testing.assert_close(regularized, plain + weights)
    (step,) = seen
    assert step.input is y and step.target is target
    torch.testing.assert_close(step.output, solver._forward(y))

    for penalty in [None, "firm", "prox"]:
        from LION.losses.jacobian_penalty import (
            FirmlyNonexpansivePenalty,
            ProximalDenoiserPenalty,
        )

        made = {
            None: None,
            "firm": FirmlyNonexpansivePenalty(weight=1.0, power_iterations=3),
            "prox": ProximalDenoiserPenalty(weight=1.0, power_iterations=3),
        }[penalty]
        solver = gaussian_denoiser_solver(augment=False)
        solver.model_regularization = made
        target = torch.rand(2, 1, 8, 8)
        torch.manual_seed(0)
        with_penalty = solver.mini_batch_step(None, target)
        solver.model_regularization = None
        torch.manual_seed(0)
        without = solver.mini_batch_step(None, target)
        if penalty is None:
            assert with_penalty == without
        else:  # the penalty is at least weight * (1 - epsilon)
            assert with_penalty >= without + 0.85
            with_penalty.backward()


def test_gaussian_denoiser_validates_at_fixed_noise_levels(tmp_path):
    solver = gaussian_denoiser_solver(validation_noise_levels=[0.0, 0.1])
    seen = []
    original = solver._denoise
    solver._denoise = lambda clean, level: seen.append(level.clone()) or original(
        clean, level
    )
    solver.set_saving(tmp_path, "model")
    solver.set_validation(loader(n=4, batch_size=2), 1, validation_fname="v.pt")
    value = solver.validate()
    assert np.isfinite(value)
    # each batch at each level, every image the same level
    assert [float(level[0]) for level in seen] == pytest.approx([0.0, 0.1] * 2)
    assert all(level.unique().numel() == 1 for level in seen)


# --------------------------------------------------------------------------
# ARSolver trains the regulariser; validation and testing reconstruct with a
# LearnedRegularizerReconstructor made outside it (here with an identity "CT"
# operator)


@pytest.fixture
def identity_ct(monkeypatch):
    monkeypatch.setattr(
        Geometry, "make_operator", lambda self: IdentityStandIn((1, 8, 8))
    )


def make_ar_solver(**params):
    model = small_dncnn()
    solver_params = ar_module.ARSolver.default_parameters()
    for name, value in params.items():
        setattr(solver_params, name, value)
    return ar_module.ARSolver(
        model,
        torch.optim.Adam(model.parameters()),
        operator_params=Geometry.default_parameters(),
        device=torch.device("cpu"),
        solver_params=solver_params,
        verbose=False,
    )


def learned_reconstructor(**settings):
    """Made outside the solver, without a model."""
    from LION.reconstructors import LearnedRegularizerReconstructor

    parameters = LearnedRegularizerReconstructor.default_parameters()
    parameters.n_steps, parameters.step_size = 2, 1e-2
    for name, value in settings.items():
        setattr(parameters, name, value)
    return LearnedRegularizerReconstructor(
        IdentityStandIn((1, 8, 8)), parameters=parameters
    )


def test_training_step(identity_ct):
    solver = make_ar_solver()
    solver.set_training(loader(n=4, batch_size=2))
    solver.train_loss = None
    loss = solver.mini_batch_step(*next(iter(loader(n=2, batch_size=2))))
    assert torch.isfinite(loss)


def test_any_solver_can_validate_through_a_reconstructor(tmp_path, identity_ct):
    """The reconstructor is optional for other solvers: given one, validation
    reconstructs through it with the model being trained."""
    solver = make_solver()
    reconstructor = learned_reconstructor()
    seen = []
    original = reconstructor.reconstruct
    reconstructor.reconstruct = lambda d, verbose=1: seen.append(
        reconstructor.model
    ) or original(d)
    solver.set_saving(tmp_path, "model")
    solver.set_validation(
        loader(n=2, batch_size=2), 1, torch.nn.MSELoss(), reconstructor=reconstructor
    )
    solver.validate()
    assert seen == [solver.model]

    solver = make_ar_solver()
    reconstructor = learned_reconstructor()
    calls = []
    original = reconstructor.reconstruct
    reconstructor.reconstruct = lambda d, verbose=1: calls.append(d.shape) or original(
        d, verbose=verbose
    )
    solver.set_saving(tmp_path, "model")
    solver.set_validation(
        loader(n=4, batch_size=2), validation_freq=1, reconstructor=reconstructor
    )
    assert solver.validate() > 0
    assert calls == [torch.Size([2, 1, 8, 8])] * 2
    # recorded with the training, by its spec (not the model)
    written = solver.training_record("final", 0).serialize()["validation"]
    assert written["reconstructor"]["parameters"]["n_steps"] == 2
    assert "model" not in written["reconstructor"]


# --------------------------------------------------------------------------
# Sparse2Inverse sub-operators for any number of angular splits


def mean_reconstruction(y, op):
    """AnglesOp's default reconstruction: the sinogram's mean, counted."""
    op.reconstructed += 1
    return y.mean() * torch.ones(y.shape[0], *op.domain_shape)


class AnglesOp:
    """A stand-in CT operator: the angles it was made with. It projects an
    image to a constant sinogram over its angles (the image's mean), and its
    default reconstruction is the sinogram's mean."""

    def __init__(self, geometry):
        self.angles = np.asarray(geometry.angles)
        self.domain_shape = tuple(geometry.image_shape)
        self.reconstructed = 0

    def __call__(self, image):
        return image.mean() * torch.ones(1, 1, len(self.angles), 8)

    def autograd(self):
        return self, None

    def default_reconstruction_fn(self):
        return mean_reconstruction


@pytest.fixture
def fake_ct(monkeypatch):
    """CT operators are replaced by AnglesOp."""
    monkeypatch.setattr(Geometry, "make_operator", lambda self: AnglesOp(self))
    monkeypatch.setattr(LIONmodel, "_make_operator", lambda self, autograd=False: None)


def make_s2i_solver(splits, operator_params, inputs=None, recon_fn=None):
    params = Sparse2InverseSolver.default_parameters()
    params.sino_split_count = splits
    if inputs is not None:
        params.input_split_count = inputs
    if recon_fn is not None:
        params.recon_fn = recon_fn
    model = small_dncnn()
    return Sparse2InverseSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        solver_params=params,
        operator_params=operator_params,
        device=torch.device("cpu"),
        verbose=False,
    )


def test_inputs_and_target_projectors(fake_ct, monkeypatch):
    for splits, inputs in [(3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (5, 2), (5, 3)]:
        operator_params = Geometry.default_parameters()  # 360 angles
        solver = make_s2i_solver(splits, operator_params, inputs)
        angles = np.asarray(operator_params.angles)

        # the single input sinogram is split into interleaved angle subsets
        assert len(solver.sub_ops) == splits
        for i, sub_op in enumerate(solver.sub_ops):
            np.testing.assert_array_equal(sub_op.angles, angles[i::splits])

        # every combination of `inputs` splits is a network input...
        assert solver.split_combinations == list(
            itertools.combinations(range(splits), inputs)
        )
        # ...and the target is the projection onto all the remaining angles
        for combo in solver.split_combinations:
            target = tuple(s for s in range(splits) if s not in combo)
            expected = sorted(i for s in target for i in range(s, len(angles), splits))
            np.testing.assert_array_equal(
                solver.combo_ops_autograd[target].angles, angles[expected]
            )
        # the solver's own geometry is not modified
        assert len(solver.operator_params.angles) == len(angles)

    # Noise2Inverse splits the geometry and the sinograms through the operator
    # parameters, and reconstructs each split with its own operator's default
    # reconstruction.
    from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver

    operator_params = Geometry.default_parameters()  # 360 angles
    operator_params.image_shape = [1, 8, 8]
    angles = np.asarray(operator_params.angles)
    params = Noise2InverseSolver.default_parameters()  # no recon_fn
    solver = make_n2i_solver(monkeypatch, params, operator_params)
    for j, sub_op in enumerate(solver.sub_ops):
        np.testing.assert_array_equal(sub_op.angles, angles[j::4])
    assert len(solver.operator_params.angles) == len(angles)  # not changed

    out = solver.reconstruct(torch.rand(2, 1, 360, 8))
    assert out.shape == (2, 1, 8, 8)
    assert [op.reconstructed for op in solver.sub_ops] == [1, 1, 1, 1]


@pytest.mark.parametrize("splits, inputs", [(4, 2), (5, 3)])
def test_training_step_uses_each_combination(fake_ct, monkeypatch, splits, inputs):
    """One mini-batch: p sub-reconstructions per combination, averaged, then
    compared with the held-out projections."""
    calls = []

    def recon_fn(sub_sino, op):
        calls.append(sub_sino.shape[2])
        return sub_sino.mean() * torch.ones(sub_sino.shape[0], 1, 8, 8)

    operator_params = Geometry.default_parameters()  # 360 angles
    solver = make_s2i_solver(splits, operator_params, inputs, recon_fn=recon_fn)
    sinos = torch.rand(2, 1, 360, 8)
    loss = solver.mini_batch_step(sinos, None)
    assert torch.isfinite(loss) and loss > 0
    n_combinations = len(list(itertools.combinations(range(splits), inputs)))
    assert len(calls) == n_combinations * inputs
    assert all(n == 360 // splits for n in calls)


def test_validation_and_test_use_the_reconstruction(fake_ct, tmp_path):
    """Self-supervised solvers validate and test their reconstruct() output
    (they used to return 0 from validate(), which crashed train())."""
    operator_params = Geometry.default_parameters()
    operator_params.image_shape = [1, 8, 8]
    calls = []

    def recon_fn(sub_sino, op):
        calls.append(sub_sino.shape)
        return sub_sino.mean() * torch.ones(sub_sino.shape[0], 1, 8, 8)

    solver = make_s2i_solver(4, operator_params, 2, recon_fn=recon_fn)
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


def test_sparse2inverse_reconstructs_each_split_with_its_operator(fake_ct):
    """Without recon_fn, each split is reconstructed with its operator's
    default reconstruction."""
    operator_params = Geometry.default_parameters()
    operator_params.image_shape = [1, 8, 8]
    solver = make_s2i_solver(4, operator_params, 2)
    out = solver.reconstruct(torch.rand(2, 1, 360, 8))
    assert out.shape == (2, 1, 8, 8)
    # each split is in 3 of the 6 pairs
    assert [op.reconstructed for op in solver.sub_ops] == [3, 3, 3, 3]


def test_the_default_recon_fn_is_replaced_by_the_operators(fake_ct):
    """When the solver is made, "default" becomes the operator's default
    reconstruction function, so the record says which was used. A given
    recon_fn is kept, and another string is refused."""
    solver = make_s2i_solver(4, Geometry.default_parameters(), 2)
    assert solver.solver_params.recon_fn is mean_reconstruction
    record = spec.encode(solver.solver_params)
    assert record["recon_fn"]["_target_"].endswith("mean_reconstruction")

    given = functools.partial(mean_reconstruction)
    solver = make_s2i_solver(4, Geometry.default_parameters(), 2, recon_fn=given)
    assert solver.solver_params.recon_fn is given

    with pytest.raises(ValueError, match='"default"'):
        make_s2i_solver(4, Geometry.default_parameters(), 2, recon_fn="fdk")


# --------------------------------------------------------------------------
# Noise2Inverse


def make_n2i_solver(monkeypatch, params, operator_params=None):
    import LION.optimizers.Noise2InverseSolver as n2i

    monkeypatch.setattr(Geometry, "make_operator", lambda self: AnglesOp(self))
    model = small_dncnn()
    return n2i.Noise2InverseSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=1e-2),
        torch.nn.MSELoss(),
        solver_params=params,
        operator_params=operator_params or Geometry.default_parameters(),
        device=torch.device("cpu"),
        verbose=False,
    )


def test_noise2inverse_one_X_strategy(monkeypatch):
    """1:X: each split in turn is the input, the mean of the others the
    target (the paper's other strategy)."""
    from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver

    assert Noise2InverseSolver.one_X_strategy(4) == [
        [2, 3, 4],
        [1, 3, 4],
        [1, 2, 4],
        [1, 2, 3],
    ]
    operator_params = Geometry.default_parameters()  # 360 angles
    operator_params.image_shape = [1, 8, 8]
    params = Noise2InverseSolver.default_parameters()
    params.cali_J = Noise2InverseSolver.one_X_strategy(params.sino_split_count)
    solver = make_n2i_solver(monkeypatch, params, operator_params)
    y = torch.rand(2, 1, 360, 8)
    assert torch.isfinite(solver.mini_batch_step(y, None))
    assert solver.reconstruct(y).shape == (2, 1, 8, 8)


# --------------------------------------------------------------------------
# Equivariant imaging


def test_equivariant_solver_gradient_goes_through_the_second_fdk(monkeypatch):
    """For image-input models, the equivariance term's gradient flows through
    the FDK of the transformed image (the input of the second network pass)."""
    import LION.optimizers.EquivariantSolver as ei

    calls = []

    def fake_fdk(sino, op, differentiable=False):
        calls.append(differentiable)
        return sino if differentiable else sino.detach()

    monkeypatch.setattr(Geometry, "make_operator", lambda self: IdentityStandIn())
    monkeypatch.setattr(ei, "fdk", fake_fdk)

    model = small_dncnn()
    solver = ei.EquivariantSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=1e-2),
        torch.nn.MSELoss(),
        Geometry.default_parameters(),
        device=torch.device("cpu"),
    )
    loss = solver.mini_batch_step(torch.rand(2, 1, 8, 8), None)
    loss.backward()
    assert calls == [False, True]  # the data's FDK needs no gradient


# --------------------------------------------------------------------------
# An image-domain model's input: FDK, or its own initial reconstruction


def adjoint_reconstruction(sino, op, scale=1.0):
    """A reconstruction for tests (any (sinogram, operator) function will do)."""
    return scale * torch.stack([op.adjoint(s) for s in sino])


def test_image_models_get_their_initial_reconstruction(tmp_path):
    import functools

    from LION.optimizers.LIONsolver import model_forward
    from LION.utils.parameter import LIONParameter
    from pytests.helpers import ImageMatrixOp

    torch.manual_seed(0)
    op = ImageMatrixOp(torch.randn(30, 64), (1, 8, 8))
    y = torch.randn(2, 30)
    model = small_dncnn().eval()
    model.set_initial_reconstruction(functools.partial(adjoint_reconstruction, scale=2))
    expected = model(adjoint_reconstruction(y, op, scale=2))
    torch.testing.assert_close(model_forward(model, y, op), expected)

    # saved and made again with the model
    model.save(tmp_path / "model", dataset=LIONParameter(), training=LIONParameter())
    loaded, _, _ = type(model).load(
        tmp_path / "model", supress_warnings=True, allow=["pytests."]
    )
    loaded = loaded.cpu().eval()
    torch.testing.assert_close(model_forward(loaded, y, op), expected)

    import functools

    from LION.operators.Operator import Operator
    from LION.optimizers.LIONsolver import model_forward
    from pytests.helpers import ImageMatrixOp

    torch.manual_seed(0)
    op = ImageMatrixOp(torch.randn(30, 64), (1, 8, 8))
    y = torch.randn(2, 30)
    model = small_dncnn().eval()
    assert model.model_parameters.initial_reconstruction == "default"
    expected = model(op.default_reconstruction(y))
    torch.testing.assert_close(model_forward(model, y, op), expected)
    # replaced by the operator's, so the model is saved with the function used
    assert model.model_parameters.initial_reconstruction is helpers_adjoint

    model.set_initial_reconstruction(functools.partial(adjoint_reconstruction, scale=2))
    assert not torch.allclose(model_forward(model, y, op), expected)
    model.set_initial_reconstruction(None)  # back to the operator's
    assert model.model_parameters.initial_reconstruction == "default"
    torch.testing.assert_close(model_forward(model, y, op), expected)

    with pytest.raises(NotImplementedError, match="no default reconstruction"):
        Operator.default_reconstruction_fn(op)  # an operator without one
