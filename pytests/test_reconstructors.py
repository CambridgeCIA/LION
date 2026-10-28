"""Reconstructors (PnP, learned regulariser) with a matrix operator, no CT needed."""

import pytest
import torch

from LION.reconstructors import (
    LearnedRegularizerReconstructor,
    LIONReconstructor,
    PnP,
    estimate_alpha,
)
from LION.utils import spec
from LION.utils.parameter import LIONParameter
from pytests.helpers import ImageMatrixOp, ScaleOp, small_dncnn

# --------------------------------------------------------------------------
# PnP: with an identity prior every algorithm must converge to the
# least-squares solution


@pytest.fixture
def pnp_problem():
    torch.manual_seed(0)
    shape = (1, 4, 4)
    A = torch.randn(40, 16) / 4
    x_true = torch.rand(shape)
    op = ImageMatrixOp(A, shape)
    return op, x_true, op(x_true)


class Identity:
    def __init__(self):
        self.shapes = []

    def __call__(self, x):
        self.shapes.append(tuple(x.shape))
        return x


def pnp(op, model, algorithm, **settings):
    """A PnP with ``algorithm``'s default settings, but ``settings``."""
    parameters = PnP.default_parameters(algorithm)
    for name, value in settings.items():
        setattr(parameters, name, value)
    return PnP(op, model, parameters)


def learned(op, model, **settings):
    """A learned regulariser reconstructor with the default settings, but
    ``settings``."""
    parameters = LearnedRegularizerReconstructor.default_parameters()
    for name, value in settings.items():
        setattr(parameters, name, value)
    return LearnedRegularizerReconstructor(op, model, parameters)


ALGORITHM_KWARGS = {
    "ADMM": dict(eta=1.0, max_iter=200, cg_max_iter=50, cg_tol=1e-10),
    "HQS": dict(mu=1.0, max_iter=200, cg_max_iter=50, cg_tol=1e-10),
    "FBS": dict(max_iter=3000),
}


@pytest.mark.parametrize("algorithm", ["ADMM", "HQS", "FBS"])
def test_identity_prior_gives_least_squares(pnp_problem, algorithm):
    op, x_true, y = pnp_problem
    prior = Identity()
    recon = pnp(op, prior, algorithm, **ALGORITHM_KWARGS[algorithm]).reconstruct(y)
    assert recon.shape == op.domain_shape
    torch.testing.assert_close(recon, x_true, atol=1e-3, rtol=1e-3)
    # plain callables always get the image in the operator's domain shape
    assert set(prior.shapes) == {op.domain_shape}


@pytest.mark.parametrize("algorithm", ["ADMM", "HQS", "FBS"])
def test_lionmodel_prior_gets_a_batch(pnp_problem, algorithm):
    op, _, y = pnp_problem
    model = small_dncnn().eval()
    seen = []
    model.register_forward_pre_hook(lambda m, args: seen.append(tuple(args[0].shape)))
    kwargs = dict(ALGORITHM_KWARGS[algorithm], max_iter=2)
    with torch.no_grad():
        recon = pnp(op, model, algorithm, **kwargs).reconstruct(y)
    assert recon.shape == op.domain_shape
    assert set(seen) == {(1, *op.domain_shape)}


def test_vector_measurements_are_batched(pnp_problem):
    """A batch of vector measurements (B, M) is a batch, not one sample: the
    batch is found from the operator's range shape, not by assuming CT."""
    op, x_true, y = pnp_problem
    assert y.dim() == 1  # (M,)
    recon = pnp(op, Identity(), "HQS", **ALGORITHM_KWARGS["HQS"])
    out = recon.reconstruct(torch.stack([y, 2 * y]))
    assert out.shape == (2, *op.domain_shape)
    torch.testing.assert_close(out[0], x_true, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out[1], 2 * x_true, atol=1e-3, rtol=1e-3)


def test_unknown_algorithm(pnp_problem):
    with pytest.raises(ValueError, match="Unknown algorithm"):
        PnP.default_parameters("nope")
    parameters = PnP.default_parameters()
    parameters.algorithm = "nope"
    with pytest.raises(ValueError, match="Unknown algorithm"):
        PnP(pnp_problem[0], Identity(), parameters)


def test_fbs_computes_the_operator_norm_once(pnp_problem, monkeypatch):
    import importlib

    # the module, not the PnP class that LION.reconstructors exports under that name
    pnp_module = importlib.import_module("LION.reconstructors.PnP")

    op, x_true, y = pnp_problem
    calls = []
    real = pnp_module.power_method
    monkeypatch.setattr(
        pnp_module, "power_method", lambda *a, **kw: calls.append(1) or real(*a, **kw)
    )
    recon = pnp(op, Identity(), "FBS", max_iter=2)
    for _ in range(3):  # e.g. three samples of a batch
        recon.reconstruct(y)
    assert len(calls) == 1


@pytest.mark.parametrize("algorithm", ["ADMM", "HQS", "FBS"])
def test_starting_image_can_be_given(pnp_problem, algorithm):
    op, x_true, y = pnp_problem
    kwargs = dict(ALGORITHM_KWARGS[algorithm], max_iter=1)
    recon = pnp(op, Identity(), algorithm, **kwargs)
    default = recon.reconstruct(y)
    from_truth = recon.reconstruct(y, x0=x_true.clone())
    error = lambda x: torch.linalg.vector_norm(x - x_true)
    assert error(from_truth) < error(default)


# --------------------------------------------------------------------------
# LearnedRegularizerReconstructor, with "sinograms" of shape (1, 6, 5)


@pytest.fixture
def lr_problem():
    torch.manual_seed(0)
    shape = (1, 4, 4)
    A = torch.randn(30, 16, dtype=torch.float64) / 4
    op = ImageMatrixOp(A, shape, range_shape=(1, 6, 5))
    y = (A @ torch.rand(16, dtype=torch.float64)).reshape(op.range_shape)
    return op, A, y


def quadratic(x):
    """R(x) = 0.5 ||x||^2, so the minimiser solves (A^T A + alpha I) x = A^T y."""
    return 0.5 * torch.sum(x**2)


def small_regularizer():
    return small_dncnn().double()


def test_converges_to_the_minimiser(lr_problem):
    op, A, y = lr_problem
    alpha = 0.5
    recon = learned(op, quadratic, alpha=alpha, n_steps=2000, step_size=1.0)
    x = recon.reconstruct_sample(y)
    expected = torch.linalg.solve(
        A.T @ A + alpha * torch.eye(16, dtype=A.dtype), A.T @ y.reshape(-1)
    )
    assert x.shape == op.domain_shape
    torch.testing.assert_close(x.reshape(-1), expected, atol=1e-6, rtol=1e-6)


def test_energy_decreases_monotonically(lr_problem):
    op, _, y = lr_problem
    recon = learned(op, quadratic, alpha=0.5, n_steps=1, step_size=10.0)
    x = recon.initial_guess(y)
    energies = [recon.energy(x, y).item()]
    for _ in range(20):
        x = recon.reconstruct_sample(y, x0=x)
        energies.append(recon.energy(x, y).item())
    assert all(b <= a for a, b in zip(energies, energies[1:]))


def test_batches_are_reconstructed_per_sample(lr_problem):
    op, _, y = lr_problem
    recon = learned(op, quadratic, alpha=0.5, n_steps=50, step_size=1.0)
    batch = torch.stack([y, 2 * y])  # (B, C, angles, detectors)
    out = recon.reconstruct(batch)
    assert out.shape == (2, *op.domain_shape)
    torch.testing.assert_close(out[0], recon.reconstruct_sample(y))
    torch.testing.assert_close(out[1], recon.reconstruct_sample(2 * y))


def test_lionmodel_regularizer_works_under_no_grad_without_touching_weights(
    lr_problem,
):
    op, _, y = lr_problem
    model = small_regularizer().eval()
    seen = []
    model.register_forward_pre_hook(lambda m, args: seen.append(tuple(args[0].shape)))
    recon = learned(op, model, alpha=0.1, n_steps=3, step_size=1.0)
    with torch.no_grad():
        x = recon.reconstruct_sample(y)
    assert x.shape == op.domain_shape and not x.requires_grad
    assert set(seen) == {(1, *op.domain_shape)}  # LIONmodels get a batch of one
    assert all(p.grad is None for p in model.parameters())


def test_estimate_alpha_is_mean_over_samples(lr_problem):
    op, A, _ = lr_problem
    torch.manual_seed(1)
    images = torch.rand(5, 1, 4, 4, dtype=torch.float64)
    sinos = torch.stack([op(x) for x in images]) + 0.01
    loader = [(sinos[:3], images[:3]), (sinos[3:], images[3:])]  # unequal batches
    expected = torch.stack(
        [torch.linalg.vector_norm(op.adjoint(op(x) - s)) for s, x in zip(sinos, images)]
    ).mean()
    assert estimate_alpha(op, loader) == pytest.approx(expected.item())


def test_rejects_bad_backtracking(lr_problem):
    with pytest.raises(ValueError):
        learned(lr_problem[0], quadratic, backtracking=1.0)


def test_explicit_gradient_matches_autograd(lr_problem):
    """grad E = A^T(Ax - y) + alpha grad R, with the data term written out."""
    op, _, y = lr_problem
    model = small_regularizer().eval()
    recon = learned(op, model, alpha=0.3)
    x = torch.rand(op.domain_shape, dtype=torch.float64, requires_grad=True)
    energy = 0.5 * torch.sum((op(x) - y) ** 2) + 0.3 * model(x.unsqueeze(0)).sum()
    (expected,) = torch.autograd.grad(energy, x)
    torch.testing.assert_close(recon.gradient(x, y), expected)
    torch.testing.assert_close(recon.energy(x, y), energy.detach())


def test_lionmodel_regularizer_sees_normalised_images(lr_problem):
    """The regulariser is applied with the normalisation it was trained with."""
    op, _, y = lr_problem
    model = small_regularizer().eval()
    model.set_normalisation("custom", data_range=(0.0, 10.0))
    seen = []
    model.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach()))
    recon = learned(op, model, alpha=0.1)
    x = 10 * torch.rand(op.domain_shape, dtype=torch.float64)
    recon.regularizer_value(x)
    torch.testing.assert_close(seen[-1], x.unsqueeze(0) / 10)


# --------------------------------------------------------------------------
# Parameters, specs and instantiate: a reconstructor with its parameters is
# one method, recorded and made again (the model is given separately)


def test_default_parameters_are_each_algorithms():
    assert PnP.default_parameters().algorithm == "ADMM"
    assert PnP.default_parameters("HQS").max_iter == 100
    assert PnP.default_parameters("FBS").max_iter == 10
    assert not hasattr(PnP.default_parameters("FBS"), "eta")
    # a setting missing from the parameters has its default
    parameters = LIONParameter(algorithm="HQS")
    recon = PnP(ScaleOp(1.0, (1, 4, 4)), Identity(), parameters)
    assert recon.setting("mu") == 0.1


def test_spec_is_the_operator_and_parameters_not_the_model():
    recon = pnp(ScaleOp(2.0, (1, 4, 4)), small_dncnn(), "HQS", max_iter=3)
    written = spec.to_spec(recon)
    assert set(written) == {"_target_", "operator", "parameters"}
    assert written["parameters"]["max_iter"] == 3


def test_made_again_without_a_model_then_given_one():
    op = ScaleOp(2.0, (1, 4, 4))
    recon = pnp(op, Identity(), "HQS", max_iter=3, cg_max_iter=5)
    again = LIONReconstructor.instantiate(spec.to_spec(recon), allow=["pytests."])
    assert type(again) is PnP and again.model is None
    assert vars(again.parameters) == vars(recon.parameters)
    y = op(torch.rand(1, 4, 4))
    with pytest.raises(ValueError, match="has no model"):
        again.reconstruct(y)
    again.model = Identity()
    torch.testing.assert_close(again.reconstruct(y), recon.reconstruct(y))
    with_model = PnP.instantiate(spec.to_spec(recon), Identity(), allow=["pytests."])
    torch.testing.assert_close(with_model.reconstruct(y), recon.reconstruct(y))


def test_learned_regularizer_made_again():
    recon = learned(ScaleOp(1.0, (1, 4, 4)), quadratic, alpha=0.3, n_steps=4)
    again = LearnedRegularizerReconstructor.instantiate(
        spec.to_spec(recon), quadratic, allow=["pytests."]
    )
    assert vars(again.parameters) == vars(recon.parameters)
    assert again.model is quadratic


def test_made_again_from_a_test_record(tmp_path, monkeypatch):
    """Evaluations of reconstructors are saved and reused; the reconstructor
    is made again from its test record, and a model with other weights warns."""
    import LION.evaluation.evaluation as evaluation_module
    from LION.evaluation import Evaluation
    from pytests.test_evaluation import data

    op = ScaleOp(1.0, (1, 16, 16))
    net = small_dncnn().eval()
    recon = pnp(op, net, "FBS", max_iter=2)
    evaluation = Evaluation(
        dataset=data(), operator=op, save_folder=tmp_path, device="cpu"
    )
    record = evaluation.run(recon, "pnp")
    assert len(list(tmp_path.glob("pnp-*.json"))) == 1  # saved: describable
    monkeypatch.setattr(
        evaluation_module, "scores", lambda *a, **k: pytest.fail("recomputed")
    )
    assert evaluation.run(recon, "pnp").summary.PSNR.mean == record.summary.PSNR.mean
    again = LIONReconstructor.instantiate(record, net, allow=["pytests."])
    assert type(again) is PnP and again.parameters.max_iter == 2
    with pytest.warns(UserWarning, match="other weights"):
        LIONReconstructor.instantiate(record, small_dncnn(), allow=["pytests."])
