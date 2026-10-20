"""LearnedRegularizerReconstructor with a matrix operator (no CT needed)."""

import pytest
import torch

from LION.models.CNNs.dncnn import DnCNN
from LION.operators import Operator
from LION.reconstructors import LearnedRegularizerReconstructor, estimate_alpha


class ImageMatrixOp(Operator):
    """y = A vec(x) for (1, H, W) images x, with "sinograms" of shape (1, 6, 5)."""

    def __init__(self, A, shape):
        super().__init__()
        self.A, self._shape = A, shape

    def __call__(self, x, out=None):
        return self.forward(x)

    def forward(self, x):
        return (self.A @ x.reshape(-1)).reshape(self.range_shape)

    def adjoint(self, y):
        return (self.A.T @ y.reshape(-1)).reshape(self._shape)

    @property
    def domain_shape(self):
        return self._shape

    @property
    def range_shape(self):
        return (1, 6, 5)


@pytest.fixture
def problem():
    torch.manual_seed(0)
    shape = (1, 4, 4)
    A = torch.randn(30, 16, dtype=torch.float64) / 4
    op = ImageMatrixOp(A, shape)
    y = (A @ torch.rand(16, dtype=torch.float64)).reshape(op.range_shape)
    return op, A, y


def quadratic(x):
    """R(x) = 0.5 ||x||^2, so the minimiser solves (A^T A + alpha I) x = A^T y."""
    return 0.5 * torch.sum(x**2)


def test_converges_to_the_minimiser(problem):
    op, A, y = problem
    alpha = 0.5
    recon = LearnedRegularizerReconstructor(
        op, quadratic, alpha=alpha, n_steps=2000, step_size=1.0
    )
    x = recon.reconstruct_sample(y)
    expected = torch.linalg.solve(
        A.T @ A + alpha * torch.eye(16, dtype=A.dtype), A.T @ y.reshape(-1)
    )
    assert x.shape == op.domain_shape
    torch.testing.assert_close(x.reshape(-1), expected, atol=1e-6, rtol=1e-6)


def test_energy_decreases_monotonically(problem):
    op, _, y = problem
    recon = LearnedRegularizerReconstructor(
        op, quadratic, alpha=0.5, n_steps=1, step_size=10.0
    )
    x = recon.initial_guess(y)
    energies = [recon.energy(x, y).item()]
    for _ in range(20):
        x = recon.reconstruct_sample(y, x0=x)
        energies.append(recon.energy(x, y).item())
    assert all(b <= a for a, b in zip(energies, energies[1:]))


def test_batches_are_reconstructed_per_sample(problem):
    op, _, y = problem
    recon = LearnedRegularizerReconstructor(
        op, quadratic, alpha=0.5, n_steps=50, step_size=1.0
    )
    batch = torch.stack([y, 2 * y])  # (B, C, angles, detectors)
    out = recon.reconstruct(batch)
    assert out.shape == (2, *op.domain_shape)
    torch.testing.assert_close(out[0], recon.reconstruct_sample(y))
    torch.testing.assert_close(out[1], recon.reconstruct_sample(2 * y))


def small_regularizer():
    params = DnCNN.default_parameters()
    params.blocks, params.int_channels = 3, 4
    return DnCNN(params).double()


def test_lionmodel_regularizer_works_under_no_grad_without_touching_weights(problem):
    op, _, y = problem
    model = small_regularizer().eval()
    seen = []
    model.register_forward_pre_hook(lambda m, args: seen.append(tuple(args[0].shape)))
    recon = LearnedRegularizerReconstructor(
        op, model, alpha=0.1, n_steps=3, step_size=1.0
    )
    with torch.no_grad():
        x = recon.reconstruct_sample(y)
    assert x.shape == op.domain_shape and not x.requires_grad
    assert set(seen) == {(1, *op.domain_shape)}  # LIONmodels get a batch of one
    assert all(p.grad is None for p in model.parameters())


def test_estimate_alpha_is_mean_over_samples(problem):
    op, A, _ = problem
    torch.manual_seed(1)
    images = torch.rand(5, 1, 4, 4, dtype=torch.float64)
    sinos = torch.stack([op(x) for x in images]) + 0.01
    loader = [(sinos[:3], images[:3]), (sinos[3:], images[3:])]  # unequal batches
    expected = torch.stack(
        [torch.linalg.vector_norm(op.adjoint(op(x) - s)) for s, x in zip(sinos, images)]
    ).mean()
    assert estimate_alpha(op, loader) == pytest.approx(expected.item())


def test_rejects_bad_backtracking(problem):
    with pytest.raises(ValueError):
        LearnedRegularizerReconstructor(problem[0], quadratic, backtracking=1.0)


def test_explicit_gradient_matches_autograd(problem):
    """grad E = A^T(Ax - y) + alpha grad R, with the data term written out."""
    op, _, y = problem
    model = small_regularizer().eval()
    recon = LearnedRegularizerReconstructor(op, model, alpha=0.3)
    x = torch.rand(op.domain_shape, dtype=torch.float64, requires_grad=True)
    energy = 0.5 * torch.sum((op(x) - y) ** 2) + 0.3 * model(x.unsqueeze(0)).sum()
    (expected,) = torch.autograd.grad(energy, x)
    torch.testing.assert_close(recon.gradient(x, y), expected)
    torch.testing.assert_close(recon.energy(x, y), energy.detach())
