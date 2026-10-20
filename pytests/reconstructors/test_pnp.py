"""PnP with a non-CT operator: with an identity prior every algorithm must
converge to the least-squares solution."""

import pytest
import torch

from LION.models.CNNs.dncnn import DnCNN
from LION.operators import Operator
from LION.reconstructors import PnP


class ImageMatrixOp(Operator):
    """y = A vec(x) for (1, H, W) images x."""

    def __init__(self, A, shape):
        super().__init__()
        self.A, self._shape = A, shape

    def __call__(self, x, out=None):
        return self.forward(x)

    def forward(self, x):
        return self.A @ x.reshape(-1)

    def adjoint(self, y):
        return (self.A.T @ y).reshape(self._shape)

    @property
    def domain_shape(self):
        return self._shape

    @property
    def range_shape(self):
        return (self.A.shape[0],)


@pytest.fixture
def problem():
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


ALGORITHM_KWARGS = {
    "ADMM": dict(eta=1.0, max_iter=200, cg_max_iter=50, cg_tol=1e-10),
    "HQS": dict(mu=1.0, max_iter=200, cg_max_iter=50, cg_tol=1e-10),
    "FBS": dict(max_iter=3000),
}


@pytest.mark.parametrize("algorithm", ["ADMM", "HQS", "FBS"])
def test_identity_prior_gives_least_squares(problem, algorithm):
    op, x_true, y = problem
    prior = Identity()
    recon = PnP(op, prior, algorithm=algorithm).reconstruct(
        y, **ALGORITHM_KWARGS[algorithm]
    )
    assert recon.shape == op.domain_shape
    torch.testing.assert_close(recon, x_true, atol=1e-3, rtol=1e-3)
    # plain callables always get the image in the operator's domain shape
    assert set(prior.shapes) == {op.domain_shape}


@pytest.mark.parametrize("algorithm", ["ADMM", "HQS", "FBS"])
def test_lionmodel_prior_gets_a_batch(problem, algorithm):
    op, _, y = problem
    params = DnCNN.default_parameters()
    params.blocks, params.int_channels = 3, 4
    model = DnCNN(params).eval()
    seen = []
    model.register_forward_pre_hook(lambda m, args: seen.append(tuple(args[0].shape)))
    kwargs = dict(ALGORITHM_KWARGS[algorithm], max_iter=2)
    with torch.no_grad():
        recon = PnP(op, model, algorithm=algorithm).reconstruct(y, **kwargs)
    assert recon.shape == op.domain_shape
    assert set(seen) == {(1, *op.domain_shape)}


def test_unknown_algorithm(problem):
    with pytest.raises(ValueError):
        PnP(problem[0], Identity(), algorithm="nope")
