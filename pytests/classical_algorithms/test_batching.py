"""Batching of single-sample reconstructions (fdk/sirt/tv_min) and FISTA shapes."""

import sys
import types

import pytest
import torch

from LION.classical_algorithms import fista_l1, sirt
from LION.classical_algorithms._batching import batched_reconstruction
from LION.exceptions.exceptions import NoDataException
from LION.operators import Operator


class FakeCTOp:
    domain_shape = (1, 4, 4)


def _sum_recon(op, sino_i, i):
    return sino_i.sum() * torch.ones(op.domain_shape)


def test_batched_and_single_sample():
    sino = torch.rand(3, 1, 5, 6)
    out = batched_reconstruction(_sum_recon, sino, FakeCTOp())
    assert out.shape == (3, 1, 4, 4)
    torch.testing.assert_close(out[1], sino[1].sum() * torch.ones(1, 4, 4))
    single = batched_reconstruction(_sum_recon, sino[0], FakeCTOp())
    assert single.shape == (1, 4, 4)


@pytest.mark.parametrize("shape", [(5, 6), (1, 1, 1, 5, 6)])
def test_wrong_dimensions_raise_value_error(shape):
    with pytest.raises(ValueError, match="Expected a sinogram"):
        batched_reconstruction(_sum_recon, torch.rand(*shape), FakeCTOp())


def test_empty_batch():
    with pytest.raises(NoDataException):
        batched_reconstruction(_sum_recon, torch.rand(0, 1, 5, 6), FakeCTOp())


@pytest.fixture
def fake_ts_sirt(monkeypatch):
    calls = []

    def ts_sirt(op, sino, num_iterations, mn, mx, x_init, *rest):
        calls.append(x_init)
        return torch.zeros(op.domain_shape)

    module = types.ModuleType("ts_algorithms")
    module.sirt = ts_sirt
    monkeypatch.setitem(sys.modules, "ts_algorithms", module)
    return calls


def test_sirt_uses_one_initial_guess_per_sample(fake_ts_sirt):
    x_init = torch.arange(2.0).view(2, 1, 1, 1) * torch.ones(2, *FakeCTOp.domain_shape)
    sirt(torch.rand(2, 1, 5, 6), FakeCTOp(), 3, x_init=x_init)
    assert len(fake_ts_sirt) == 2
    for i, given in enumerate(fake_ts_sirt):
        torch.testing.assert_close(given, x_init[i])


def test_sirt_shares_unbatched_initial_guess(fake_ts_sirt):
    x_init = torch.ones(FakeCTOp.domain_shape)
    sirt(torch.rand(2, 1, 5, 6), FakeCTOp(), 3, x_init=x_init)
    assert all(given is x_init for given in fake_ts_sirt)


class ImageMatrixOp(Operator):
    """y = A vec(x) for 2D images x."""

    def __init__(self, A, image_shape):
        super().__init__()
        self.A, self._shape = A, image_shape

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


def test_fista_with_image_shaped_domain():
    torch.manual_seed(0)
    A = torch.randn(40, 16)
    op = ImageMatrixOp(A, (4, 4))
    x_true = torch.randn(4, 4)
    y = op(x_true)
    x = fista_l1(op, y, lam=0.0, max_iter=2000, tol=1e-10)
    assert x.shape == (4, 4)
    torch.testing.assert_close(x, x_true, atol=1e-3, rtol=1e-3)
