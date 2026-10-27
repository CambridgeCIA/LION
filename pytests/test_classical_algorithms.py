"""Classical algorithms without CT: batching of per-sample reconstructions,
conjugate gradient, FISTA and SPGL1. FDK/FBP with tomosipo is in test_ct.py."""

import sys
import types

import pytest
import torch
from LION.classical_algorithms._batching import batched_reconstruction
from LION.classical_algorithms.conjugate_gradient import conjugate_gradient
from LION.classical_algorithms.fista import fista_l1
from LION.classical_algorithms.sirt import sirt
from LION.classical_algorithms.spgl1_torch import spgl1_torch
from LION.exceptions.exceptions import NoDataException
from LION.operators.CompositeOp import CompositeOp
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp, Subsampler
from LION.operators.Wavelet2D import Wavelet2D

from pytests.helpers import ImageMatrixOp

# --------------------------------------------------------------------------
# Batching (fdk/sirt/tv_min reconstruct one sample at a time)


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


# --------------------------------------------------------------------------
# Conjugate gradient


@pytest.fixture
def spd_system():
    torch.manual_seed(0)
    n = 100
    M = torch.randn(n, n, dtype=torch.float64)
    A = torch.eye(n, dtype=torch.float64) + M @ M.T / n
    b = torch.randn(n, dtype=torch.float64)
    return A, b


class Counter:
    def __init__(self, A):
        self.A, self.calls = A, 0

    def __call__(self, v):
        self.calls += 1
        return self.A @ v


@pytest.mark.parametrize("tol", [1e-2, 1e-6, 1e-10])
def test_stops_at_requested_relative_residual(spd_system, tol):
    A, b = spd_system
    x = conjugate_gradient(Counter(A), b, torch.zeros_like(b), max_iter=500, tol=tol)
    assert torch.norm(b - A @ x) / torch.norm(b) < tol


def test_looser_tolerance_needs_fewer_iterations(spd_system):
    A, b = spd_system
    loose, tight = Counter(A), Counter(A)
    conjugate_gradient(loose, b, torch.zeros_like(b), max_iter=500, tol=1e-2)
    conjugate_gradient(tight, b, torch.zeros_like(b), max_iter=500, tol=1e-10)
    assert loose.calls < tight.calls


def test_solution_matches_direct_solve(spd_system):
    A, b = spd_system
    x = conjugate_gradient(
        lambda v: A @ v, b, torch.zeros_like(b), max_iter=500, tol=1e-12
    )
    torch.testing.assert_close(x, torch.linalg.solve(A, b))


def test_zero_rhs_and_exact_start(spd_system):
    A, b = spd_system
    zero = torch.zeros_like(b)
    torch.testing.assert_close(
        conjugate_gradient(lambda v: A @ v, zero, b, 10, 1e-6), zero
    )
    exact = torch.linalg.solve(A, b)
    calls = Counter(A)
    conjugate_gradient(calls, b, exact, 10, 1e-6)
    assert calls.calls == 1  # only the initial residual


# --------------------------------------------------------------------------
# FISTA and SPGL1 on A = Phi Psi^{-1} (photocurrent map after inverse wavelet)

J = 4  # 16x16 images
H = W = 1 << J


def wavelet_and_pcm(device):
    wavelet = Wavelet2D((H, W), wavelet_name="db4", device=device)  # Psi
    subsampler = Subsampler(n=H * W, coarseJ=J - 1, delta=1.0 / 4)
    phi = PhotocurrentMapOp(J=J, subsampler=subsampler, device=device)  # Phi
    return wavelet, phi, CompositeOp(wavelet, phi, device=device)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_fista_l1(device) -> None:
    """FISTA L1 with no regularisation (lam=0) gives the least-squares solution."""
    wavelet, phi, A_op = wavelet_and_pcm(torch.device(device))
    image = torch.rand((1, 1, H, W), dtype=torch.float32, device=device)
    y = phi(image)
    w_hat = fista_l1(
        op=A_op,
        y=y,
        lam=0,  # No regularization, should recover the input
        max_iter=1,
        tol=1e-4,
        L=None,
        verbose=False,
        progress_bar=False,
    )
    recon = wavelet.inverse(w_hat)
    zero_filled = phi.pseudo_inv(y)  # A(y) is the least squares solution
    torch.testing.assert_close(recon, zero_filled, atol=1e-6, rtol=1e-6)


def test_fista_with_image_shaped_domain():
    torch.manual_seed(0)
    A = torch.randn(40, 16)
    op = ImageMatrixOp(A, (4, 4))
    x_true = torch.randn(4, 4)
    y = op(x_true)
    x = fista_l1(op, y, lam=0.0, max_iter=2000, tol=1e-10)
    assert x.shape == (4, 4)
    torch.testing.assert_close(x, x_true, atol=1e-3, rtol=1e-3)


def test_spgl1() -> None:
    _, phi, A_op = wavelet_and_pcm("cpu")
    x = torch.rand(A_op.domain_shape, device="cpu")
    y = phi(x)
    assert spgl1_torch(op=A_op, y=y, iter_lim=1) is not None
