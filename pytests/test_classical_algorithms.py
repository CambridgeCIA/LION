# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Classical algorithms without CT: batching of per-sample reconstructions,
conjugate gradient, FISTA and SPGL1. FDK/FBP with tomosipo is in test_ct.py."""

import sys
import types

import pytest
import torch

from LION.classical_algorithms import fista_l1
from LION.operators.ct._batching import batched_reconstruction
from LION.classical_algorithms.conjugate_gradient import conjugate_gradient
from LION.classical_algorithms.spgl1_torch import spgl1_torch
from LION.operators import CompositeOp, Wavelet2D
from LION.operators.pcm import PhotocurrentMapOp, Subsampler

# --------------------------------------------------------------------------
# Batching (fdk/sirt/tv_min reconstruct one sample at a time)


class FakeCTOp:
    domain_shape = (1, 4, 4)


def _sum_recon(op, sino_i, i):
    return sino_i.sum() * torch.ones(op.domain_shape)


def test_batched_and_single_sample(monkeypatch):
    sino = torch.rand(3, 1, 5, 6)
    out = batched_reconstruction(_sum_recon, sino, FakeCTOp())
    assert out.shape == (3, 1, 4, 4)
    torch.testing.assert_close(out[1], sino[1].sum() * torch.ones(1, 4, 4))
    single = batched_reconstruction(_sum_recon, sino[0], FakeCTOp())
    assert single.shape == (1, 4, 4)

    # SIRT and TV-min call ts_algorithms once per sample, with that sample, and
    # refuse an empty batch.
    for name in ["sirt", "tv_min"]:
        from LION.operators import ct

        given = []

        def one_sample(op, sino, *rest, **kwargs):
            given.append(sino)
            return torch.zeros(op.domain_shape)

        module = types.ModuleType("ts_algorithms")
        module.sirt = module.tv_min2d = one_sample
        monkeypatch.setitem(sys.modules, "ts_algorithms", module)
        recon = getattr(ct, name)
        arguments = {"num_iterations": 2} if name == "sirt" else {"lam": 0.1}

        sino = torch.rand(2, 1, 5, 6)
        assert recon(sino, FakeCTOp(), **arguments).shape == (2, 1, 4, 4)
        assert [g.shape for g in given] == [(1, 5, 6), (1, 5, 6)]
        torch.testing.assert_close(given[1], sino[1])
        with pytest.raises(ValueError):
            recon(torch.rand(0, 1, 5, 6), FakeCTOp(), **arguments)


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


def test_solution_matches_direct_solve(spd_system):
    A, b = spd_system
    x = conjugate_gradient(
        lambda v: A @ v, b, torch.zeros_like(b), max_iter=500, tol=1e-12
    )
    torch.testing.assert_close(x, torch.linalg.solve(A, b))


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
    )
    recon = wavelet.inverse(w_hat)
    zero_filled = phi.pseudo_inv(y)  # A(y) is the least squares solution
    torch.testing.assert_close(recon, zero_filled, atol=1e-6, rtol=1e-6)


def test_spgl1() -> None:
    _, phi, A_op = wavelet_and_pcm("cpu")
    x = torch.rand(A_op.domain_shape, device="cpu")
    y = phi(x)
    assert spgl1_torch(op=A_op, y=y, iter_lim=1) is not None
