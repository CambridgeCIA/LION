"""Tests for FISTA L1."""

import pytest
import torch
from LION.classical_algorithms.fista import fista_l1
from LION.operators.CompositeOp import CompositeOp
from LION.operators.multilevel_sample import multilevel_sample
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.operators.Wavelet2D import Wavelet2D


def test_fista_l1() -> None:
    """Test reconstruction with FISTA L1 with no regularization (lam=0) simply
    gives the least squares solution.

    This test runs specifically on CPU device.
    """
    device = torch.device("cpu")
    # Create a random image
    J = 4  # 16x16 image
    H = W = 1 << J
    coarseJ = J - 1
    delta = 1.0 / 4
    image = torch.rand((1, 1, H, W), dtype=torch.float32, device=device)

    # Wavelet transform Psi
    wavelet = Wavelet2D((H, W), wavelet_name="db4", device=device)

    # Photocurrent mapping operator Phi
    sampled_indices = multilevel_sample(
        J=J, num_samples=int(delta * H * W), coarse_J=coarseJ, alpha=1.0
    )
    phi = PhotocurrentMapOp(J=J, sampled_indices=sampled_indices, device=device)

    # Composite operator A = Phi Psi^{-1}
    A_op = CompositeOp(wavelet, phi, device=device)

    # Measurements y (replace with real photocurrent data)
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


@pytest.mark.cuda  # Add argument `-m "not cuda"` to the `pytest` command line to skip this test
def test_fista_l1_cuda() -> None:
    """Test reconstruction with FISTA L1 with no regularization (lam=0) simply
    gives the least squares solution.

    This test runs specifically on CUDA device.
    """
    device = torch.device("cuda")
    # Create a random image
    J = 4  # 16x16 images
    H = W = 1 << J
    coarseJ = J - 1
    delta = 1.0 / 4
    image = torch.rand((1, 1, H, W), dtype=torch.float32, device=device)

    # Wavelet transform Psi
    wavelet = Wavelet2D((H, W), wavelet_name="db4", device=device)

    # Photocurrent mapping operator Phi
    sampled_indices = multilevel_sample(
        J=J, num_samples=int(delta * H * W), coarse_J=coarseJ, alpha=1.0
    )
    phi = PhotocurrentMapOp(J=J, sampled_indices=sampled_indices, device=device)

    # Composite operator A = Phi Psi^{-1}
    A_op = CompositeOp(wavelet, phi, device=device)

    # Measurements y (replace with real photocurrent data)
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
