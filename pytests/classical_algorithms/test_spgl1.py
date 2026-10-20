"""Tests for SPGL1."""

import torch
from LION.classical_algorithms.spgl1_torch import spgl1_torch
from LION.operators import CompositeOp, PhotocurrentMapOp, Subsampler, Wavelet2D


def test_spgl1() -> None:
    device = "cpu"
    # Image size
    J = 4  # 16x16 images
    H = W = 1 << J
    coarseJ = J - 1
    delta = 1.0 / 4

    # Wavelet transform Psi
    wavelet = Wavelet2D((H, W), wavelet_name="db4", device=device)

    # Photocurrent mapping operator Phi
    subsampler = Subsampler(n=H * W, coarseJ=coarseJ, delta=delta)
    phi = PhotocurrentMapOp(J=J, subsampler=subsampler, device=device)

    # Composite operator A = Phi Psi^{-1}
    A_op = CompositeOp(wavelet, phi, device=device)

    x = torch.rand(A_op.domain_shape, device=device)
    y = phi(x)

    assert spgl1_torch(op=A_op, y=y, iter_lim=1) is not None
