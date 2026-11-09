# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""The penalties on a denoiser's Jacobian (LION.losses.jacobian_penalty)."""

import pytest
import torch

from LION.utils.parameter import LIONParameter

from LION.losses.jacobian_penalty import (
    FirmlyNonexpansivePenalty,
    ProximalDenoiserPenalty,
    jacobian_spectral_norm,
)


def test_the_spectral_norm_is_that_of_each_images_jacobian():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
    x = torch.rand(2, 1, 5, 5)
    # the convolution's matrix: its Jacobian everywhere
    basis = torch.eye(25).reshape(25, 1, 5, 5)
    matrix = conv(basis).reshape(25, 25).T
    expected = torch.linalg.matrix_norm(matrix, ord=2)
    norms = jacobian_spectral_norm(conv, x, power_iterations=200)
    torch.testing.assert_close(norms, expected.expand(2), rtol=1e-3, atol=1e-5)

    conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
    jacobian_spectral_norm(conv, torch.rand(2, 1, 5, 5), 5).sum().backward()
    assert conv.weight.grad is not None and conv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("scale", [0.3, 0.9])
def test_the_penalties_follow_their_papers(scale):
    """For D = scale * I: ||J_{2D-I}|| = |2 scale - 1|, ||J_{I-D}|| = 1 - scale."""
    denoise = lambda x: scale * x  # noqa: E731
    clean = torch.rand(3, 1, 4, 4)
    noisy = clean + 0.1 * torch.randn_like(clean)
    step = LIONParameter()
    step.input, step.output, step.target = noisy, denoise(noisy), clean
    step.forward = denoise
    firm = FirmlyNonexpansivePenalty(weight=2.0, epsilon=0.05)
    expected = 2.0 * max((2 * scale - 1) ** 2, 1 - 0.05)
    torch.testing.assert_close(firm(None, step), torch.tensor(expected))
    prox = ProximalDenoiserPenalty(weight=3.0, epsilon=0.1)
    expected = 3.0 * max(1 - scale, 1 - 0.1)
    torch.testing.assert_close(prox(None, step), torch.tensor(expected))

    step.target = None  # a solver without clean images
    with pytest.raises(ValueError, match="needs the clean images"):
        firm(None, step)
