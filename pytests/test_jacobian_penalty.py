# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""The penalties on a denoiser's Jacobian (LION.losses.jacobian_penalty)."""

import pytest
import torch

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


def test_the_spectral_norm_is_differentiable_in_the_parameters():
    conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
    jacobian_spectral_norm(conv, torch.rand(2, 1, 5, 5), 5).sum().backward()
    assert conv.weight.grad is not None and conv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("scale", [0.3, 0.9])
def test_the_penalties_follow_their_papers(scale):
    """For D = scale * I: ||J_{2D-I}|| = |2 scale - 1|, ||J_{I-D}|| = 1 - scale."""
    denoise = lambda x: scale * x  # noqa: E731
    clean = torch.rand(3, 1, 4, 4)
    noisy = clean + 0.1 * torch.randn_like(clean)
    firm = FirmlyNonexpansivePenalty(weight=2.0, epsilon=0.05)
    expected = 2.0 * max((2 * scale - 1) ** 2, 1 - 0.05)
    torch.testing.assert_close(
        firm(denoise, clean, noisy, denoise(noisy)), torch.tensor(expected)
    )
    prox = ProximalDenoiserPenalty(weight=3.0, epsilon=0.1)
    expected = 3.0 * max(1 - scale, 1 - 0.1)
    torch.testing.assert_close(
        prox(denoise, clean, noisy, denoise(noisy)), torch.tensor(expected)
    )


def test_the_defaults_are_the_papers():
    firm, prox = FirmlyNonexpansivePenalty(), ProximalDenoiserPenalty()
    assert (firm.weight, firm.epsilon) == (1e-5, 0.05)
    assert (prox.weight, prox.epsilon, prox.power_iterations) == (1e-3, 0.1, 50)
    assert "10.1137/20M1387961" in FirmlyNonexpansivePenalty.cite()
    assert "9483--9505" in ProximalDenoiserPenalty.cite()


def test_a_penalty_is_recorded_and_made_again():
    from LION.utils import spec

    penalty = ProximalDenoiserPenalty(weight=1e-2)
    written = spec.encode(penalty)
    assert written == {
        "_target_": "LION.losses.jacobian_penalty.ProximalDenoiserPenalty",
        "weight": 1e-2,
        "epsilon": 0.1,
        "power_iterations": 50,
    }
    made = spec.build(written)
    assert isinstance(made, ProximalDenoiserPenalty) and made.weight == 1e-2
