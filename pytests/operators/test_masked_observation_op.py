"""Tests for dense masked observations on spectral cubes and matrices."""

import numpy as np
import pytest
import torch

from LION.operators.MaskedObservationOp import MaskedObservationOp


@pytest.mark.parametrize("shape", [(4, 5), (3, 4, 5)])
def test_masked_observation_forward_and_adjoint(shape):
    torch.manual_seed(9)
    mask = torch.rand(shape) < 0.4
    op = MaskedObservationOp(mask)

    x = torch.randn(shape, dtype=torch.float64)
    y = torch.randn(shape, dtype=torch.float64)
    expected = torch.where(mask, x, torch.zeros_like(x))

    assert op.domain_shape == shape
    assert op.range_shape == shape
    torch.testing.assert_close(op(x), expected)
    torch.testing.assert_close(op.gram(x), expected)
    torch.testing.assert_close(op.adjoint(y), torch.where(mask, y, torch.zeros_like(y)))
    torch.testing.assert_close(
        torch.vdot(op(x).flatten(), y.flatten()),
        torch.vdot(x.flatten(), op.adjoint(y).flatten()),
    )


def test_masked_observation_batched_and_complex():
    mask = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=np.int8)
    op = MaskedObservationOp(mask)
    x = torch.randn((3, *mask.shape), dtype=torch.complex128)

    assert op.mask.dtype == torch.bool
    torch.testing.assert_close(op(x), x * torch.tensor(mask, dtype=torch.bool))
    torch.testing.assert_close(op.adjoint(op(x)), op(x))


def test_masked_observation_autograd_and_device():
    mask = torch.tensor([[[True, False], [False, True]]])
    op = MaskedObservationOp(mask).to_device(torch.device("cpu"))
    x = torch.randn(*mask.shape, requires_grad=True)
    op(x).sum().backward()

    assert op.mask.device.type == "cpu"
    torch.testing.assert_close(x.grad, mask.to(dtype=x.dtype))


@pytest.mark.parametrize(
    "mask",
    [torch.ones(5), torch.ones(2, 3, 4, 5), torch.tensor([[1.0, 0.5]])],
)
def test_masked_observation_rejects_invalid_masks(mask):
    with pytest.raises(ValueError):
        MaskedObservationOp(mask)


def test_masked_observation_rejects_wrong_input_shape():
    op = MaskedObservationOp(torch.ones(2, 3, 4))
    with pytest.raises(ValueError, match="Expected trailing dimensions"):
        op(torch.ones(2, 3, 5))
