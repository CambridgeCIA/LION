"""Tests for the photocurrent mapping operator."""

import torch

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp, Subsampler
from pytests.helper import dotproduct_adjointness_test


def test_pcm_autograd_op_forward_and_backward():
    J = 4  # 16x16 images
    N = 1 << J
    subsampler = Subsampler(n=N * N, delta=0.25, coarseJ=J - 1)
    operator = PhotocurrentMapOp(J=J, subsampler=subsampler)

    torch.manual_seed(0)
    input_tensor = torch.randn(*operator.domain_shape, requires_grad=True)

    output_tensor = operator(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.requires_grad
    assert input_tensor.grad is None

    output_tensor.mean().backward()

    assert input_tensor.grad is not None
    assert isinstance(input_tensor.grad, torch.Tensor)
    assert input_tensor.grad.shape == input_tensor.shape
    assert torch.isfinite(input_tensor.grad).all()


def test_pcm_op_adjointness():
    """Test photocurrent mapping operator adjoint property."""
    J = 4  # 16x16 images
    N = 1 << J
    subsampler = Subsampler(n=N * N, delta=0.25, coarseJ=J - 1)
    operator = PhotocurrentMapOp(J=J, subsampler=subsampler)

    # Check the default operator shapes
    assert operator.domain_shape == (16, 16)
    assert operator.range_shape == (64,)

    # Create a test input for the forward and backward projections
    test_image = torch.rand(*operator.domain_shape)
    test_measurement = torch.rand(*operator.range_shape)
    dotproduct_adjointness_test(operator, test_image, test_measurement)

    # Test pseudoinverse runs without error
    assert operator.pseudo_inv(test_measurement) is not None
