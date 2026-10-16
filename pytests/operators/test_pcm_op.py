"""Tests for the photocurrent mapping operator."""

import torch
from LION.operators.multilevel_sample import multilevel_sample
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from pytests.helper import dotproduct_adjointness_test


def make_operator() -> PhotocurrentMapOp:
    J = 4
    N = 1 << J
    sampled_indices = multilevel_sample(
        J=J, num_samples=int(0.25 * N * N), coarse_J=J - 1, alpha=1.0
    )
    return PhotocurrentMapOp(j_order=J, s_indices=sampled_indices)


def test_pcm_autograd_op_forward_and_backward():
    operator = make_operator()

    torch.manual_seed(0)
    input_tensor = torch.randn(*operator.domain_shape, requires_grad=True)
    output_tensor = operator(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.requires_grad
    assert input_tensor.grad is None

    output_tensor.mean().backward()

    assert input_tensor.grad is not None
    assert input_tensor.grad.shape == input_tensor.shape
    assert torch.isfinite(input_tensor.grad).all()


def test_pcm_batch_autograd_op_forward_and_backward():
    operator = make_operator()

    torch.manual_seed(0)
    batch = torch.randn(3, *operator.domain_shape, requires_grad=True)
    output = operator(batch)

    assert output.shape == (3, *operator.range_shape)
    output.square().mean().backward()

    assert batch.grad is not None
    assert batch.grad.shape == batch.shape
    assert torch.isfinite(batch.grad).all()


def test_pcm_op_adjointness():
    operator = make_operator()

    assert operator.domain_shape == (16, 16)
    assert operator.range_shape == (64,)

    test_image = torch.rand(*operator.domain_shape)
    test_measurement = torch.rand(*operator.range_shape)
    dotproduct_adjointness_test(operator, test_image, test_measurement)

    assert operator.pseudo_inv(test_measurement) is not None


def test_pcm_batch_forward_matches_stacked_single_forward():
    operator = make_operator()
    torch.manual_seed(1)

    batch = torch.randn(4, *operator.domain_shape)
    actual = operator(batch)
    expected = torch.stack([operator(batch[i]) for i in range(batch.shape[0])])

    assert actual.shape == (4, *operator.range_shape)
    torch.testing.assert_close(actual, expected)


def test_pcm_batch_adjoint_matches_stacked_single_adjoint():
    operator = make_operator()
    torch.manual_seed(2)

    measurements = torch.randn(4, *operator.range_shape)
    actual = operator.adjoint(measurements)
    expected = torch.stack(
        [operator.adjoint(measurements[i]) for i in range(measurements.shape[0])]
    )

    assert actual.shape == (4, *operator.domain_shape)
    torch.testing.assert_close(actual, expected)


def test_pcm_batch_pseudoinverse_matches_stacked_single_pseudoinverse():
    operator = make_operator()
    torch.manual_seed(3)

    measurements = torch.randn(4, *operator.range_shape)
    actual = operator.pseudo_inv(measurements)
    expected = torch.stack(
        [operator.pseudo_inv(measurements[i]) for i in range(measurements.shape[0])]
    )

    assert actual.shape == (4, *operator.domain_shape)
    torch.testing.assert_close(actual, expected)


def test_pcm_batch_adjointness():
    operator = make_operator()
    torch.manual_seed(4)

    x = torch.randn(3, *operator.domain_shape, dtype=torch.float64)
    y = torch.randn(3, *operator.range_shape, dtype=torch.float64)

    lhs = torch.sum(operator(x) * y)
    rhs = torch.sum(x * operator.adjoint(y))

    torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)
