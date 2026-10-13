"""Tests for CT operator."""

import pytest
import torch
from pytests.helper import dotproduct_adjointness_test


@pytest.mark.tomosipo  # Add argument `-m "not tomosipo"` to the `pytest` command line to skip this test
def test_ct_autograd_op_forward_and_backward():
    """Check that to_autograd wraps the CT operator correctly and supports backprop."""
    from LION.CTtools.ct_geometry import Geometry
    from LION.CTtools.ct_utils import make_operator
    from tomosipo.torch_support import to_autograd

    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)
    autograd_operator = to_autograd(operator)

    torch.manual_seed(0)
    input_tensor = torch.randn(*geometry.image_shape, requires_grad=True)

    output_tensor = autograd_operator(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.requires_grad
    assert input_tensor.grad is None

    output_tensor.mean().backward()

    assert input_tensor.grad is not None
    assert isinstance(input_tensor.grad, torch.Tensor)
    assert input_tensor.grad.shape == input_tensor.shape
    assert torch.isfinite(input_tensor.grad).all()


@pytest.mark.tomosipo  # Add argument `-m "not tomosipo"` to the `pytest` command line to skip this test
def test_ct_autograd_op_matches_original_operator():
    """Check that the autograd wrapper produces the same output as the original operator."""
    import numpy as np
    from LION.CTtools.ct_geometry import Geometry
    from LION.CTtools.ct_utils import make_operator
    from tomosipo.torch_support import to_autograd

    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)
    autograd_operator = to_autograd(operator)

    torch.manual_seed(1)
    input_tensor = torch.randn(*geometry.image_shape, requires_grad=True)

    input_np = input_tensor.detach().cpu().numpy()
    output_np = operator(input_np)
    output_autograd = autograd_operator(input_tensor).detach().cpu().numpy()

    np.testing.assert_allclose(output_autograd, output_np, rtol=1e-5, atol=1e-5)


@pytest.mark.tomosipo  # Add argument `-m "not tomosipo"` to the `pytest` command line to skip this test
def test_original_op_does_not_propagate_grad():
    """Check that the original operator output does not require gradients and backward fails."""
    from LION.CTtools.ct_geometry import Geometry
    from LION.CTtools.ct_utils import make_operator

    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)

    torch.manual_seed(2)
    input_tensor = torch.randn(*geometry.image_shape, requires_grad=True)

    # Original operator is not autograd aware
    output_tensor = operator(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.requires_grad is False

    # Backprop should fail with the expected autograd error
    with pytest.raises(
        RuntimeError, match="does not require grad and does not have a grad_fn"
    ):
        output_tensor.mean().backward()


@pytest.mark.tomosipo  # Add argument `-m "not tomosipo"` to the `pytest` command line to skip this test
def test_ct_op_adjointness():
    """Test CT operator adjoint property."""
    from LION.CTtools.ct_geometry import Geometry
    from LION.CTtools.ct_utils import make_operator

    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)

    # Check the default operator shapes
    assert operator.domain_shape == (1, 512, 512)
    assert operator.range_shape == (1, 360, 900)

    # Create a test input for the forward and backward projections
    test_volume = torch.rand(*operator.domain_shape)
    test_projection = torch.rand(*operator.range_shape)
    # Note: tomosipo implementation uses an accelerated method that is not
    # exactly adjoint, so we use looser tolerances here.
    dotproduct_adjointness_test(
        operator,
        test_volume,
        test_projection,
        relative_tolerance=1e-1,
        absolute_tolerance=1e-1,
    )


@pytest.mark.tomosipo  # Add argument `-m "not tomosipo"` to the `pytest` command line to skip this test
def test_ct_op_backward_compatibility_with_tomosipo():
    from LION.CTtools.ct_geometry import Geometry
    from LION.CTtools.ct_utils import make_operator

    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)

    # Check the default operator shapes
    assert operator.domain_shape == (1, 512, 512)
    assert operator.range_shape == (1, 360, 900)

    # Create a test input for the forward and backward projections
    test_volume = torch.rand(*operator.domain_shape)
    test_projection = torch.rand(*operator.range_shape)

    # Make sure that tomosipo Operator's attributes are accessible
    assert operator._fp(volume=test_volume, out=None) is not None
    assert operator._bp(projection=test_projection, out=None) is not None
    assert operator.transpose() is not None
    assert operator.T is not None
    assert operator.domain is not None
    assert operator.range is not None
