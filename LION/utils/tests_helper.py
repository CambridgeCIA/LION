"""Helper/Utilities for test functions."""

import torch
from LION.operators import Operator


def dotproduct_adjointness_test(
    operator: Operator,
    u: torch.Tensor,
    v: torch.Tensor,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-5,
) -> None:
    """Test the adjointness of linear operator and operator.H.

    Test if
         <Operator(u),v> == <u, Operator^H(v)>
         for one u ∈ domain and one v ∈ range of Operator.
    and if the shapes match.

    Note: This property should hold for all u and v.
    Commonly, this function is called with two random vectors u and v.

    Parameters
    ----------
    operator
        linear operator
    u
        element of the domain of the operator
    v
        element of the range of the operator
    relative_tolerance
        default is pytorch's default for float16
    absolute_tolerance
        default is pytorch's default for float16

    Raises
    ------
    `AssertionError`
        if the adjointness property does not hold
    `AssertionError`
        if the shape of operator(u) and v does not match
        if the shape of u and operator.H(v) does not match

    """
    forward_u = operator(u)
    adjoint_v = operator.adjoint(v)
    if not isinstance(forward_u, torch.Tensor):
        forward_u = torch.as_tensor(forward_u)
    if not isinstance(adjoint_v, torch.Tensor):
        adjoint_v = torch.as_tensor(adjoint_v)

    # explicitly check the shapes, as flatten makes the dot product insensitive to wrong shapes
    assert forward_u.shape == v.shape
    assert adjoint_v.shape == u.shape

    dotproduct_range = torch.vdot(forward_u.flatten(), v.flatten())
    dotproduct_domain = torch.vdot(u.flatten().flatten(), adjoint_v.flatten())
    torch.testing.assert_close(
        dotproduct_range,
        dotproduct_domain,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
