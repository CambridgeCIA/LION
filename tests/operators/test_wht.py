"""Test wht behaviour."""

import torch
from LION.operators.Operator import Operator
from spyrit.core.torch import fwht, ifwht
from tests.helper import dotproduct_adjointness_test


def test_wht_adjointness():
    """Test WHT operator adjoint property."""
    n = 256
    x = torch.rand(n)
    y = torch.rand(n)

    class WhtOp(Operator):
        """Provide the wht op test double used by this module."""

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the callable operation."""
            return fwht(x, dim=0)

        def adjoint(self, y: torch.Tensor) -> torch.Tensor:
            """Handle adjoint for the PaDIS workflow."""
            return fwht(y, dim=0)

    operator = WhtOp()
    dotproduct_adjointness_test(operator, x, y)

    torch.testing.assert_close(ifwht(operator(y)), y)
