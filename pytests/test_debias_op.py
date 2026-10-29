"""The debiasing least squares step used by the photocurrent mapping example.

``debias_ls`` is called by ``scripts/example_scripts/photocurrent_mapping.py``.
These tests use a diagonal operator so that the least squares solution is known
in closed form and nothing here needs a geometry, a dataset or a GPU.
"""

import torch

from LION.operators.DebiasOp import debias_ls
from LION.operators.Operator import Operator

N = 12
SUPPORT = torch.tensor([2, 5, 6, 9])


class Diagonal(Operator):
    """A diagonal operator, i.e. an invertible system with a known solution."""

    def __init__(self, diag: torch.Tensor):
        super().__init__()
        self.diag = diag

    @property
    def domain_shape(self) -> tuple[int, ...]:
        return (self.diag.numel(),)

    @property
    def range_shape(self) -> tuple[int, ...]:
        return (self.diag.numel(),)

    def __call__(self, v: torch.Tensor, out=None) -> torch.Tensor:
        return self.forward(v)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.diag * v

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        return self.diag * r


def problem() -> tuple[Diagonal, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measurements of a signal that is only supported on ``SUPPORT``."""
    diag = torch.linspace(1.0, 2.0, N)
    x_true = torch.linspace(-1.0, 1.0, N)
    w = torch.zeros(N)
    w[SUPPORT] = 1.0
    signal = torch.zeros(N)
    signal[SUPPORT] = x_true[SUPPORT]
    return Diagonal(diag), diag * signal, w, x_true


def test_debias_ls_solves_the_system_on_the_support():
    op, y, w, x_true = problem()
    w_deb = debias_ls(op, y, w, max_iter=500, tol=0.0)
    torch.testing.assert_close(w_deb[SUPPORT], x_true[SUPPORT], rtol=1e-4, atol=1e-4)
    outside = torch.ones(N, dtype=torch.bool)
    outside[SUPPORT] = False
    assert torch.count_nonzero(w_deb[outside]) == 0


def test_debias_ls_without_support_returns_the_input_unchanged():
    op, y, _, _ = problem()
    w = torch.zeros(N)
    assert torch.equal(debias_ls(op, y, w), w)
