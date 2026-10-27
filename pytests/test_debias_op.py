"""The debiasing least squares step used by the photocurrent mapping example.

``debias_ls`` is called by ``scripts/example_scripts/photocurrent_mapping.py``.
These tests use a diagonal operator so that the least squares solution is known
in closed form and nothing here needs a geometry, a dataset or a GPU.
"""

import torch

from LION.operators.DebiasOp import DebiasOp, debias_ls
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


def test_debias_op_can_be_constructed():
    """``op`` is an nn.Module, so it can only be stored once Module is set up."""
    op, y, w, _ = problem()
    debias_op = DebiasOp(op, y, w)
    assert debias_op.domain_shape == (SUPPORT.numel(),)
    assert debias_op.range_shape == (N,)


def test_debias_ls_solves_the_system_on_the_support():
    op, y, w, x_true = problem()
    w_deb = debias_ls(op, y, w, max_iter=500, tol=0.0)
    torch.testing.assert_close(w_deb[SUPPORT], x_true[SUPPORT], rtol=1e-4, atol=1e-4)
    outside = torch.ones(N, dtype=torch.bool)
    outside[SUPPORT] = False
    assert torch.count_nonzero(w_deb[outside]) == 0


def test_debias_ls_reports_progress_when_asked_to():
    """``verbose`` is a parameter, not the ``tabnanny`` module flag of the same name."""
    op, y, w, _ = problem()
    quiet = debias_ls(op, y, w, max_iter=20, tol=0.0)
    loud = debias_ls(op, y, w, max_iter=20, tol=0.0, verbose=True, progress_bar=True)
    torch.testing.assert_close(loud, quiet)


def test_debias_ls_prints_the_objective_when_verbose(capsys):
    op, y, w, _ = problem()
    debias_ls(op, y, w, max_iter=3, tol=0.0, verbose=True)
    printed = capsys.readouterr().out
    assert printed.count("Debiasing LS  f =") == 3
    assert "rel_change < tol:" in printed


def test_debias_ls_without_support_returns_the_input_unchanged():
    op, y, _, _ = problem()
    w = torch.zeros(N)
    assert torch.equal(debias_ls(op, y, w), w)
