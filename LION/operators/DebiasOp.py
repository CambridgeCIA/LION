# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Thanh Trung (Troy) Vu
# =============================================================================
"""Debiasing least squares operator on the support of w."""

from __future__ import annotations

import torch
from tqdm import tqdm

from LION.operators.Operator import Operator
from LION.utils.math import power_method


class DebiasOp(Operator):
    """The forward operator ``operator`` restricted to the support of ``w`` (an
    l1 minimiser; its entries above ``support_tol``), for debiasing it by least
    squares on that support (Koutsourakis, Thompson and Blakesley, "Toward
    Megapixel Resolution Compressed Sensing Current Mapping of Photovoltaic
    Devices Using Digital Light Processing", Solar RRL 5(11):2100467, 2021,
    doi:10.1002/solr.202100467.) ``y``: the measurements (M,)."""

    def __init__(
        self,
        op: Operator,
        y: torch.Tensor,
        w: torch.Tensor,
        support_tol: float = 1e-3,
    ):
        # nn.Module refuses to register a submodule before Module.__init__ has
        # run, and ``op`` is one, so without this line every construction fails
        # with "cannot assign module before Module.__init__() call".
        super().__init__()
        self.op = op
        self.y = y
        self.w = w
        self.support_tol = support_tol

        self.support = torch.nonzero(
            torch.abs(w) > support_tol, as_tuple=False
        ).squeeze(1)
        self.device = op.device

    def __call__(self, v: torch.Tensor, out=None) -> torch.Tensor:
        """The measurements (M,) of ``v``, a vector on the support (Ns,).
        ``out``: unused, kept for the ``__call__(x, out)`` signature."""
        return self.forward(v)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """The measurements (M,) of ``v``, a vector on the support (Ns,). Call
        the operator, ``op(v)``, rather than this method (as for torch
        modules)."""
        w_full = torch.zeros_like(self.w)
        w_full[self.support] = v
        return self.op.forward(w_full)

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        """The adjoint: the vector on the support (Ns,) of the measurements
        ``r`` (M,)."""
        g_full = self.op.adjoint(r)
        return g_full[self.support]

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Return the shape of the support domain."""
        return (self.support.numel(),)

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of the measurement range."""
        return self.op.range_shape


def debias_ls(
    op: Operator,
    y: torch.Tensor,
    w: torch.Tensor,
    support_tol: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-5,
    verbose: int = 1,
) -> torch.Tensor:
    """``w`` (an l1 minimiser, (Nw,)) debiased: least squares fitted to the
    measurements ``y`` (M,) on its support (the entries above ``support_tol``),
    by gradient descent. Returns the debiased coefficients (Nw,).

    ``op``: the forward operator; ``max_iter``, ``tol``: the iterations and the
    relative stopping threshold; ``verbose``: 0 nothing, 1 (default) a progress
    bar, 2 with the objective and the relative change."""
    device = w.device

    support = torch.nonzero(torch.abs(w) > support_tol, as_tuple=False).squeeze(1)
    if support.numel() == 0:
        return w.clone()

    op_s = DebiasOp(op, y, w, support_tol=support_tol)

    # Check: Why is the squaring needed here?
    L = power_method(op_s, device=device) ** 2
    step = 1.0 / (L + 1e-12)

    v = w[support].clone()

    iterator = tqdm(range(max_iter), desc="Debiasing LS", disable=verbose < 1)
    for _ in iterator:
        r = op_s(v) - y
        grad = op_s.adjoint(r)
        v_next = v - step * grad

        rel_change = torch.norm(v_next - v) / (torch.norm(v) + 1e-8)
        v = v_next

        if verbose >= 2:
            data_term = 0.5 * torch.norm(op_s(v) - y).pow(2).item()
            iterator.set_postfix(
                f=f"{data_term:.4e}", rel_change=f"{rel_change.item():.2e}"
            )

        if rel_change.item() < tol:
            break

    w_deb = torch.zeros_like(w)
    w_deb[support] = v
    return w_deb
