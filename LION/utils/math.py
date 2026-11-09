# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri, Thanh Trung (Troy) Vu
# =============================================================================
"""Utility math functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # LION.operators imports this module (DebiasOp)
    from LION.operators.Operator import Operator


def power_method(
    op: Operator,
    maxiter: int = 100,
    tol: float = 1e-6,
    device: torch.device | str | None = None,
    seed: int | None = 0,
) -> torch.Tensor:
    """The norm of the operator ``op`` (its largest singular value), estimated
    by power iterations on AᵀA, as a tensor.

    ``maxiter``: the most iterations; ``tol``: stop once ||A v - sigma u|| <=
    tol * sigma (relative, so independent of the operator's scale); ``device``:
    where to compute (the operator's if None); ``seed``: of the random starting
    vector, so that the estimate is the same on every call (e.g. step sizes
    computed from it; None: torch's global random state)."""
    if device is None:
        device = getattr(op, "device", None)
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    # drawn on the CPU, so that the same seed gives the same vector on any device
    arr_old = torch.randn(*op.domain_shape, generator=generator).to(device)
    error = tol + 1
    i = 0
    while error >= tol:

        # very verbose and inefficient for now
        omega = op(arr_old)
        alpha = torch.linalg.norm(omega)
        u = (1.0 / alpha) * omega
        z = op.adjoint(u)
        beta = torch.linalg.norm(z)
        arr = (1.0 / beta) * z
        error = torch.linalg.norm(op(arr) - beta * u) / beta
        sigma = beta
        arr_old = arr
        i += 1
        if i >= maxiter:
            return sigma

    return sigma
