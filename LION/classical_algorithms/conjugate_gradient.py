# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Thanh Trung (Troy) Vu, Ander Biguri
# =============================================================================
"""Conjugate gradient solver"""

from typing import Callable

import torch


def conjugate_gradient(
    matmul_closure: Callable[[torch.Tensor], torch.Tensor],
    d: torch.Tensor,
    x0: torch.Tensor,
    max_iter: int,
    tol: float,
) -> torch.Tensor:
    """Solves A x = ``d`` by the conjugate gradient method, A symmetric
    positive definite, given as ``matmul_closure(x)`` = A x. Starts from
    ``x0``; stops after ``max_iter`` iterations or when the relative residual
    ||d - A x|| / ||d|| is below ``tol``. Returns x."""
    b_norm = torch.norm(d)
    x = x0.clone()
    if b_norm == 0:
        return torch.zeros_like(x)

    r = d - matmul_closure(x)
    p = r.clone()  # search direction
    rr = torch.sum(r**2)
    if torch.sqrt(rr) / b_norm < tol:
        return x

    for _ in range(max_iter):
        z = matmul_closure(p)

        pz = torch.sum(p * z)
        # Breakdown: p^T A p must be positive for a symmetric positive definite
        # A. (An absolute threshold here would stop early on small residuals.)
        if pz <= 0:
            break
        alpha = rr / pz
        x += alpha * p
        r -= alpha * z

        # Stop on the relative residual ||b - Ax|| / ||b||
        if torch.norm(r) / b_norm < tol:
            break

        rr_next = torch.sum(r**2)
        beta = rr_next / rr
        p = r + beta * p
        rr = rr_next

    return x
