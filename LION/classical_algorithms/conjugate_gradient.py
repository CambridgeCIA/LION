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
    """
    Conjugate gradient solver.

    Parameters
    ----------
    matmul_closure : Callable[[torch.Tensor], torch.Tensor]
        A function that performs the matrix-vector multiplication.
    d : torch.Tensor
        The right-hand side vector.
    x0 : torch.Tensor
        The initial guess for the solution.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance on the relative residual ``||d - A x|| / ||d||``.

    Returns
    -------
    torch.Tensor
        The approximate solution vector.

    """
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
