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
        Tolerance for convergence.

    Returns
    -------
    torch.Tensor
        The approximate solution vector.

    """
    x = x0.clone()
    r = d - matmul_closure(x)
    d = r.clone()
    rr = torch.sum(r**2)

    for _ in range(max_iter):
        z = matmul_closure(d)

        dz = torch.sum(d * z)
        # Check for breakdown
        if abs(dz) < 1e-14:
            break
        alpha = rr / dz
        x += alpha * d
        r -= alpha * z

        if torch.norm(r) / torch.norm(d) < tol:
            break

        rr_next = torch.sum(r**2)
        beta = rr_next / rr
        d = r + beta * d
        rr = rr_next

    return x
