"""Conjugate gradient solver"""

from typing import Callable

import torch


def conjugate_gradient(
    matmul_closure: Callable[[torch.Tensor], torch.Tensor],
    d: torch.Tensor,
    x0: torch.Tensor,
    max_iter: int,
    eps: float = 1e-14,
    rel_tol: float = 0.0,
    prog_bar: Callable | None = None,
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

    iterator = (
        prog_bar(range(max_iter), desc="CG iterations") if prog_bar else range(max_iter)
    )
    for _ in iterator:
        z = matmul_closure(d)

        dz = torch.sum(d * z)
        # Check for breakdown
        if abs(dz) < eps:
            break
        alpha = rr / dz
        x += alpha * d
        r -= alpha * z

        if torch.norm(r) / torch.norm(d) < rel_tol:
            break

        rr_next = torch.sum(r**2)
        beta = rr_next / rr
        d = r + beta * d
        rr = rr_next

    return x
