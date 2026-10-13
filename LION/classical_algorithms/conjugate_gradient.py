"""Conjugate gradient solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from LION.utils.parameter import LIONParameter


@dataclass(kw_only=True)
class CGProxParameter(LIONParameter):
    """Configuration for conjugate gradient solver.

    Attributes:
        iters: The number of conjugate gradient iterations.
        eta: The ADMM penalty parameter.
        eps: The absolute tolerance for the conjugate gradient solver.
        rel_tol: The relative tolerance for the conjugate gradient solver.
        data_weight: The weight for the data fidelity term.
        normalize: Whether to normalize by the number of measurements.
            If True, assume no batch dimension and divide by total number of elements.
    """

    iters: int = 20
    eta: float = 0.01
    eps: float = 1e-20
    rel_tol: float = 0.0
    data_weight: float = 1.0
    normalize: bool = True


def conjugate_gradient(
    matmul_closure: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    x0: torch.Tensor,
    max_iter: int,
    eps: float = 1e-14,
    rel_tol: float = 1e-8,
    dim: tuple[int, ...]
    | None = (-2, -1),  # By default use last two dims, often height and width of image
    prog_bar: Callable | None = None,
) -> torch.Tensor:
    """Conjugate gradient solver.

    Args:
        matmul_closure: A function that performs the matrix-vector multiplication.
        rhs: The right-hand side vector.
        x0: The initial guess for the solution.
        max_iter: Maximum number of iterations.
        eps: Numerical stability threshold to prevent division by zero.
        rel_tol: Relative tolerance for convergence.
            The algorithm stops when the residual norm is at most rel_tol times the norm of the right-hand side.
        dim: Contiguous trailing dimensions. e.g., (-2, -1) for the last two dimensions.
            Cannot be non-contiguous like (-2,) or (-3, -1). If None, use all dimensions.
        prog_bar: Optional callable that wraps an iterable to provide a progress bar (e.g., tqdm).

    Returns:
        The approximate solution vector.
    """
    if dim is None:
        dim = tuple(range(rhs.ndim))  # Use all dimensions
    broadcast_index = (...,) + (None,) * len(
        dim
    )  # Broadcast alpha and beta to the shape of x, r, and p

    x = x0.clone()

    r = rhs - matmul_closure(x)
    p = r.clone()

    r_squared = torch.sum(r**2, dim=dim)

    threshold = rel_tol * torch.linalg.vector_norm(rhs, dim=dim)

    iterator = prog_bar(range(max_iter)) if prog_bar else range(max_iter)
    for _ in iterator:
        if torch.all(torch.linalg.vector_norm(r, dim=dim) <= threshold):
            break

        Ap = matmul_closure(p)

        pAp = torch.sum(p * Ap, dim=dim)

        # Check for breakdown
        active = torch.abs(pAp) >= eps
        if not torch.any(active):  # If nothing is active, stop
            break
        # Else continue, ignoring the inactive ones

        alpha = torch.zeros_like(r_squared)
        alpha[active] = r_squared[active] / pAp[active]
        x += alpha[broadcast_index] * p
        r -= alpha[broadcast_index] * Ap

        r_squared_next = torch.sum(r**2, dim=dim)
        beta = torch.zeros_like(r_squared)
        beta[active] = r_squared_next[active] / r_squared[active]

        p = r + beta[broadcast_index] * p
        r_squared = r_squared_next
    return x
