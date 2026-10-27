"""Conjugate gradient solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from LION.utils.parameter import LIONParameter


@dataclass(kw_only=True)
class CGProxParameter(LIONParameter):
    """Configuration for conjugate gradient solver.

    Attributes:
    ----------
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
    eps: float = 0.0
    rel_tol: float = 0.0
    data_weight: float = 1.0
    normalize: bool = True


def conjugate_gradient(
    matmul_closure: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    x0: torch.Tensor,
    max_iter: int,
    eps: float = 0.0,  # TODO: Should we use a very small value like 1e-30 instead?
    rel_tol: float = 1e-8,
    dim: tuple[int, ...] | None = (-2, -1),
    prog_bar: Callable | None = None,
) -> torch.Tensor:
    """Conjugate gradient solver.

    Args:
        matmul_closure: A function that performs the matrix-vector multiplication.
        rhs: The right-hand side vector.
        x0: The initial guess for the solution.
        max_iter: Maximum number of iterations.
        eps: Numerical stability threshold to prevent division by zero.
        rel_tol: Relative tolerance on the relative residual ``||d - A x|| / ||d||`` for convergence.
            The algorithm stops when the residual norm is at most rel_tol times the norm of the right-hand side.
        dim: Contiguous trailing dimensions. e.g., (-2, -1) for the last two dimensions.
            Cannot be non-contiguous like (-2,) or (-3, -1). If None, use all dimensions.
            By default use last two dims, often height and width of 2D image.
        prog_bar: Optional callable that wraps an iterable to provide a progress bar (e.g., tqdm).

    Returns:
        The approximate solution vector.
    """
    if dim is None:
        dim = tuple(range(rhs.ndim))  # Use all dimensions
    # Broadcast alpha and beta to the shape of x, r, and p
    broadcast_index = (...,) + (None,) * len(dim)

    norm_rhs = torch.linalg.vector_norm(rhs, dim=dim)
    if torch.all(norm_rhs == 0):  # Should we choose a small epsilon instead of zero?
        return torch.zeros_like(x0)

    x = x0.clone()

    # NOTE: Assuming matmul_closure works with dim in the same way
    r = rhs - matmul_closure(x)
    p = r.clone()  # search direction

    r_squared = torch.sum(r**2, dim=dim)

    # Should we square this so we don't have to take sqrt of norm(r) later? Numerically unstable if threshold is small?
    threshold = rel_tol * norm_rhs

    iterator = prog_bar(range(max_iter)) if prog_bar else range(max_iter)
    for _ in iterator:
        if torch.all(torch.linalg.vector_norm(r, dim=dim) <= threshold):
            break

        Ap = matmul_closure(p)

        pAp = torch.sum(p * Ap, dim=dim)

        # Breakdown: p^T A p must be positive for a symmetric positive definite
        # A. (An absolute threshold here may stop early on small residuals.)
        active = torch.abs(pAp) >= eps
        if not torch.any(active):  # If nothing is active, stop
            break

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
