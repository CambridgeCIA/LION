# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Thanh Trung (Troy) Vu
# =============================================================================
"""FISTA algorithm for l1-regularized problems."""

import math

import torch
from tqdm import tqdm

from LION.operators import Operator
from LION.utils.math import power_method


def soft_threshold(v: torch.Tensor, tau: float) -> torch.Tensor:
    """The soft thresholding of ``v`` by ``tau``: sign(v) max(|v| - tau, 0),
    the proximal operator of tau ||.||_1."""
    return torch.sign(v) * torch.clamp(torch.abs(v) - tau, min=0.0)


def fista_l1(
    op: Operator,
    y: torch.Tensor,
    lam: float,
    max_iter: int = 200,
    tol: float = 1e-4,
    L: float | None = None,
    verbose: int = 1,
) -> torch.Tensor:
    """Solves min_w ½||A w - y||² + lam ||w||_1 by FISTA (Beck and Teboulle,
    2009), an accelerated proximal-gradient method; see Daubechies, Defrise and
    De Mol (2004) for ISTA and Parikh and Boyd (2014) for proximal algorithms.
    Returns w, with the shape of ``op.adjoint(y)``.

    ``op``: the linear operator, called as ``op(w)`` and ``op.adjoint(r)``;
    ``y``: the measurements (M,); ``lam``: the l1 weight; ``max_iter``;
    ``tol``: stop when ||w_next - w|| / (||w|| + 1e-8) < tol; ``L``: the
    Lipschitz constant of AᵀA (None: estimated by the power method);
    ``verbose``: 0 nothing, 1 (default) a progress bar, 2 with the objective
    and the relative change (one more ``op(w)`` per iteration).

    References: Beck and Teboulle, "A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems", SIAM J. Imaging Sciences
    2(1):183-202, 2009. Daubechies, Defrise and De Mol, "An iterative
    thresholding algorithm for linear inverse problems with a sparsity
    constraint", Comm. Pure Appl. Math. 57(11):1413-1457, 2004. Parikh and
    Boyd, "Proximal Algorithms", Foundations and Trends in Optimization
    1(3):127-239, 2014."""
    y = y.detach()
    device = y.device

    # Shape of the unknown inferred from one adjoint call
    w0: torch.Tensor = op.adjoint(torch.zeros_like(y))

    if L is None:
        # Power method estimates ||A||_2; Lipschitz constant is ||A||_2^2
        L = power_method(op, device=device).item() ** 2
    step = 1.0 / (L + 1e-12)

    w = torch.zeros_like(w0, device=device)
    z = w.clone()
    t = 1.0

    iterator = tqdm(range(max_iter), desc="FISTA l1", disable=verbose < 1)
    for _ in iterator:
        Az: torch.Tensor = op(z)
        grad = op.adjoint(Az - y)  # gradient of data term, same shape as w

        w_next = soft_threshold(z - step * grad, lam * step)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = w_next + (t - 1.0) / t_next * (w_next - w)

        rel_change = torch.norm(w_next - w) / (torch.norm(w) + 1e-8)
        w = w_next
        t = t_next

        if verbose >= 2:
            data_term = 0.5 * torch.norm((op(w)) - y).pow(2).item()
            l1_term = lam * torch.norm(w, p=1).item()
            iterator.set_postfix(
                f=f"{data_term + l1_term:.4e}", rel_change=f"{rel_change.item():.2e}"
            )

        if rel_change.item() < tol:
            break

    return w
