"""FISTA algorithm for l1-regularized problems."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from LION.operators.Operator import Operator
from LION.utils.math import power_method


def soft_threshold(v: torch.Tensor, tau: float) -> torch.Tensor:
    r"""Soft thresholding operator.

    It is defined as:

    .. math::

        S_{\tau}(v) = \mathrm{sign}(v) \cdot \\max(|v| - \tau, 0)

    Args:
        v: Input tensor.
        tau: Threshold parameter.

    Returns:
        Result after applying soft thresholding.
    """

    return torch.sign(v) * torch.clamp(torch.abs(v) - tau, min=0.0)


def fista_l1(
    op: Operator,
    y: torch.Tensor,
    lam: float,
    max_iter: int = 200,
    tol: float = 1e-4,
    L: float | None = None,
    verbose: bool = False,
    prog_bar: Callable | None = None,
) -> torch.Tensor:
    r"""Solve :math:`\min_w \tfrac12\lVert A w - y\rVert_2^2 + \lambda \lVert w\rVert_1`
    by FISTA.

    Implements the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for
    :math:`\ell_1`-regularised least squares [BeckTeboulle2009]_. FISTA is an accelerated proximal-gradient method for
    composite objectives :math:`f(w) + \lambda \lVert w\rVert_1` with smooth data-fidelity term
    :math:`f(w) = \tfrac12\lVert A w - y\rVert_2^2`; see [DaubechiesDefriseDeMol2004]_ for the original ISTA scheme and
    [ParikhBoyd2014]_ for a general overview of proximal-gradient methods.

    Args:
        op: Linear operator implementing the forward and adjoint. It is called as ``op(w)`` and ``op.adjoint(r)``.
        y: Measurements, shape ``(M,)``.
        lam: :math:`\ell_1` regularisation parameter.
        max_iter: Maximum number of iterations.
        tol: Relative stopping threshold on :math:`w`. The iteration stops once
            ``norm(w_next - w) / (norm(w) + 1e-8) < tol``.
        L: Lipschitz constant of :math:`A^\top A`. If ``None``, estimated by a power method on the normal operator
        :math:`A^\top A`, following the standard practice in FISTA-type schemes [BeckTeboulle2009]_.
        verbose: If True, prints basic progress such as objective value and relative change.
        prog_bar: If True, wraps the iteration in a ``tqdm`` progress bar.

    Returns:
        w: Estimated coefficient vector, with the shape of ``op.adjoint(y)``.

    References:
    ----------
    .. [DaubechiesDefriseDeMol2004] I. Daubechies, M. Defrise, and C. De Mol,
       "An iterative thresholding algorithm for linear inverse problems with a
       sparsity constraint", Communications on Pure and Applied Mathematics,
       57(11):1413-1457, 2004.
    .. [BeckTeboulle2009] A. Beck and M. Teboulle, "A fast iterative
       shrinkage-thresholding algorithm for linear inverse problems", SIAM
       Journal on Imaging Sciences, 2(1):183-202, 2009.
    .. [ParikhBoyd2014] N. Parikh and S. Boyd, "Proximal Algorithms",
       Foundations and Trends in Optimization, 1(3):127-239, 2014.
    """

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

    iterator = (
        prog_bar(range(max_iter), desc="FISTA l1") if prog_bar else range(max_iter)
    )
    for k in iterator:
        Az: torch.Tensor = op(z)
        grad = op.adjoint(Az - y)  # gradient of data term, same shape as w

        w_next = soft_threshold(z - step * grad, lam * step)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = w_next + (t - 1.0) / t_next * (w_next - w)

        rel_change = torch.norm(w_next - w) / (torch.norm(w) + 1e-8)
        w = w_next
        t = t_next

        if verbose:
            data_term = 0.5 * torch.norm((op(w)) - y).pow(2).item()
            l1_term = lam * torch.norm(w, p=1).item()
            print(
                f"Iter {k:4d}  f = {data_term + l1_term:.4e}  "
                f"rel_change = {rel_change.item():.2e}  tol = {tol:.2e}  "
                f"rel_change < tol: {rel_change.item() < tol}"
            )

        if rel_change.item() < tol:
            break

    return w
