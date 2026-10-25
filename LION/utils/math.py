"""Utility math functions."""

from __future__ import annotations

import numpy as np
import torch

from LION.operators.Operator import Operator


def power_method(
    op: Operator,
    maxiter: int = 100,
    tol: float = 1e-6,
    device: torch.device | str | None = None,
    seed: int | None = 0,
) -> torch.Tensor:
    """Estimate operator norm by power iteration using torch.

    Parameters
    ----------
    op : Operator
        The operator for which to estimate the norm.
    maxiter : int
        Number of power iterations.
    tol : float
        Relative tolerance: stops once ||A v - sigma u|| <= tol * sigma, i.e.
        independently of the scale of the operator.
    device : torch.device | str | None
        Device for computations. If None, use operator's device.
    seed : int | None
        Seed of the random starting vector, so the estimate is the same on
        every call (e.g. step sizes computed from it). None uses torch's
        global random state.

    Returns
    -------
    sigma : torch.Tensor
        Estimated operator norm.
    """
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


def test_convexity(net, x, device):
    # check convexity of the net numerically
    print("running a numerical convexity test...")
    n_trials = 100
    convexity = 0
    for trial in np.arange(n_trials):
        x1 = torch.rand(x.size()).to(device)
        x2 = torch.rand(x.size()).to(device)
        alpha = torch.rand(1).to(device)

        cvx_combo_of_input = net(alpha * x1 + (1 - alpha) * x2)
        cvx_combo_of_output = alpha * net(x1) + (1 - alpha) * net(x2)

        convexity += cvx_combo_of_input.mean() <= cvx_combo_of_output.mean()
    if convexity == n_trials:
        flag = True
        print("Passed convexity test!")
    else:
        flag = False
        print("Failed convexity test!")
    return flag
