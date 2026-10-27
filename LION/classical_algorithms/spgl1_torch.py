"""SPGL1 sparse reconstruction with torch operators."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator
from spgl1 import spg_bp

from LION.operators.Operator import Operator
from LION.utils.parameter import LIONParameter


def spgl1_torch(
    op: Operator, y: torch.Tensor, **spgl1_kwargs
) -> tuple[torch.Tensor, Any]:
    r"""Use SPGL1 with Basis Pursuit (BP) to solve an l1 sparse reconstruction.

    This is a thin wrapper around the ``spgl1.spg_bp`` that uses torch operators.
    SPGL1 is a spectral projected-gradient method for constrained l1 problems;
    see [BergFriedlander2008]_ and [BergFriedlander2010]_.

    This wrapper uses the same calling convention (argument names and behaviour) as ``spgl1.spg_bp``;
    see [SPGL1Python]_ for details.

    Args:
        op: Linear operator implementing the forward map ``op(w)`` and its adjoint ``op.adjoint(r)``.
        y: Measurements, shape ``(M,)``.
        spgl1_kwargs: Extra keyword args forwarded to ``spgl1.spg_bp``
            (e.g., tolerances, iteration limits; see [SPGL1Python]_).

    Returns:
        Estimated coefficient vector in the same shape as ``op.adjoint(y*0)`` and the SPGL1 info dictionary.

    References:
    ----------
    .. [BergFriedlander2008] E. van den Berg and M. P. Friedlander,
        "Probing the Pareto frontier for basis pursuit solutions",
        SIAM Journal on Scientific Computing, 31(2):890-912, 2008.
    .. [BergFriedlander2010] E. van den Berg and M. P. Friedlander,
        "Sparse optimisation with least-squares constraints",
        TR-2010-02, Department of Computer Science, University of British Columbia, 2010.
    .. [SPGL1Python] SPGL1: Spectral Projected Gradient for L1 minimisation,
        Python package documentation, https://spgl1.readthedocs.io/
    """
    device = y.device
    y = y.detach()

    # Infer coefficient shape from one adjoint call
    with torch.no_grad():
        w0 = op.adjoint(torch.zeros_like(y))
    w0 = w0.detach()
    n_w = w0.numel()
    n_y = y.numel()

    def matvec(w_np: np.ndarray) -> np.ndarray:
        w_t = torch.from_numpy(w_np.astype(np.float32)).to(device).view_as(w0)
        y_t = op(w_t)
        return y_t.detach().cpu().numpy().ravel()

    def rmatvec(r_np: np.ndarray) -> np.ndarray:
        r_t = torch.from_numpy(r_np.astype(np.float32)).to(device).view_as(y)
        g_t = op.adjoint(r_t)
        return g_t.detach().cpu().numpy().ravel()

    A_linop = LinearOperator(
        shape=(n_y, n_w), matvec=matvec, rmatvec=rmatvec, dtype=np.float32
    )

    y_np = y.detach().cpu().numpy().ravel()
    # x0_np = np.zeros(n_w, dtype=np.float32)
    # x_np, _, _, _ = spgl1(A_linop, y_np, x0=x0_np, **spgl1_kwargs)
    # print(f"Type of A_linop: {type(A_linop)}")
    # print(f"Type of y_np: {type(y_np)}")
    # for k in spgl1_kwargs:
    #     print(f"SPGL1 kwarg: {k} = {spgl1_kwargs[k]} with type {type(spgl1_kwargs[k])}")
    x_np, _, _, info = spg_bp(A_linop, y_np, **spgl1_kwargs)

    w_hat = torch.from_numpy(x_np.astype(np.float32)).to(device).view_as(w0)
    return w_hat, info


@dataclass(kw_only=True)
class SPGL1Parameter(LIONParameter):
    """Configuration for SPGL1 reconstruction.

    Args:
        factor: Factor to multiply the data by before passing to SPGL1.
            Useful for scaling the data to a range that SPGL1 can handle well.
        iters: Maximum number of iterations for SPGL1.
        debias_iters: Maximum number of iterations for the debiasing step.
        debias_support_tol: Tolerance for support detection in the debiasing step.
        debias_tol: Tolerance for convergence in the debiasing step.
        wavelet_name: Name of the wavelet (e.g., "db4") to use for the reconstruction.
    """

    factor: float = 1.0
    iters: int = 100
    debias_iters: int = 10
    debias_support_tol: float = 1e-5
    debias_tol: float = 1e-7
    wavelet_name: str = "db4"
