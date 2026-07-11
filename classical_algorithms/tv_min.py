"""Chambolle--Pock total-variation reconstruction."""

from __future__ import annotations
from typing import Callable, Optional
import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import tv_min2d as ts_tv_min
import tomosipo as ts

from LION.exceptions.exceptions import NoDataException


def tv_min(
    sino: torch.Tensor,
    op: ts.Operator.Operator | Geometry,
    lam: float,
    num_iterations: int = 500,
    L: Optional[float] = None,
    non_negativity: bool = False,
    progress_bar: bool = False,
    callbacks: list[Callable] = [],
) -> torch.Tensor:
    """Minimise a TV-regularised CT objective with Chambolle--Pock.

    Parameters
    ----------
    sino : torch.Tensor
        Batched sinograms in ``NCHW``-like projection layout.
    op : tomosipo.Operator or Geometry
        Forward projection operator or source geometry.
    lam : float
        TV regularisation weight.
    num_iterations : int, optional
        Chambolle--Pock iterations.
    L : float, optional
        Precomputed operator norm used by the backend.
    non_negativity : bool, optional
        Constrain reconstruction values to be non-negative.
    progress_bar : bool, optional
        Display backend progress.
    callbacks : list of callable, optional
        Iteration callbacks passed to ``ts_algorithms.tv_min2d``.

    Returns
    -------
    torch.Tensor
        Batched TV reconstructions.
    """
    B, _, _, _ = sino.shape
    if B == 0:
        raise NoDataException("Given 0 batches, no data to operate on!")
    if isinstance(op, Geometry):
        op = make_operator(op)
    recon = sino.new_zeros(B, *op.domain_shape)
    for i in range(B):
        sub_recon = ts_tv_min(
            op,
            sino[i],
            lam,
            num_iterations,
            L,
            non_negativity,
            progress_bar,
            callbacks,
        )
        recon[i] = sub_recon
    return recon
