from typing import Callable, Optional
import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import tv_min2d as ts_tv_min
import tomosipo as ts


def tv_min_from_geo(
    sino: torch.Tensor,
    geo: Geometry,
    lam: float,
    num_iterations: int = 500,
    L: Optional[float] = None,
    non_negativity: bool = False,
    progress_bar: bool = False,
    callbacks: list[Callable] = [],
):
    op = make_operator(geo)
    return tv_min(
        sino, op, lam, num_iterations, L, non_negativity, progress_bar, callbacks
    )


def tv_min(
    sino: torch.Tensor,
    op: ts.Operator.Operator,
    lam: float,
    num_iterations: int = 500,
    L: Optional[float] = None,
    non_negativity: bool = False,
    progress_bar: bool = False,
    callbacks: list[Callable] = [],
) -> torch.Tensor:
    """Computes the total-variation minimization using Chambolle-Pock on a batched input.\n
        See ts_algorithms.tv_min2d for more details.
    """
    B, _, _, _ = sino.shape
    recon = sino.new_zeros(B, *op.domain_shape)
    assert B > 0, "Given 0 batches, no data to operate on!"
    for i in range(B):
        sub_recon = ts_tv_min(
            op, sino, lam, num_iterations, L, non_negativity, progress_bar, callbacks
        )
        recon[i] = sub_recon
    return recon
