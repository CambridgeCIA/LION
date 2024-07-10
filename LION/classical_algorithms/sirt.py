from typing import Callable, Optional
import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import sirt as ts_sirt
import tomosipo as ts


def sirt_from_geo(
    sino: torch.Tensor,
    geo: Geometry,
    num_iterations: int,
    min_constraint: Optional[float] = None,
    max_constraint: Optional[float] = None,
    x_init: Optional[torch.Tensor] = None,
    volume_mask: Optional[torch.Tensor] = None,
    projection_mask: Optional[torch.Tensor] = None,
    progress_bar: bool = False,
    callbacks: list[Callable] = [],
):
    op = make_operator(geo)
    return sirt(
        sino,
        op,
        num_iterations,
        min_constraint,
        max_constraint,
        x_init,
        volume_mask,
        projection_mask,
        progress_bar,
        callbacks,
    )


def sirt(
    sino: torch.Tensor,
    op: ts.Operator.Operator,
    num_iterations: int,
    min_constraint: Optional[float] = None,
    max_constraint: Optional[float] = None,
    x_init: Optional[torch.Tensor] = None,
    volume_mask: Optional[torch.Tensor] = None,
    projection_mask: Optional[torch.Tensor] = None,
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
        sub_recon = ts_sirt(
            op,
            sino,
            num_iterations,
            min_constraint,
            max_constraint,
            x_init,
            volume_mask,
            projection_mask,
            progress_bar,
            callbacks,
        )
        recon[i] = sub_recon
    return recon
