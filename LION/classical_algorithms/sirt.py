from __future__ import annotations
from typing import Callable, Optional
import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import sirt as ts_sirt
import tomosipo as ts

from LION.exceptions.exceptions import NoDataException


def sirt(
    sino: torch.Tensor,
    op: ts.Operator.Operator | Geometry,
    num_iterations: int,
    min_constraint: Optional[float] = None,
    max_constraint: Optional[float] = None,
    x_init: Optional[torch.Tensor] = None,
    volume_mask: Optional[torch.Tensor] = None,
    projection_mask: Optional[torch.Tensor] = None,
    progress_bar: bool = False,
    callbacks: list[Callable] = [],
) -> torch.Tensor:
    """Compute SIRT reconstructions on a batched input.

    See ts_algorithms.sirt for more details.
    """
    if sino.dim() == 4:
        B, _, _, _ = sino.shape
        remove_batch = False
    elif sino.dim() == 3:
        B = 1
        sino = sino.unsqueeze(0)
        remove_batch = True
    if B == 0:
        raise NoDataException("Given 0 batches, no data to operate on!")
    if isinstance(op, Geometry):
        op = make_operator(op)
    recon = sino.new_zeros(B, *op.domain_shape)
    for i in range(B):
        sub_recon = ts_sirt(
            op,
            sino[i],
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
    if remove_batch:
        recon = recon.squeeze(0)
    return recon
