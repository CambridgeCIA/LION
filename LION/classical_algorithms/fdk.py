from __future__ import annotations
import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator

from LION.exceptions.exceptions import NoDataException
from LION.operators.Operator import Operator


# add option to clip?
def fdk(sino: torch.Tensor, op: Operator | Geometry, clip=True) -> torch.Tensor:
    from ts_algorithms import fdk as ts_fdk

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
    # ts fdk doesn't support mini-batches so we apply it one at a time to each batch
    for i in range(B):
        sub_recon = ts_fdk(op, sino[i])
        if clip:
            sub_recon = torch.clip(sub_recon, min=0)
        recon[i] = sub_recon
    if remove_batch:
        recon = recon.squeeze(0)
    return recon
