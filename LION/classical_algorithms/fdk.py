import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import fdk as ts_fdk
import tomosipo as ts

from LION.exceptions.exceptions import NoDataException


def fdk_from_geo(sino: torch.Tensor, geo: Geometry, clip=True):
    op = make_operator(geo)
    return fdk(sino, op, clip)

# add option to clip?
def fdk(sino: torch.Tensor, op: ts.Operator.Operator, clip=True) -> torch.Tensor:
    B, _, _, _ = sino.shape
    if B == 0: 
        raise NoDataException("Given 0 batches, no data to operate on!")
    recon = sino.new_zeros(B, *op.domain_shape)
    # ts fdk doesn't support mini-batches so we apply it one at a time to each batch
    for i in range(B):
        sub_recon = ts_fdk(op, sino[i])
        if clip:
            sub_recon = torch.clip(sub_recon, min=0)
        recon[i] = sub_recon
    return recon
