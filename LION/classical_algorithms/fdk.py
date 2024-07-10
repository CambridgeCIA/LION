import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import fdk as ts_fdk
import tomosipo as ts


def fdk_from_geo(sino: torch.Tensor, geo: Geometry):
    op = make_operator(geo)
    return fdk(sino, op, *geo.image_size[1:])


def fdk(sino: torch.Tensor, op: ts.Operator.Operator) -> torch.Tensor:
    B, _, _, _ = sino.shape
    assert B > 0, "Given 0 batches, no data to operate on!"
    recon = sino.new_zeros(B, *op.domain_shape)
    # ts fdk doesn't support mini-batches so we apply it one at a time to each batch
    for i in range(B):
        sub_recon = ts_fdk(op, sino[i])
        # don't want this line, not expected when user calls fdk, they can always add it afterwards. Don't hide sideffects.
        # sub_recon = torch.clip(sub_recon, min=0)
        recon[i] = sub_recon
    return recon
