import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import fdk as ts_fdk

def fdk(sino: torch.Tensor, geo: Geometry):
    B, _, _, _ = sino.shape
    recon = sino.new_zeros(B, 1, *geo.image_shape[1:])
    op = make_operator(geo)
    # ts fdk doesn't support mini-batches so we apply it one at a time to each batch
    for i in range(B):
        sub_recon = ts_fdk(op, sino)
        sub_recon = torch.clip(sub_recon, min=0)
        recon[i] = sub_recon
    return recon
