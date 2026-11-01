# Check (to remove before merging): LION's FDK of a clean foam sinogram should
# match its image. Run in the lion env, in the folder that holds Dataset_Hackathon/.
import json
import os

import h5py
import numpy as np
import torch

import LION.CTtools.ct_geometry as ctgeo
from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
from LION.utils.device import default_device

folder = "./Dataset_Hackathon/"
with open(os.path.join(folder, "geometry.json")) as f:
    values = json.load(f)
values["angles"] = np.array(values["angles"])
geometry = ctgeo.Geometry(**values)
op = make_operator(geometry)

for i in range(3):
    with h5py.File(os.path.join(folder, f"sino_{i:04d}.h5")) as f:
        sino = torch.from_numpy(f["sinogram"][:]).float()[None, None]
    with h5py.File(os.path.join(folder, f"gt_{i:04d}.h5")) as f:
        image = f["image"][:]
    recon = fdk(sino.to(default_device()), op).squeeze().cpu().numpy()
    centre = np.s_[64:-64, 64:-64]  # away from the edge of the field of view
    a, b = recon[centre].ravel(), image[centre].ravel()
    print(
        f"sample {i}: correlation {np.corrcoef(a, b)[0, 1]:.3f}, "
        f"mean FDK / mean image {a.mean() / b.mean():.3f}, "
        f"flipped image correlation {np.corrcoef(a, image[::-1][centre].ravel())[0, 1]:.3f}"
    )
