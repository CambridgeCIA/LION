# Check (to remove before merging): LG, LPD and cLPD's first step, before
# (ts_algorithms.fdk on a 2D sinogram, clipped) and after (LION's fdk).
# Run on holly: python check_one_fdk.py
import numpy as np
import torch
import ts_algorithms

from LION.classical_algorithms.fdk import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI

torch.manual_seed(0)
fan = LIDC_IDRI.default_geometry(np.linspace(0, 2 * np.pi, 360, endpoint=False))
parallel = Geometry(
    mode="parallel",
    image_shape=fan.image_shape,
    image_size=fan.image_size,
    detector_shape=fan.detector_shape,
    detector_size=fan.detector_size,
    angles=np.linspace(0, np.pi, 360, endpoint=False),
)
for name, geometry in (("fan", fan), ("parallel", parallel)):
    op = make_operator(geometry)
    image = torch.zeros(1, *geometry.image_shape[1:], device="cuda")
    image[0, 150:350, 100:400] = 1.0
    image[0, 220:260, 200:240] = 2.0
    sino = op(image)  # (1, angles, detectors)
    new = fdk(sino, op)[0]
    try:
        old = torch.clip(ts_algorithms.fdk(op, sino[0]), min=0)
    except Exception as error:  # noqa: BLE001
        print(f"{name}: before, ts_algorithms.fdk raised {error!r}")
        continue
    old = old.reshape(new.shape)
    diff = (old - new).abs().max().item()
    print(
        f"{name}: max |before - after| = {diff:.3g} "
        f"(image max {new.max().item():.3g}); "
        f"after vs phantom: mean abs error {(new - image[0]).abs().mean().item():.3g}, "
        f"before: {(old - image[0]).abs().mean().item():.3g}"
    )
