# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Photon starvation with the physical noise model: an abdomen-sized phantom
(330 x 250 mm, thickest path ~350 mm of water) in a parallel-beam geometry
(scikit-image's radon/iradon, so CPU only, ~1 minute), with LION's noise.

Compares the relative model at I0 = 1e4 with the physical model at I0 = 1e4,
1e5 and 1e6, with LION's old rule for starved rays (signal <= 0 -> 1e-6) and
with a floor of 1 photon (sinogram_add_noise's min_counts). Saves
starvation_demo.png next to this script (or to the path given).
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio as psnr  # noqa: E402
from skimage.transform import iradon, radon  # noqa: E402

from LION.CTtools import ct_utils as ct  # noqa: E402

N, fov = 512, 400.0
px = fov / N
y, x = (np.mgrid[:N, :N] - N / 2 + 0.5) * px
img = np.zeros((N, N))
img[(x / 165) ** 2 + (y / 125) ** 2 < 1] = 0.95  # fat shell
img[(x / 155) ** 2 + (y / 115) ** 2 < 1] = 1.05  # soft tissue: 330 x 250 mm
img[((x + 50) / 60) ** 2 + ((y + 20) / 50) ** 2 < 1] = 1.06  # liver-ish
img[(x / 18) ** 2 + ((y - 70) / 18) ** 2 < 1] = 1.5  # vertebra
img[((x - 60) / 25) ** 2 + ((y + 40) / 20) ** 2 < 1] = 0.3  # gas
for cx in (-120, 120):
    img[((x - cx) / 6) ** 2 + ((y - 30) / 30) ** 2 < 1] = 1.6  # ribs
theta = np.linspace(0, 180, 720, endpoint=False)
p = radon(img, theta, circle=False) * px  # density * mm
print("max path (water-eq mm)", p.max() / 1.0137)
sino = torch.from_numpy(p.T[None].astype(np.float32))  # (1, angles, detectors)
mu = ct.DEFAULT_MASS_ATTENUATION  # water at 80 keV


def recon(s):
    return (
        iradon(s[0].numpy().T, theta, circle=False, output_size=N, filter_name="ramp")
        / px
    )


cases = [
    ("relative, I0 1e4", dict(I0=1e4, model="relative")),
    (
        "physical, I0 1e4, old 1e-6 rule",
        dict(I0=1e4, mass_attenuation=mu, min_counts=None),
    ),
    (
        "physical, I0 1e4, floor 1 photon",
        dict(I0=1e4, mass_attenuation=mu, min_counts=1.0),
    ),
    (
        "physical, I0 1e5, old 1e-6 rule",
        dict(I0=1e5, mass_attenuation=mu, min_counts=None),
    ),
    (
        "physical, I0 1e5, floor 1 photon",
        dict(I0=1e5, mass_attenuation=mu, min_counts=1.0),
    ),
    (
        "physical, I0 1e6, floor 1 photon",
        dict(I0=1e6, mass_attenuation=mu, min_counts=1.0),
    ),
]
fig, ax = plt.subplots(2, 3, figsize=(15, 11))
for a, (name, kw) in zip(ax.flat, cases):
    torch.manual_seed(0)
    n = ct.sinogram_add_noise(sino, sigma=5, **kw)
    r = recon(n)
    T = torch.exp(-sino / (1 / (mu * 0.1)))
    starved = (
        ((kw["I0"] * T) < 10).float().mean().item() if "mass_attenuation" in kw else 0
    )
    val = psnr(img, r, data_range=img.max())
    print(
        f"{name}: PSNR {val:.1f}, rays < 10 photons {100*starved:.1f} %, max p_noisy/max p {n.max().item()/sino.max().item():.2f}"
    )
    a.imshow(r, cmap="gray", vmin=0.8, vmax=1.3)
    a.set_title(
        f"{name}\nPSNR {val:.1f} dB, starved rays {100*starved:.0f} %", fontsize=10
    )
    a.axis("off")
fig.suptitle(
    f"330 x 250 mm abdomen phantom (thickest path {p.max()/1.0137:.0f} mm water), 80 keV, sigma = 5; window 0.8-1.3 g/cm3"
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = (
    sys.argv[1]
    if len(sys.argv) > 1
    else pathlib.Path(__file__).with_name("starvation_demo.png")
)
fig.savefig(out, dpi=110)
print(f"Saved {out}")
