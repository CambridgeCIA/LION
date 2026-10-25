# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Part A of the noise study: the relative and physical noise models on water
cylinders of 100-450 mm, in the default LIDC-IDRI geometry. CPU only, no data
needed, about a minute.

The relative model (LION's default) divides each sinogram by its maximum, so
the most attenuating ray of every object keeps I0/e photons: a 10 cm child
and a 45 cm adult get the same noise at their centre, and no object ever
reaches photon starvation. The physical model keeps the real transmission:
I0 is photons per detector pixel in vacuum, and bigger objects get noisier.

For each experiment's I0 this prints
- the photons left at the central ray under each model,
- the physical I0 that gives the same total sinogram noise (and so the same
  average FBP image noise) as the relative model, per diameter,
- the fraction of rays that are photon-starved (< 10 photons) under the
  physical model at the experiment's nominal I0,
and checks the analytic numbers against LION's actual noise simulation. It
saves a figure, water_cylinders.png, next to this script (or to --out).
"""

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from LION.CTtools import ct_utils as ct
from LION.CTtools.ct_geometry import Geometry

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import noise_tools as nt  # noqa: E402

# The I0 of LION's CT experiments (ct_experiments.py), and the default
# electronic noise and cross-talk.
EXPERIMENTS = {"clinical": 10000, "low dose": 3500, "extremely low dose": 1000}
SIGMA, SIGMA_BLUR = 5.0, 0.3015
DIAMETERS = [100, 150, 200, 250, 300, 350, 400, 450]  # mm


def main(energy_keV=60, out=None):
    mu_rho = ct.WATER_MASS_ATTENUATION[energy_keV]
    s_phys = nt.physical_scale(mu_rho)
    w = nt.blur_variance_factor(SIGMA_BLUR)
    geometry = Geometry.default_parameters()

    # a few views are enough: the cylinder looks the same from every angle
    sinos = torch.stack(
        [nt.water_cylinder_sinogram(d, geometry)[:, :4] for d in DIAMETERS]
    )  # (diameters, 1, 4, detectors)
    centre = sinos.shape[-1] // 2

    print(f"Water at {energy_keV} keV: mu/rho = {mu_rho} cm^2/g")
    print(f"Default geometry: {geometry.detector_shape[-1]} detectors, fan beam\n")

    equivalent = {}
    for name, I0 in EXPERIMENTS.items():
        eq = nt.equivalent_physical_I0(
            sinos, I0, mu_rho, sigma=SIGMA, w=w
        )  # (diameters,)
        equivalent[name] = eq
        print(f"--- {name}: I0 = {I0} ---")
        print(
            "diameter | photons at centre, relative | photons at centre, physical"
            " | physical I0 for the same image noise | starved rays, physical"
        )
        for k, d in enumerate(DIAMETERS):
            p = sinos[k]
            T_phys = torch.exp(-p / s_phys)
            inside = p[0, 0] > 0
            starved = (I0 * T_phys[0, 0][inside] < 10).double().mean().item()
            print(
                f"{d:5d} mm | {I0 / np.e:10.0f} | {I0 * T_phys.min().item():10.1f}"
                f" | {eq[k].item():12.0f} ({eq[k].item() / I0:6.2f} x I0)"
                f" | {100 * starved:5.1f} %"
            )
        print()

    # Check the analytic numbers against LION's simulation: at the equivalent
    # I0, the physical model must give the noise of the relative model.
    print("Check against sinogram_add_noise (4000 views per diameter):")
    torch.manual_seed(0)
    I0 = EXPERIMENTS["low dose"]
    for k in (0, 3, 6):
        d = DIAMETERS[k]
        p = nt.water_cylinder_sinogram(d, geometry)[:, :1].repeat(1, 4000, 1).float()
        rel = ct.sinogram_add_noise(
            p, I0=I0, sigma=SIGMA, sigma_blur=SIGMA_BLUR, model="relative"
        )
        phys = ct.sinogram_add_noise(
            p,
            I0=equivalent["low dose"][k].item(),
            sigma=SIGMA,
            sigma_blur=SIGMA_BLUR,
            mass_attenuation=mu_rho,
            min_counts=None,  # as the formula: no floor
        )
        # total over the rays through the cylinder of the per-ray variance
        # (over the 4000 views)
        inside = p[0, 0] > 0
        total_rel = rel[0].double().var(dim=0)[inside].sum().item()
        total_phys = phys[0].double().var(dim=0)[inside].sum().item()
        predicted = (
            nt.predicted_variance(
                p[0, 0].double(), I0, nt.relative_scale(p.double()), SIGMA, w
            )[inside]
            .sum()
            .item()
        )
        print(
            f"{d} mm: total variance, relative {total_rel:.4g} (predicted"
            f" {predicted:.4g}), physical at the equivalent I0 {total_phys:.4g};"
            f" central ray std: relative {rel[0, :, centre].std():.3g},"
            f" physical {phys[0, :, centre].std():.3g}"
        )
    print(
        "Same total noise, distributed differently: the physical model puts\n"
        "more of it on the most attenuating rays. Where the physical total is\n"
        "larger than predicted, some rays got (almost) no photons: zero counts,\n"
        "or a detector signal pushed to <= 0 by the electronic noise, which\n"
        "sinogram_add_noise replaces by 1e-6 before the log (a huge line\n"
        "integral). The first-order formula misses that.\n"
    )

    # Figure: central-ray noise against size, and the equivalent I0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, I0 in EXPERIMENTS.items():
        rel_std = [
            nt.predicted_variance(
                sinos[k][0, 0, centre], I0, sinos[k].max(), SIGMA, w
            ).sqrt()
            for k in range(len(DIAMETERS))
        ]
        phys_std = [
            nt.predicted_variance(sinos[k][0, 0, centre], I0, s_phys, SIGMA, w).sqrt()
            for k in range(len(DIAMETERS))
        ]
        (line,) = axes[0].semilogy(DIAMETERS, rel_std, "--", label=f"relative, {name}")
        axes[0].semilogy(
            DIAMETERS, phys_std, "-", color=line.get_color(), label=f"physical, {name}"
        )
        axes[1].semilogy(DIAMETERS, equivalent[name] / I0, label=name)
    axes[0].set_xlabel("water cylinder diameter (mm)")
    axes[0].set_ylabel("noise std of the central ray (density x mm)")
    axes[0].set_title("Noise at the same nominal I0")
    axes[0].legend(fontsize=7)
    axes[1].axhline(1, color="grey", lw=0.5)
    axes[1].set_xlabel("water cylinder diameter (mm)")
    axes[1].set_ylabel("physical I0 / relative I0")
    axes[1].set_title("Physical I0 giving the relative model's image noise")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    out = (
        pathlib.Path(out)
        if out
        else pathlib.Path(__file__).with_name("water_cylinders.png")
    )
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--energy", type=int, default=60, choices=ct.WATER_MASS_ATTENUATION
    )
    parser.add_argument("--out", default=None, help="where to save the figure")
    args = parser.parse_args()
    main(args.energy, args.out)
