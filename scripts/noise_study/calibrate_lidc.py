# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Part B of the noise study: what the relative-model I0 of LION's LIDC-IDRI
experiments means in photons, slice by slice, and which physical I0 would
give the same reconstruction quality. Needs a GPU, tomosipo and the
LIDC-IDRI data (as for any LION experiment). About 10 minutes with the
defaults.

For each experiment (clinical, low dose, extremely low dose) it takes
--n-slices slices spread over the test set and

1. (analytic, every slice) computes the physical I0 that gives the same
   total sinogram noise as the relative model at the experiment's I0 (see
   noise_tools.equivalent_physical_I0). Under the relative model this
   "effective dose" changes from slice to slice: the script reports its
   spread and how it grows with the slice's attenuation.
2. (simulated, --fdk-slices of them) reconstructs with FDK the relative-model
   sinogram and physical-model sinograms over a grid of I0, and finds the
   physical I0 with the same PSNR as the relative model, per slice. This
   includes what the formula cannot: photon starvation, clipping, the
   reconstruction itself.

The median of 2. is the physical I0 that keeps an experiment as hard on
average as it is now. The spread of 1. and 2. is what the relative model does
that a scanner does not: it changes the dose with the slice.

Writes lidc_calibration.csv (one row per slice and experiment),
lidc_calibration.png and lidc_examples.png to --out (default: next to this
script).
"""

import argparse
import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from LION.classical_algorithms import fdk
from LION.CTtools import ct_utils as ct
from LION.experiments import ct_experiments
from LION.metrics.psnr import PSNR
from LION.metrics.ssim import SSIM

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import noise_tools as nt  # noqa: E402

EXPERIMENTS = {
    "clinical": ct_experiments.clinicalCTRecon,
    "low dose": ct_experiments.LowDoseCTRecon,
    "extremely low dose": ct_experiments.ExtremeLowDoseCTRecon,
}
# Physical I0 grid for the PSNR matching (photons per detector pixel in vacuum)
I0_GRID = np.geomspace(1e3, 1e7, 17)
# Rays through less than 1 mm of water are air: left out of the analytic match
MIN_LINE_INTEGRAL = 1.0


def add_noise(p, noise, I0, mass_attenuation=None):
    return ct.sinogram_add_noise(
        p,
        I0=I0,
        sigma=noise.sigma,
        sigma_blur=noise.sigma_blur,
        mass_attenuation=mass_attenuation,
    )


def matching_I0(psnr_grid, psnr_target):
    """The I0 of the grid, interpolated in log I0, where the PSNR reaches the
    target (PSNR increases with I0). nan if the target is outside the grid."""
    log_grid = np.log(I0_GRID)
    order = np.argsort(psnr_grid)
    psnr_sorted = np.asarray(psnr_grid)[order]
    if not psnr_sorted[0] <= psnr_target <= psnr_sorted[-1]:
        return float("nan")
    return float(np.exp(np.interp(psnr_target, psnr_sorted, log_grid[order])))


def main(args):
    device = torch.device(args.device)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)
    mu_rho = ct.WATER_MASS_ATTENUATION[args.energy]
    psnr, ssim = PSNR(), SSIM()
    torch.manual_seed(0)

    rows, examples = [], {}
    for name, Experiment in EXPERIMENTS.items():
        experiment = Experiment(dataset="LIDC-IDRI")
        noise = experiment.param.noise_params
        geometry = experiment.geometry
        op = ct.make_operator(geometry)
        w = nt.blur_variance_factor(noise.sigma_blur)
        data = experiment.get_testing_dataset()
        indices = np.linspace(0, len(data) - 1, args.n_slices).round().astype(int)
        fdk_every = max(1, args.n_slices // args.fdk_slices)
        print(f"--- {name}: relative I0 = {noise.I0}, {len(indices)} slices ---")

        for count, index in enumerate(indices):
            _, image = data[int(index)]  # the density image (g/cm^3)
            image = image.to(device).unsqueeze(0)  # (1, 1, H, W)
            p = op(image[0]).unsqueeze(0)  # noise-free line integrals (1, 1, A, D)

            row = {
                "experiment": name,
                "slice": int(index),
                "relative_I0": noise.I0,
                # thickest water-equivalent path through the slice, in mm
                "max_path_mm": p.max().item() / 1.0137,
                "equivalent_I0_analytic": nt.equivalent_physical_I0(
                    p.cpu(),
                    noise.I0,
                    mu_rho,
                    sigma=noise.sigma,
                    w=w,
                    min_line_integral=MIN_LINE_INTEGRAL,
                ).item(),
            }

            if count % fdk_every == 0:
                rel = fdk(add_noise(p, noise, noise.I0), op)
                row["psnr_relative"] = psnr(rel, image).item()
                row["ssim_relative"] = ssim(rel, image).item()
                grid = [
                    psnr(fdk(add_noise(p, noise, I0, mu_rho), op), image).item()
                    for I0 in I0_GRID
                ]
                row["equivalent_I0_fdk"] = matching_I0(grid, row["psnr_relative"])
                # keep the least and most attenuating slices to show
                ex = examples.setdefault(name, {})
                for key, better in (("small", np.less), ("large", np.greater)):
                    if key not in ex or better(
                        row["max_path_mm"], ex[key]["max_path_mm"]
                    ):
                        ex[key] = {
                            "p": p,
                            "image": image,
                            "max_path_mm": row["max_path_mm"],
                        }
            rows.append(row)

        summarise(name, [r for r in rows if r["experiment"] == name])

    # recommended physical I0: the median FDK match of each experiment
    recommended = {}
    for name in EXPERIMENTS:
        matches = [
            r.get("equivalent_I0_fdk", np.nan) for r in rows if r["experiment"] == name
        ]
        recommended[name] = float(np.nanmedian(matches))
    print(
        "\nPhysical I0 that keeps each experiment as hard on average (median FDK match):"
    )
    for name, I0 in recommended.items():
        print(f"  {name}: {I0:.3g}")

    write_csv(rows, out / "lidc_calibration.csv")
    plot_calibration(rows, out / "lidc_calibration.png")
    plot_examples(examples, recommended, mu_rho, out / "lidc_examples.png")
    print(
        f"Saved lidc_calibration.csv, lidc_calibration.png, lidc_examples.png to {out}"
    )


def summarise(name, rows):
    analytic = np.array([r["equivalent_I0_analytic"] for r in rows])
    path = np.array([r["max_path_mm"] for r in rows])
    q = np.percentile(analytic, [5, 50, 95])
    print(
        f"analytic equivalent physical I0: median {q[1]:.3g}, 5-95 % {q[0]:.3g} - {q[2]:.3g}"
        f" (x{q[2] / q[0]:.1f}); correlation with the thickest path:"
        f" {np.corrcoef(np.log(analytic), path)[0, 1]:.2f}"
    )
    fdk_match = np.array(
        [r["equivalent_I0_fdk"] for r in rows if "equivalent_I0_fdk" in r]
    )
    if fdk_match.size:
        q = np.nanpercentile(fdk_match, [5, 50, 95])
        print(
            f"FDK-PSNR equivalent physical I0:  median {q[1]:.3g}, 5-95 % {q[0]:.3g} - {q[2]:.3g}"
            f" ({np.isnan(fdk_match).sum()} slices outside the I0 grid)"
        )


def write_csv(rows, path):
    keys = sorted(
        {k for r in rows for k in r},
        key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99,
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_calibration(rows, path):
    fig, axes = plt.subplots(
        1, len(EXPERIMENTS), figsize=(5 * len(EXPERIMENTS), 4), sharey=True
    )
    for ax, name in zip(axes, EXPERIMENTS):
        rs = [r for r in rows if r["experiment"] == name]
        ax.semilogy(
            [r["max_path_mm"] for r in rs],
            [r["equivalent_I0_analytic"] for r in rs],
            ".",
            alpha=0.5,
            label="analytic (total noise)",
        )
        fd = [r for r in rs if "equivalent_I0_fdk" in r]
        ax.semilogy(
            [r["max_path_mm"] for r in fd],
            [r["equivalent_I0_fdk"] for r in fd],
            "o",
            mfc="none",
            label="FDK PSNR match",
        )
        ax.axhline(rs[0]["relative_I0"], color="grey", lw=0.5, label="relative I0")
        ax.set_title(f"{name}: relative I0 = {rs[0]['relative_I0']}")
        ax.set_xlabel("thickest water-equivalent path in the slice (mm)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("physical I0 with the same noise")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_examples(examples, recommended, mu_rho, path):
    """FDK of the least and the most attenuating slice: relative model at the
    experiment's I0 vs physical model at the recommended I0. Same average
    difficulty; the difference is where the noise goes."""
    fig, axes = plt.subplots(
        len(examples), 4, figsize=(16, 4 * len(examples)), squeeze=False
    )
    for row_axes, (name, ex) in zip(axes, examples.items()):
        experiment = EXPERIMENTS[name](dataset="LIDC-IDRI")
        noise, op = experiment.param.noise_params, ct.make_operator(experiment.geometry)
        col = 0
        for key in ("small", "large"):
            p, image = ex[key]["p"], ex[key]["image"]
            for model in ("relative", "physical"):
                if model == "relative":
                    recon = fdk(add_noise(p, noise, noise.I0), op)
                else:
                    recon = fdk(add_noise(p, noise, recommended[name], mu_rho), op)
                ax = row_axes[col]
                ax.imshow(recon[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1.3)
                ax.set_title(
                    f"{name}, {ex[key]['max_path_mm']:.0f} mm path: {model}\n"
                    f"PSNR {PSNR()(recon, image).item():.1f}",
                    fontsize=9,
                )
                ax.axis("off")
                col += 1
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-slices", type=int, default=300)
    parser.add_argument("--fdk-slices", type=int, default=30)
    parser.add_argument(
        "--energy", type=int, default=60, choices=ct.WATER_MASS_ATTENUATION
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    main(parser.parse_args())
