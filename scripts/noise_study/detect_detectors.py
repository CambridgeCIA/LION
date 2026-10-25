# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Visual check of the 2DeteCT detectors: does the measured low-dose noise
differ between detector 1 and detector 2, and does LION's 2DeteCT noise
simulation reproduce it with one I0 for all slices, or only with one I0 per
detector? Needs the 2DeteCT data and a GPU with tomosipo (for FDK). A few
minutes.

For --slices-per-detector slices of each detector (spread over its scans), it
reconstructs with FDK, from the same scan:

- mode2: high dose, the reference;
- mode1: the measured low dose;
- mode2 + LION's 2DeteCT noise (deteCT.add_sinogram_noise, as in Kiss et al.
  2024) at one I0 for every slice (--I0, default 200, the paper's value);
- the same at one I0 per detector (--I0-detector1, --I0-detector2; defaults
  290 and 100, what detect_measured_noise.py measured, in simulator units).

detect_detectors.png: per slice, the four reconstructions (top) and their
noise, i.e. each minus the mode2 reconstruction, on one colour scale
(bottom). The noise std inside the object is in each title. mode1 minus mode2
also contains mode2's own noise, ~3 % of mode1's variance (mode2 has ~40x
more photons), so it is comparable with the simulations, which add noise to
mode2.

detect_detectors_noise.png: the noise variance of the sinograms against the
line integral (from neighbouring views, as in detect_measured_noise.py),
measured mode1 against both simulations, one panel per detector.

detect_detectors.csv: the numbers.
"""

import argparse
import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from LION.classical_algorithms import fdk

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import noise_tools as nt  # noqa: E402
from detect_measured_noise import (  # noqa: E402
    N_BINS,
    P_MIN,
    load_all_slices,
    sinogram,
)


def pick_slices(data, detector, n):
    """n slice positions of one detector, spread over its scans (not at the
    first or last slice of the detector's range)."""
    positions = np.flatnonzero(data.slice_dataframe["detector"].values == detector)
    picks = np.linspace(0, len(positions) - 1, n + 2)[1:-1].round().astype(int)
    return positions[picks]


def noise_curve(p):
    """(line integral, noise variance) of a sinogram, binned."""
    r, guide = nt.angular_residual(p.double())
    top = torch.quantile(guide.reshape(-1)[::97].float(), 0.995).item()
    centres, var, counts = nt.binned_noise_variance(
        guide, r, np.linspace(P_MIN, top, N_BINS + 1)
    )
    fit = nt.fit_noise_model(centres, var, counts)
    return centres, var, fit["I0"]


def main(args):
    out = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = load_all_slices()
    op = data.get_operator()
    per_detector = {1: args.I0_detector1, 2: args.I0_detector2}
    torch.manual_seed(0)

    columns = [
        "mode2 (reference)",
        "mode1 (measured low dose)",
        f"simulated, I0 {args.I0:g} for all",
        "simulated, I0 per detector",
    ]
    slices, rows = [], []
    for detector in (1, 2):
        for index in pick_slices(data, detector, args.slices_per_detector):
            info = data.slice_dataframe.iloc[int(index)]
            p2 = sinogram(data, index, "mode2").float()
            p1 = sinogram(data, index, "mode1").float()
            with torch.no_grad():  # the loader's cross-talk is a Conv2d
                one = data.add_sinogram_noise(p2, args.I0).float()
                own = data.add_sinogram_noise(p2, per_detector[detector]).float()
            sinos = [p2, p1, one, own]
            recons = [fdk(s.to(device), op).cpu() for s in sinos]  # (1, H, W)
            inside = recons[0] > 0.1 * recons[0].max()
            stds = [float((r - recons[0])[inside].std()) for r in recons]
            curves = [noise_curve(s) for s in sinos[1:]]
            row = {
                "slice": info["slice_identifier"],
                "detector": detector,
                "I0_per_detector": per_detector[detector],
                "image_noise_std_mode1": stds[1],
                "image_noise_std_one_I0": stds[2],
                "image_noise_std_per_detector": stds[3],
                "sinogram_I0_fit_mode1": curves[0][2],
                "sinogram_I0_fit_one_I0": curves[1][2],
                "sinogram_I0_fit_per_detector": curves[2][2],
            }
            rows.append(row)
            slices.append((row, recons, stds, curves))
            print(
                f"{row['slice']} (detector {detector}): image noise std "
                f"mode1 {stds[1]:.4g}, I0 {args.I0:g} {stds[2]:.4g}, "
                f"I0 {per_detector[detector]:g} {stds[3]:.4g}"
            )

    with open(out / "detect_detectors.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_images(slices, columns, per_detector, out / "detect_detectors.png")
    plot_curves(slices, args.I0, per_detector, out / "detect_detectors_noise.png")
    print(f"Saved detect_detectors.png, detect_detectors_noise.png, .csv to {out}")


def plot_images(slices, columns, per_detector, path):
    fig, axes = plt.subplots(
        2 * len(slices), 4, figsize=(16, 8.4 * len(slices)), squeeze=False
    )
    for k, (row, recons, stds, _) in enumerate(slices):
        top, bottom = axes[2 * k], axes[2 * k + 1]
        vmax = float(np.percentile(recons[0].numpy(), 99.5))
        noise_range = 3 * stds[1]  # the measured noise sets the colour scale
        for j, (recon, std) in enumerate(zip(recons, stds)):
            title = columns[j]
            if j == 3:
                title = f"simulated, I0 {per_detector[row['detector']]:g} (detector)"
            top[j].imshow(recon[0].numpy(), cmap="gray", vmin=0, vmax=vmax)
            top[j].set_title(
                f"{row['slice']}, detector {row['detector']}\n{title}", fontsize=9
            )
            if j == 0:
                bottom[j].text(0.5, 0.5, "minus mode2:\nthe noise", ha="center")
            else:
                bottom[j].imshow(
                    (recon - recons[0])[0].numpy(),
                    cmap="gray",
                    vmin=-noise_range,
                    vmax=noise_range,
                )
                bottom[j].set_title(f"noise std {std:.3g}", fontsize=9)
        for ax in (*top, *bottom):
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def plot_curves(slices, I0, per_detector, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, detector in zip(axes, (1, 2)):
        for k, (row, _, _, curves) in enumerate(
            s for s in slices if s[0]["detector"] == detector
        ):
            colour = f"C{k}"
            labels = [
                "measured mode1",
                f"simulated, I0 {I0:g}",
                f"simulated, I0 {per_detector[detector]:g}",
            ]
            for (centres, var, _), style, label in zip(
                curves, ("-", "--", ":"), labels
            ):
                ax.semilogy(
                    centres, var, style, color=colour, label=f"{row['slice']}: {label}"
                )
        ax.set_title(f"detector {detector}")
        ax.set_xlabel("line integral p = -log T")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("noise variance of the sinogram")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slices-per-detector", type=int, default=2)
    parser.add_argument("--I0", type=float, default=200)
    parser.add_argument("--I0-detector1", type=float, default=290)
    parser.add_argument("--I0-detector2", type=float, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    main(parser.parse_args())
