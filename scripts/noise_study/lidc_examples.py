# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Visual check of LION's LIDC-IDRI experiments: the noisy sinograms the
experiments now give (physical model, ct_experiments.CLINICAL_I0 etc., water
at 80 keV), reconstructed with FDK, next to the old relative model at its old
I0 and the ground truth. Needs a GPU, tomosipo and the LIDC-IDRI data. A few
minutes.

It looks at --n-slices slices of the test set and shows three of them: a
small, a typical and a large patient (5th, 50th and 95th percentile of the
thickest water-equivalent path). For each experiment (clinical, low dose,
extremely low dose) it writes lidc_examples_<experiment>.png: one row per
slice; columns: ground truth, old (relative), new (the experiment as it is
now). PSNR and SSIM against the ground truth are in the titles.

The new sinograms come from the experiment itself (dataset[i]), so this
shows exactly what training and testing on these experiments will see.
--detector photon_counting shows the same experiments with a photon-counting
detector instead.

--sweep 1e5,3e4,1e4,3e3,1e3 writes lidc_sweep.png instead: the three patients
with the old relative model at its three doses and the physical model at
each I0 given, side by side, to choose the experiments' I0 by eye.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from LION.classical_algorithms import fdk
from LION.CTtools import ct_utils as ct
from LION.experiments import ct_experiments
from LION.metrics.psnr import PSNR
from LION.metrics.ssim import SSIM

EXPERIMENTS = {
    "clinical": ct_experiments.clinicalCTRecon,
    "low dose": ct_experiments.LowDoseCTRecon,
    "extremely low dose": ct_experiments.ExtremeLowDoseCTRecon,
}
# The relative model's I0 of these experiments before 25 Oct 2026
RELATIVE_I0 = {"clinical": 10000, "low dose": 3500, "extremely low dose": 1000}


def pick_slices(data, op, n_slices, device):
    """Indices of a small, a typical and a large patient: the 5th, 50th and
    95th percentile of the thickest path, over n_slices of the dataset."""
    indices = np.linspace(0, len(data) - 1, n_slices).round().astype(int)
    paths = []
    for index in indices:
        _, image = data[int(index)]
        paths.append(op(image.to(device)).max().item() / 1.0137)  # mm of water
    paths = np.array(paths)
    picks = {}
    for label, q in (("small", 5), ("typical", 50), ("large", 95)):
        k = int(np.argmin(np.abs(paths - np.percentile(paths, q))))
        picks[label] = (int(indices[k]), paths[k])
    return picks


def sweep(args):
    """One figure: the three patients (rows); ground truth, the old relative
    model at its three doses, and the physical model at each --sweep I0."""
    device = torch.device(args.device)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)
    psnr = PSNR()
    torch.manual_seed(0)
    experiment = ct_experiments.clinicalCTRecon(dataset="LIDC-IDRI")
    noise = experiment.param.noise_params
    op = ct.make_operator(experiment.geometry)
    data = experiment.get_testing_dataset()
    picks = pick_slices(data, op, args.n_slices, device)
    I0s = [float(v) for v in args.sweep.split(",")]
    columns = [("ground truth", None)]
    columns += [
        (f"old {name}: relative, I0 {I0}", I0) for name, I0 in RELATIVE_I0.items()
    ]
    columns += [(f"physical, I0 {I0:g}", I0) for I0 in I0s]
    n_old = 1 + len(RELATIVE_I0)
    fig, axes = plt.subplots(
        len(picks), len(columns), figsize=(3.6 * len(columns), 3.9 * len(picks))
    )
    for row, (label, (index, path)) in zip(axes, picks.items()):
        _, image = data[index]
        image = image.to(device).unsqueeze(0)
        clean = op(image[0])
        line = []
        for k, (title, I0) in enumerate(columns):
            if I0 is None:
                img = image
            else:
                model = (
                    dict(model="relative")
                    if k < n_old
                    else dict(detector=args.detector)  # the physical model
                )
                sino = ct.sinogram_add_noise(
                    clean,
                    I0=I0,
                    sigma=noise.sigma,
                    sigma_blur=noise.sigma_blur,
                    **model,
                )
                img = fdk(sino.unsqueeze(0), op)
                value = psnr(img, image).item()
                title += f"\nPSNR {value:.1f}"
                line.append(f"{title.splitlines()[0]}: {value:.1f}")
            ax = row[k]
            ax.imshow(img[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=args.vmax)
            ax.set_title(f"{label} ({path:.0f} mm)\n{title}", fontsize=8)
            ax.axis("off")
        print(f"{label} patient, PSNR: " + "; ".join(line))
    fig.tight_layout()
    path = out / "lidc_sweep.png"
    fig.savefig(path, dpi=90)
    print(f"Saved {path}")


def main(args):
    if args.sweep:
        return sweep(args)
    device = torch.device(args.device)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)
    psnr, ssim = PSNR(), SSIM()
    torch.manual_seed(0)
    picks = None
    for name, Experiment in EXPERIMENTS.items():
        params = Experiment.default_parameters()
        if args.detector != "energy_integrating":
            params.noise_params.detector = args.detector
        experiment = Experiment(params, dataset="LIDC-IDRI")
        noise = experiment.param.noise_params
        op = ct.make_operator(experiment.geometry)
        data = experiment.get_testing_dataset()
        if picks is None:  # the same slices for every experiment
            picks = pick_slices(data, op, args.n_slices, device)
            for label, (index, path) in picks.items():
                print(
                    f"{label} patient: test slice {index}, thickest path {path:.0f} mm"
                )

        fig, axes = plt.subplots(len(picks), 3, figsize=(13, 4.4 * len(picks)))
        for row, (label, (index, path)) in zip(axes, picks.items()):
            sino_new, image = data[index]  # the experiment's own noisy sinogram
            image = image.to(device).unsqueeze(0)
            clean = op(image[0])
            sino_old = ct.sinogram_add_noise(
                clean,
                I0=RELATIVE_I0[name],
                sigma=noise.sigma,
                sigma_blur=noise.sigma_blur,
                model="relative",
            )
            panels = [
                ("ground truth", image),
                (
                    f"old: relative, I0 {RELATIVE_I0[name]}",
                    fdk(sino_old.unsqueeze(0), op),
                ),
                (
                    f"new: {noise.noise_model}, I0 {noise.I0:g}"
                    + (
                        f", {args.detector}"
                        if args.detector != "energy_integrating"
                        else ""
                    ),
                    fdk(sino_new.to(device).unsqueeze(0), op),
                ),
            ]
            for ax, (title, img) in zip(row, panels):
                ax.imshow(img[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=args.vmax)
                if title != "ground truth":
                    title += (
                        f"\nPSNR {psnr(img, image).item():.1f}, "
                        f"SSIM {ssim(img, image).item():.3f}"
                    )
                ax.set_title(f"{label} patient ({path:.0f} mm): {title}", fontsize=9)
                ax.axis("off")
        fig.suptitle(f"LIDC-IDRI, {name} experiment", fontsize=12)
        fig.tight_layout()
        path = out / f"lidc_examples_{name.replace(' ', '_')}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-slices", type=int, default=200)
    parser.add_argument(
        "--detector",
        default="energy_integrating",
        choices=ct.DETECTORS,
        help="detector of the new sinograms",
    )
    parser.add_argument(
        "--vmax", type=float, default=1.3, help="display window: 0 to vmax g/cm^3"
    )
    parser.add_argument(
        "--sweep",
        default=None,
        help="comma-separated physical I0 (e.g. 1e5,3e4,1e4,3e3,1e3): one figure "
        "with the old relative doses and the physical model at each of these",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    main(parser.parse_args())
