# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Part C of the noise study: how does the noise of real, measured CT data
depend on attenuation? Uses the 2DeteCT raw sinograms (mode1: low dose,
mode2: high dose, same 90 kV, 3600 views). Needs the 2DeteCT data
(LION_DATA_PATH), no GPU and no tomosipo. About 15 minutes for 100 slices.

The two models make different, testable predictions about measured data:

- physical: the variance of a log-transformed measurement is
  var = w / (I0 T) + sigma^2 / (I0 T)^2, T = exp(-p): one I0 (and sigma) for
  every slice of the scanner, whatever the object;
- relative: var = s^2 (...) with T = exp(-p / s) and s the maximum of each
  slice's sinogram, so the I0 that fits a slice would depend on its maximum.

For each slice and mode the script measures the noise variance as a function
of the line integral, from differences between neighbouring views (no noise-
free reference needed, see noise_tools.angular_residual), and fits both
models. If the physical model is right, the fitted physical I0 is the same
for every slice (per detector) and the fitted relative I0 changes with the
slice's maximum; the physical model also fits the shape of the curve better.

It also
- checks the estimator with mode1 - mode2 (independent: its variance must be
  the sum of the two modes'),
- calibrates it against LION's own 2DeteCT noise simulation (mode2 +
  deteCT.add_sinogram_noise at a grid of I0), so that the measured mode1
  noise can be stated as the simulator's I0 (Kiss et al. 2024 found 200).

Writes detect_noise.csv (one row per slice) and detect_noise.png to --out.
"""

import argparse
import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from LION.data_loaders.deteCT import deteCT

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import noise_tools as nt  # noqa: E402

SIMULATOR_I0 = [100, 150, 200, 300, 400, 600]
N_BINS = 30
# Lowest line integral used: below it simulated data are clipped (LION's
# simulators cut transmission at 1), and real data most sensitive to
# flat-field errors.
P_MIN = 0.3


def load_all_slices():
    """deteCT over the whole dataset (all samples in the 'train' split),
    flat- and dark-field corrected and log-transformed, all 3600 views."""
    params = deteCT.default_parameters()
    params.task = "sino2sino"
    params.training_proportion, params.validation_proportion = 1.0, 0.0
    params.test_proportion = 0.0
    return deteCT("train", parameters=params)


def sinogram(data, index, mode):
    # the loader's own preprocessing (flat/dark field, log, orientation)
    return data.__load_and_preprocess_sinogram__(index, mode).double()


def measure(p, scale_for_relative):
    """Binned noise variance of one sinogram, and both fits."""
    r, guide = nt.angular_residual(p)
    top = torch.quantile(guide.reshape(-1)[::97].float(), 0.995).item()
    edges = np.linspace(P_MIN, top, N_BINS + 1)
    centres, var, counts = nt.binned_noise_variance(guide, r, edges)
    physical = nt.fit_noise_model(centres, var, counts, scale=1.0)
    relative = nt.fit_noise_model(centres, var, counts, scale=scale_for_relative)
    return (centres, var, counts), physical, relative


def main(args):
    out = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)
    data = load_all_slices()
    n = len(data)
    indices = np.linspace(0, n - 1, args.n_slices).round().astype(int)
    print(f"2DeteCT: {n} slices, analysing {len(indices)}")

    rows, curves = [], []
    for k, index in enumerate(indices):
        info = data.slice_dataframe.iloc[int(index)]
        p1, p2 = sinogram(data, index, "mode1"), sinogram(data, index, "mode2")
        # what the relative model would divide by: the (high dose) maximum
        s = nt.angular_residual(p2)[1].amax().item()
        row = {
            "slice": info["slice_identifier"],
            "detector": int(info["detector"]),
            "mix": int(info["mix"]),
            "max_line_integral": s,
        }
        for mode, p in (("mode1", p1), ("mode2", p2)):
            binned, physical, relative = measure(p, s)
            row.update(
                {
                    f"{mode}_I0_physical": physical["I0"],
                    f"{mode}_sigma_physical": physical["sigma"],
                    f"{mode}_misfit_physical": physical["rel_rms"],
                    f"{mode}_I0_relative": relative["I0"],
                    f"{mode}_misfit_relative": relative["rel_rms"],
                }
            )
            if mode == "mode1" and k % max(1, len(indices) // 6) == 0:
                curves.append((s, row["detector"], binned))
        # cross-check: mode1 - mode2 is independent noise of both modes
        d = (p1 - p2)[..., 3:-3, :]
        guide = nt.angular_residual(p2)[1]
        top = torch.quantile(guide.reshape(-1)[::97].float(), 0.995).item()
        edges = np.linspace(P_MIN, top, N_BINS + 1)
        c, v, cnt = nt.binned_noise_variance(guide, d, edges, r_variance_factor=1.0)
        row["difference_I0_physical"] = nt.fit_noise_model(c, v, cnt)["I0"]
        row["difference_I0_expected"] = 1 / (
            1 / row["mode1_I0_physical"] + 1 / row["mode2_I0_physical"]
        )
        rows.append(row)
        if k % 10 == 0:
            print(
                f"{k:4d}/{len(indices)} {row['slice']} det {row['detector']}:"
                f" max p {s:.2f}, mode1 I0 physical {row['mode1_I0_physical']:.0f}"
                f" relative {row['mode1_I0_relative']:.0f}"
            )

    calibration = calibrate_simulator(data, indices[:: max(1, len(indices) // 5)])
    summarise(rows, calibration)
    write_csv(rows, out / "detect_noise.csv")
    plot(rows, curves, calibration, out / "detect_noise.png")
    print(f"Saved detect_noise.csv and detect_noise.png to {out}")


def calibrate_simulator(data, indices):
    """Measured (physical-fit) I0 of mode2 + LION's 2DeteCT noise at each
    simulator I0, averaged over a few slices: the map from 'I0 measured by
    this estimator' to 'I0 of the simulator'."""
    torch.manual_seed(0)
    measured = []
    for I0 in SIMULATOR_I0:
        values = []
        for index in indices:
            p2 = sinogram(data, index, "mode2").float()
            with torch.no_grad():  # the loader's cross-talk is a Conv2d
                noisy = data.add_sinogram_noise(p2, I0).double()
            values.append(measure(noisy, 1.0)[1]["I0"])
        measured.append(float(np.median(values)))
        print(f"simulator I0 {I0}: measured as {measured[-1]:.0f}")
    return {"simulator_I0": SIMULATOR_I0, "measured_I0": measured}


def summarise(rows, calibration):
    for detector in sorted({r["detector"] for r in rows}):
        rs = [r for r in rows if r["detector"] == detector]
        m = np.array([r["max_line_integral"] for r in rs])
        print(f"\n--- detector {detector}: {len(rs)} slices ---")
        for model in ("physical", "relative"):
            I0 = np.array([r[f"mode1_I0_{model}"] for r in rs])
            misfit = np.array([r[f"mode1_misfit_{model}"] for r in rs])
            print(
                f"mode1, {model:8s} model: I0 median {np.median(I0):8.0f},"
                f" spread (CV) {np.std(I0) / np.mean(I0):5.1%},"
                f" correlation with the slice max {np.corrcoef(I0, m)[0, 1]:+.2f},"
                f" median misfit {np.median(misfit):.1%}"
            )
        ratio = np.array(
            [r["difference_I0_physical"] / r["difference_I0_expected"] for r in rs]
        )
        print(
            f"mode1 - mode2 check (measured / expected I0): median {np.median(ratio):.2f}"
        )
        mode1 = np.median([r["mode1_I0_physical"] for r in rs])
        sim = np.interp(
            np.log(mode1),
            np.log(calibration["measured_I0"]),
            np.log(calibration["simulator_I0"]),
        )
        print(f"mode1 noise = LION 2DeteCT simulator at I0 ~ {np.exp(sim):.0f}")
        sigma = np.median([r["mode1_sigma_physical"] for r in rs])
        print(f"electronic noise (physical fit): sigma ~ {sigma:.1f} counts at that I0")


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, curves, calibration, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    # (a) variance against line integral for slices of different size: one
    # curve if the physical model holds
    for s, detector, (c, v, _) in curves:
        axes[0].semilogy(c, v, ".-", label=f"max p {s:.1f}, det {detector}")
    axes[0].set_xlabel("line integral p = -log T")
    axes[0].set_ylabel("noise variance of p (mode1)")
    axes[0].set_title("Measured noise vs attenuation")
    axes[0].legend(fontsize=7)
    # (b) fitted I0 per slice under each model
    for model, marker in (("physical", "o"), ("relative", "x")):
        for detector in sorted({r["detector"] for r in rows}):
            rs = [r for r in rows if r["detector"] == detector]
            axes[1].semilogy(
                [r["max_line_integral"] for r in rs],
                [r[f"mode1_I0_{model}"] for r in rs],
                marker,
                mfc="none",
                label=f"{model}, det {detector}",
            )
    axes[1].set_xlabel("maximum line integral of the slice")
    axes[1].set_ylabel("fitted I0 (mode1)")
    axes[1].set_title("One I0 per scanner, or one per slice?")
    axes[1].legend(fontsize=7)
    # (c) estimator vs simulator
    axes[2].loglog(calibration["simulator_I0"], calibration["measured_I0"], "o-")
    axes[2].set_xlabel("I0 of deteCT.add_sinogram_noise")
    axes[2].set_ylabel("I0 measured on mode2 + simulated noise")
    axes[2].set_title("Calibration of the estimator")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-slices", type=int, default=100)
    parser.add_argument("--out", default=None)
    main(parser.parse_args())
