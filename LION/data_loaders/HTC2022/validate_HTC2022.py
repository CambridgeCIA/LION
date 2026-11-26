# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Checks LION's preparation of the HTC2022 data (``pre_process_HTC2022.py``)
against the challenge's own files, phantom by phantom:

    python LION/data_loaders/HTC2022/validate_HTC2022.py [output folder]

1. **The turn:** the challenge's limited-angle sinogram against the first rows
   of the turned full one (``limited_difference``: 0 if the same data).
2. **The geometry and the recipe, without the turn:** LION's FBP of the full
   scan as released (not turned) against the challenge's FBP (correlation),
   and its segmentation against the challenge's (Matthews correlation
   coefficient, MCC, the challenge's score; 1: the same). Each is compared
   with the challenge's images as they are and flipped or turned by 90
   degrees: the best should be "as is"; another one means LION's geometry is
   mirrored or turned against the challenge's (``HTC2022.ANGLE_SIGN``).
3. **The targets:** the targets (turned) against the challenge's images
   turned by the same angle (interpolated: approximate), correlation and MCC.
4. **The operator:** the turned sinogram against the projection of its
   reconstruction (correlation).

Writes ``validation.csv`` (one row per phantom) and ``validation.png`` (for
each phantom: the challenge's segmentation turned, the target segmentation,
and their difference) in the output folder (default:
``LION_RESULTS_PATH/datasets/HTC2022_validation``), and prints a summary.

Needs the processed data, tomosipo and a GPU.
"""

import sys

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import rotate

from LION.data_loaders.HTC2022 import ANGLE_STEP, geometry, segment
from LION.operators.ct.fdk import fdk
from LION.utils.paths import HTC2022_PROCESSED_DATASET_PATH, results_path

ORIENTATIONS = {
    "as is": lambda x: x,
    "flipped up-down": np.flipud,
    "flipped left-right": np.fliplr,
    "transposed": np.transpose,
    "turned 90": lambda x: np.rot90(x, 1),
    "turned 180": lambda x: np.rot90(x, 2),
    "turned 270": lambda x: np.rot90(x, 3),
    "anti-transposed": lambda x: np.rot90(x, 2).T,
}


def correlation(a, b):
    """Pearson correlation of two images."""
    a, b = np.ravel(a) - np.mean(a), np.ravel(b) - np.mean(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mcc(a, b):
    """Matthews correlation coefficient of two binary images (the
    challenge's score)."""
    a, b = np.ravel(a) > 0.5, np.ravel(b) > 0.5
    tp, tn = float(np.sum(a & b)), float(np.sum(~a & ~b))
    fp, fn = float(np.sum(~a & b)), float(np.sum(a & ~b))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0


def main(output):
    """Runs the checks (see the module's description), writing to the folder
    ``output``."""
    data = HTC2022_PROCESSED_DATASET_PATH
    table = pd.read_csv(data.joinpath("samples.csv"))
    op = geometry().make_operator()
    rows, figures = [], []
    for _, sample in table.iterrows():
        folder = data.joinpath(sample["name"])
        sinogram = np.load(folder.joinpath("sinogram.npy"))
        target = np.load(folder.joinpath("reconstruction.npy"))
        target_segmentation = np.load(folder.joinpath("segmentation.npy"))
        row = dict(name=sample["name"], split=sample["split"])
        row["limited_difference"] = sample["limited_difference"]

        # 2. without the turn: the full scan as released
        released = np.roll(sinogram, int(sample["shift"]), axis=0)
        ours = fdk(torch.from_numpy(released)[None], op, clip=False)[0].cpu().numpy()
        ours_segmentation = segment(ours)
        theirs = np.load(folder.joinpath("released_reconstruction.npy"))
        theirs_segmentation = np.load(folder.joinpath("released_segmentation.npy"))
        scores = {k: correlation(ours, f(theirs)) for k, f in ORIENTATIONS.items()}
        best = max(scores, key=scores.get)
        row["orientation"] = best
        row["fbp_correlation"] = scores[best]
        row["fbp_correlation_as_is"] = scores["as is"]
        row["segmentation_mcc"] = mcc(
            ours_segmentation, ORIENTATIONS[best](theirs_segmentation)
        )

        # 3. the targets against the challenge's images turned the same way
        # (the sign of the turn: the one that fits best)
        angle = sample["shift"] * ANGLE_STEP
        candidates = {}
        for sign in (1, -1):
            turned = rotate(
                ORIENTATIONS[best](theirs), sign * angle, reshape=False, order=1
            )
            candidates[sign] = (correlation(target, turned), turned)
        sign = max(candidates, key=lambda s: candidates[s][0])
        row["target_correlation"] = candidates[sign][0]
        turned_segmentation = rotate(
            ORIENTATIONS[best](theirs_segmentation).astype(float),
            sign * angle,
            reshape=False,
            order=0,
        )
        row["target_segmentation_mcc"] = mcc(target_segmentation, turned_segmentation)

        # 4. the operator: the sinogram against the projection of its FBP
        projection = op(torch.from_numpy(target)[None].float()).cpu().numpy()[0]
        row["projection_correlation"] = correlation(sinogram, projection)

        rows.append(row)
        figures.append((sample["name"], turned_segmentation, target_segmentation))

    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output.joinpath("validation.csv"), index=False)
    pd.set_option("display.width", 200)
    print(result.round(4).to_string(index=False))
    print(
        "\nOrientations found:",
        result["orientation"].value_counts().to_dict(),
        "\nlimited_difference, largest:",
        result["limited_difference"].max(),
        "\nsegmentation MCC without the turn, lowest:",
        round(result["segmentation_mcc"].min(), 4),
        "\ntarget segmentation MCC, lowest:",
        round(result["target_segmentation_mcc"].min(), 4),
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(figures), 3, figsize=(7.5, 2.5 * len(figures)))
    for (name, theirs, ours), ax in zip(figures, np.atleast_2d(axes)):
        ax[0].imshow(theirs, cmap="gray")
        ax[1].imshow(ours, cmap="gray")
        ax[2].imshow(ours - (theirs > 0.5), cmap="bwr", vmin=-1, vmax=1)
        ax[0].set_ylabel(name)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
    for a, title in zip(
        np.atleast_2d(axes)[0], ("challenge's, turned", "target", "difference")
    ):
        a.set_title(title)
    fig.tight_layout()
    fig.savefig(output.joinpath("validation.png"), dpi=80)
    print(f"\nWritten to {output}")


if __name__ == "__main__":
    folder = (
        results_path("datasets/HTC2022_validation")
        if len(sys.argv) < 2
        else __import__("pathlib").Path(sys.argv[1])
    )
    main(folder)
