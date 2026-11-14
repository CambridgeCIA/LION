# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Prepares the downloaded HTC2022 data (``download.py``) for LION's
``HTC2022`` loader, into ``LION_DATA_PATH/processed/HTC2022``:

    python LION/data_loaders/HTC2022/pre_process_HTC2022.py

For each phantom (a folder named after it, e.g. ``ta`` or ``7c``):

- ``sinogram.npy`` (720, 560): its full scan (0 to 359.5 degrees; the files'
  last angle, 360, repeats the first), turned so that the test phantoms'
  challenge arc starts at the first angle: the rows shifted cyclically by the
  arc's first angle (an exact rotation of the phantom, as the scan is a full
  circle in 0.5 degree steps);
- ``reconstruction.npy`` (512, 512): the FBP of that sinogram (LION's FDK, not
  clipped), and ``segmentation.npy``, its segmentation by the challenge's
  recipe (``HTC2022.segment``): the targets;
- for comparison (``validate_HTC2022.py``), the challenge's files as
  released, not turned: ``released_reconstruction.npy`` and
  ``released_segmentation.npy``, and for the test phantoms
  ``limited_sinogram.npy`` and ``limited_angles.npy`` (degrees).

``samples.csv`` lists them: ``name``, ``split`` ("train": the challenge's 5
training phantoms; "test": its 21), ``level`` (1 to 7; 0 for training),
``arc`` (the challenge's arc, degrees), ``start_angle`` (degrees) and
``shift`` (rows) of the turn, and ``limited_difference``: the largest
difference between the challenge's limited sinogram and the turned full one's
first rows (0 if they are the same data).

Needs tomosipo and a GPU (for the FBP).
"""

import re

import numpy as np
import pandas as pd
import scipy.io
import torch

from LION.data_loaders.HTC2022 import (
    ANGLE_STEP,
    FULL_ANGLES,
    geometry,
    segment,
)
from LION.operators.ct.fdk import fdk
from LION.utils.paths import HTC2022_PATH, HTC2022_PROCESSED_DATASET_PATH


def read_mat(file):
    """The single variable of a .mat file (MATLAB v5, or v7.3 with h5py)."""
    try:
        contents = scipy.io.loadmat(file, struct_as_record=False, squeeze_me=True)
        names = [k for k in contents if not k.startswith("__")]
        return contents[names[0]]
    except NotImplementedError:  # v7.3: HDF5
        import h5py

        with h5py.File(file, "r") as f:
            name = [k for k in f.keys() if not k.startswith("#")][0]
            return _h5_to_python(f[name])


def _h5_to_python(item):
    import h5py

    if isinstance(item, h5py.Dataset):
        return np.array(item).T  # MATLAB's column-major order
    return type("Struct", (), {k: _h5_to_python(v) for k, v in item.items()})


def image_of(file):
    """The image (2D array) a reconstruction or segmentation file holds."""
    value = read_mat(file)
    if isinstance(value, np.ndarray):
        return value
    arrays = [
        getattr(value, f)
        for f in getattr(value, "_fieldnames", vars(value))
        if isinstance(getattr(value, f), np.ndarray) and getattr(value, f).ndim == 2
    ]
    return max(arrays, key=np.size)


def scan_of(file):
    """(sinogram, angles in degrees) of a full or limited scan file."""
    scan = read_mat(file)
    return (
        np.asarray(scan.sinogram, dtype=np.float64),
        np.atleast_1d(np.asarray(scan.parameters.angles, dtype=np.float64)),
    )


def main():
    """Prepares every phantom (see the module's description)."""
    out = HTC2022_PROCESSED_DATASET_PATH
    out.mkdir(parents=True, exist_ok=True)
    op = geometry().make_operator()
    rows = []
    for full_file in sorted(HTC2022_PATH.rglob("htc2022_*_full.mat")):
        name = full_file.name[len("htc2022_") : -len("_full.mat")]
        folder = full_file.parent
        split = (
            "test" if "test" in str(full_file.relative_to(HTC2022_PATH)) else "train"
        )
        print(f"{name} ({split})")

        full, angles = scan_of(full_file)
        expected = np.arange(FULL_ANGLES + 1) * ANGLE_STEP
        if len(angles) != len(expected) or not np.allclose(angles, expected):
            raise ValueError(f"{full_file}: unexpected angles {angles[:3]}...")
        full = full[:FULL_ANGLES]

        limited_file = folder.joinpath(f"htc2022_{name}_limited.mat")
        shift, start, arc, difference = 0, 0.0, np.nan, np.nan
        sample = out.joinpath(name)
        sample.mkdir(exist_ok=True)
        if limited_file.exists():
            limited, limited_angles = scan_of(limited_file)
            start = float(limited_angles[0])
            arc = (len(limited_angles) - 1) * ANGLE_STEP
            shift = int(round(start / ANGLE_STEP)) % FULL_ANGLES
            np.save(sample.joinpath("limited_sinogram.npy"), limited.astype(np.float32))
            np.save(sample.joinpath("limited_angles.npy"), limited_angles)
        aligned = np.roll(full, -shift, axis=0)
        if limited_file.exists():
            difference = float(np.abs(limited - aligned[: len(limited)]).max())

        sinogram = torch.from_numpy(aligned.astype(np.float32))[None]
        reconstruction = fdk(sinogram, op, clip=False)[0].cpu().numpy()
        np.save(sample.joinpath("sinogram.npy"), aligned.astype(np.float32))
        np.save(sample.joinpath("reconstruction.npy"), reconstruction)
        np.save(sample.joinpath("segmentation.npy"), segment(reconstruction))
        for kind, suffix in (
            ("reconstruction", "_full_recon_fbp.mat"),
            ("segmentation", "_full_recon_fbp_seg.mat"),
        ):
            released = folder.joinpath(f"htc2022_{name}{suffix}")
            if released.exists():
                np.save(sample.joinpath(f"released_{kind}.npy"), image_of(released))

        level = re.match(r"0?(\d)[a-c]$", name)
        rows.append(
            dict(
                name=name,
                split=split,
                level=int(level.group(1)) if level and split == "test" else 0,
                arc=arc,
                start_angle=start,
                shift=shift,
                limited_difference=difference,
            )
        )
    table = pd.DataFrame(rows).sort_values(
        ["split", "level", "name"], ascending=[False, True, True]
    )
    table.to_csv(out.joinpath("samples.csv"), index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
