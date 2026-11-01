import json
import os
import tempfile

import h5py
import numpy as np
import foam_ct_phantom

# Activate foam_env
OUTPUT_DIR = "./Dataset_Hackathon/"
NUM_SAMPLES = 30
USE_CUDA = False  # True: foam_ct_phantom's CUDA projector for fan beam (needs a GPU)

# The geometry of the data, as LION's Geometry (LION.CTtools.ct_geometry) takes it,
# in mm. Any 2D fan- or parallel-beam geometry with a square image works:
# noise2inverse_foam.py reads it from OUTPUT_DIR/geometry.json.
GEOMETRY = dict(
    mode="fan",  # or "parallel" (dso and dsd are then not used)
    image_shape=[1, 512, 512],
    image_size=[300 / 512, 300, 300],
    detector_shape=[1, 900],
    detector_size=[1, 900],
    dso=575,
    dsd=1050,
    angles=np.linspace(0, 2 * np.pi, 360, endpoint=False).tolist(),
)

# The image spans 2.048 of foam_ct_phantom's length units (the phantom is a
# cylinder of radius 1). The foam's material has attenuation 1 per mm: the images
# are 0 (bubbles) and 1, and the sinograms are line integrals in mm.
IMAGE_WIDTH_IN_FOAM_UNITS = 2.048


def foam_geometries(geometry, cz):
    """foam_ct_phantom's volume and projection geometries for LION's
    ``geometry``, at height ``cz``, and how its projections map to LION's."""
    n = geometry["image_shape"][2]
    if (
        geometry["image_shape"][1] != n
        or geometry["image_size"][1] != geometry["image_size"][2]
    ):
        raise ValueError("The image must be square.")
    if geometry["detector_shape"][0] != 1:
        raise ValueError("The detector must have one row (2D data).")
    scale = IMAGE_WIDTH_IN_FOAM_UNITS / geometry["image_size"][2]  # foam units per mm
    volume = foam_ct_phantom.VolumeGeometry(
        n, n, 1, IMAGE_WIDTH_IN_FOAM_UNITS / n, 0.0, 0.0, cz
    )
    n_pixels = geometry["detector_shape"][1]
    pixel = geometry["detector_size"][1] / n_pixels * scale
    angles = np.asarray(geometry["angles"], dtype=float)
    # LION's angles in foam_ct_phantom's conventions: the same for parallel beam;
    # 3pi/2 - angle and the detector reversed for fan beam.
    if geometry["mode"] == "parallel":
        projections = foam_ct_phantom.ParallelGeometry(
            n_pixels, 1, angles, pixel, 0.0, cz
        )
        reverse_detector = False
    elif geometry["mode"] == "fan":
        projections = foam_ct_phantom.ConeGeometry(
            n_pixels,
            1,
            3 * np.pi / 2 - angles,
            pixel,
            sod=geometry["dso"] * scale,
            odd=(geometry["dsd"] - geometry["dso"]) * scale,
            zoff=cz,
            usecuda=USE_CUDA,
        )
        reverse_detector = True
    else:
        raise ValueError(f"mode must be 'fan' or 'parallel', got {geometry['mode']!r}")
    return volume, projections, reverse_detector, scale


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "geometry.json"), "w") as f:
    json.dump(GEOMETRY, f, indent=2)

phantom_file = "phantom_hackathon.h5"
if not os.path.exists(phantom_file):
    foam_ct_phantom.FoamPhantom.generate(phantom_file, seed=42, maxsize=0.1, zrange=1.0)

phantom = foam_ct_phantom.FoamPhantom(phantom_file)
with tempfile.TemporaryDirectory() as tmp:
    for i in range(NUM_SAMPLES):
        current_cz = 0.2 + (i * 0.02)
        volume, projections, reverse_detector, scale = foam_geometries(
            GEOMETRY, current_cz
        )

        phantom.generate_volume(os.path.join(tmp, "volume.h5"), volume)
        with h5py.File(os.path.join(tmp, "volume.h5"), "r") as f:
            image = f["volume"][0]
        phantom.generate_projections(os.path.join(tmp, "projs.h5"), projections)
        with h5py.File(os.path.join(tmp, "projs.h5"), "r") as f:
            sinogram = f["projs"][:, 0, :] / scale  # (angles, detector), in mm
        if reverse_detector:
            sinogram = sinogram[:, ::-1]

        with h5py.File(os.path.join(OUTPUT_DIR, f"gt_{i:04d}.h5"), "w") as f:
            f["image"] = image.astype(np.float32)
        with h5py.File(os.path.join(OUTPUT_DIR, f"sino_{i:04d}.h5"), "w") as f:
            f["sinogram"] = sinogram.astype(np.float32)
