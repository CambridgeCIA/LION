# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""CT geometry: any two of shape, size and spacing, the value computed from them,
check_geometry, saving and loading (including files whose stored spacing
disagrees with their size and shape)."""

import os
import tempfile

# the datasets' default folders are in LION_DATA_PATH
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np
import pytest

from LION.operators.ct.ct_geometry import Geometry, check_geometry
from LION.data_loaders.deteCT import deteCT
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.utils.parameter import LIONParameter

SHAPE, SIZE, SPACING = [1, 10, 20], [2.0, 5.0, 40.0], [2.0, 0.5, 2.0]
DET_SHAPE, DET_SIZE, DET_SPACING = [1, 30], [1.5, 60.0], [1.5, 2.0]


def fan(**image_and_detector):
    return Geometry(
        mode="fan",
        dso=500,
        dsd=1000,
        angles=np.linspace(0, 2 * np.pi, 8, endpoint=False),
        **image_and_detector,
    )


# --------------------------------------------------------------------------
# Any two values of a triple give the third


def test_any_two_values_give_the_third():
    for given, computed, expected in [
        ({"image_shape": SHAPE, "image_size": SIZE}, "voxel_size", SPACING),
        ({"image_shape": SHAPE, "voxel_size": SPACING}, "image_size", SIZE),
        ({"image_size": SIZE, "voxel_size": SPACING}, "image_shape", SHAPE),
        (
            {"detector_shape": DET_SHAPE, "detector_size": DET_SIZE},
            "pixel_size",
            DET_SPACING,
        ),
        (
            {"detector_shape": DET_SHAPE, "pixel_size": DET_SPACING},
            "detector_size",
            DET_SIZE,
        ),
        (
            {"detector_size": DET_SIZE, "pixel_size": DET_SPACING},
            "detector_shape",
            DET_SHAPE,
        ),
    ]:
        geometry = Geometry(**given)
        np.testing.assert_allclose(getattr(geometry, computed), expected)
        assert computed not in vars(geometry)  # computed, not stored

    # Changing a geometry's sizes or shapes changes its computed voxel and pixel
    # sizes.
    geometry = Geometry.default_parameters()
    geometry.image_size = [1.0, 1024.0, 1024.0]
    geometry.image_shape = [1, 1024, 1024]
    geometry.detector_size = [2.0, 1800.0]
    np.testing.assert_allclose(geometry.voxel_size, [1, 1, 1])
    np.testing.assert_allclose(geometry.pixel_size, [2, 2])


# --------------------------------------------------------------------------
# check_geometry


def test_invalid_values_are_reported():
    for change, message in [
        ({"mode": "cone"}, "mode"),
        ({"dso": None}, "dso must be set"),
        ({"angles": np.zeros((3, 2))}, "angles"),
        ({"image_shape": [1, 10]}, "image_shape must be 3 numbers"),
        ({"detector_size": [1.0, np.nan]}, "detector_size"),
        ({"image_shape": [1, -10, 20]}, "positive"),
    ]:
        geometry = fan(
            image_shape=SHAPE,
            image_size=SIZE,
            detector_shape=DET_SHAPE,
            detector_size=DET_SIZE,
        )
        for name, value in change.items():
            setattr(geometry, name, value)
        with pytest.raises(ValueError, match=message):
            check_geometry(geometry)

    check_geometry(
        fan(
            image_shape=SHAPE,
            voxel_size=SPACING,
            detector_size=DET_SIZE,
            pixel_size=DET_SPACING,
        )
    )
    for preset in (
        Geometry.default_parameters,
        Geometry.sparse_view_parameters,
        Geometry.sparse_angle_parameters,
        Geometry.parallel_default_parameters,
        Geometry.parallel_sparse_view_parameters,
        Geometry.parallel_sparse_angle_parameters,
        LIDC_IDRI.default_geometry,
        deteCT.get_default_geometry,
    ):
        check_geometry(preset())


# --------------------------------------------------------------------------
# Saving and loading


@pytest.mark.parametrize("how", ["Geometry.load", "init_from_parameter"])
def test_round_trip(tmp_path, how):
    geometry = fan(
        image_shape=SHAPE,
        voxel_size=SPACING,
        detector_size=DET_SIZE,
        pixel_size=DET_SPACING,
    )
    geometry.image_pos = [0.0, 1.0, -2.0]
    geometry.save(tmp_path / "geo.json")
    if how == "Geometry.load":
        loaded = Geometry()
        loaded.load(tmp_path / "geo.json")
    else:
        parameter = LIONParameter()
        parameter.load(tmp_path / "geo.json")
        loaded = Geometry.init_from_parameter(parameter)
    assert set(vars(loaded)) == set(vars(geometry))
    for name in (
        "image_shape",
        "image_size",
        "voxel_size",
        "detector_shape",
        "detector_size",
        "pixel_size",
        "image_pos",
        "angles",
        "dso",
        "dsd",
    ):
        np.testing.assert_allclose(getattr(loaded, name), getattr(geometry, name))
    assert loaded.mode == geometry.mode


def test_geometry_splits_along_its_angles():
    """A geometry splits into the geometries of some of its angles, and the
    sinogram into those angles' projections, batched or not."""
    import torch

    geometry = Geometry.default_parameters()
    n = len(geometry.angles)
    assert geometry.split_count == n
    indices = list(range(1, n, 4))
    sub = geometry.split(indices)
    np.testing.assert_array_equal(sub.angles, np.asarray(geometry.angles)[indices])
    assert len(geometry.angles) == n  # the geometry itself is not changed
    np.testing.assert_array_equal(sub.image_shape, geometry.image_shape)

    sino = torch.arange(float(n)).view(1, n, 1).expand(1, n, 3)  # value = angle
    for y in (sino, sino.unsqueeze(0)):
        sub_y = geometry.split_measurements(y, indices)
        assert sub_y.shape[-2:] == (len(indices), 3)
        assert sub_y[..., :, 0].flatten().tolist() == indices
