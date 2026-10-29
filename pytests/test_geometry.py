"""CT geometry: any two of shape, size and spacing, the value computed from them,
check_geometry, saving and loading (including files whose stored spacing
disagrees with their size and shape)."""

import copy
import json
import os
import pickle
import tempfile
import warnings

# the datasets' default folders are in LION_DATA_PATH
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np
import pytest

from LION.CTtools.ct_geometry import Geometry, check_geometry
from LION.data_loaders.deteCT import deteCT
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.operators.CTProjectionOp import CTProjectionOp
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


@pytest.mark.parametrize(
    "given, computed, expected",
    [
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
    ],
)
def test_any_two_values_give_the_third(given, computed, expected):
    geometry = Geometry(**given)
    np.testing.assert_allclose(getattr(geometry, computed), expected)
    assert computed not in vars(geometry)  # computed, not stored


def test_computed_shape_is_whole_pixels():
    geometry = Geometry(image_size=SIZE, voxel_size=SPACING)
    assert geometry.image_shape.dtype.kind == "i"
    assert geometry.image_shape.tolist() == SHAPE


def test_value_not_set_and_not_computable_is_none():
    assert Geometry(image_shape=SHAPE).voxel_size is None


def test_computed_value_follows_changes():
    """Changing a geometry's sizes or shapes changes its computed voxel and
    pixel sizes."""
    geometry = Geometry.default_parameters()
    geometry.image_size = [1.0, 1024.0, 1024.0]
    geometry.image_shape = [1, 1024, 1024]
    geometry.detector_size = [2.0, 1800.0]
    np.testing.assert_allclose(geometry.voxel_size, [1, 1, 1])
    np.testing.assert_allclose(geometry.pixel_size, [2, 2])


def test_setting_none_makes_a_value_computed_again():
    geometry = fan(
        image_shape=SHAPE,
        image_size=SIZE,
        voxel_size=SPACING,
        detector_shape=DET_SHAPE,
        detector_size=DET_SIZE,
    )
    geometry.image_shape = [1, 20, 40]
    with pytest.raises(ValueError, match="voxel_size"):
        check_geometry(geometry)
    geometry.voxel_size = None
    check_geometry(geometry)
    np.testing.assert_allclose(geometry.voxel_size, [2.0, 0.25, 1.0])


def test_values_are_arrays_however_they_are_set():
    geometry = Geometry.default_parameters()
    geometry.image_shape = [1, 64, 64]
    geometry.dso = 400
    assert isinstance(geometry.image_shape, np.ndarray)
    assert isinstance(geometry.dso, np.ndarray)


def test_init_again_forgets_earlier_values():
    geometry = Geometry(image_shape=SHAPE, voxel_size=SPACING)
    geometry.default_geo()
    assert "voxel_size" not in vars(geometry)
    np.testing.assert_allclose(
        geometry.voxel_size, Geometry.default_parameters().voxel_size
    )


def test_copies_keep_the_computed_value():
    geometry = Geometry.default_parameters()
    for copied in (copy.deepcopy(geometry), pickle.loads(pickle.dumps(geometry))):
        copied.image_shape = [1, 256, 256]
        np.testing.assert_allclose(copied.voxel_size, [300 / 512, 300 / 256, 300 / 256])
    assert not hasattr(geometry, "not_an_attribute")


# --------------------------------------------------------------------------
# Three values that disagree


def test_init_raises_when_three_values_disagree():
    with pytest.raises(ValueError, match="image_size"):
        Geometry(image_shape=SHAPE, image_size=SIZE, voxel_size=[1, 1, 1])
    with pytest.raises(ValueError, match="detector_size"):
        Geometry(detector_shape=DET_SHAPE, detector_size=DET_SIZE, pixel_size=[1, 1])


def test_init_accepts_three_values_that_agree():
    geometry = Geometry(image_shape=SHAPE, image_size=SIZE, voxel_size=SPACING)
    assert {"image_shape", "image_size", "voxel_size"} <= set(vars(geometry))


# --------------------------------------------------------------------------
# check_geometry


def test_valid_geometries_pass():
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


def test_empty_geometry_lists_every_problem():
    with pytest.raises(ValueError) as error:
        check_geometry(Geometry())
    message = str(error.value)
    for expected in (
        "mode",
        "image_shape, image_size, voxel_size",
        "detector_shape, detector_size, pixel_size",
        "angles",
    ):
        assert expected in message


def test_one_value_of_a_triple_is_not_enough():
    with pytest.raises(
        ValueError,
        match="two of image_shape, image_size, voxel_size are needed; set: image_shape",
    ):
        check_geometry(
            fan(image_shape=SHAPE, detector_shape=DET_SHAPE, detector_size=DET_SIZE)
        )


def test_shape_must_be_whole_pixels():
    geometry = fan(
        image_size=[1.0, 10.0, 10.0],
        voxel_size=[1.0, 3.0, 3.0],
        detector_shape=DET_SHAPE,
        detector_size=DET_SIZE,
    )
    with pytest.raises(ValueError, match="whole numbers"):
        check_geometry(geometry)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"mode": "cone"}, "mode"),
        ({"dsd": 100}, "dsd"),
        ({"dso": None}, "dso must be set"),
        ({"dso": -5}, "dso must be positive"),
        ({"angles": np.zeros((3, 2))}, "angles"),
        ({"angles": []}, "angles"),
        ({"image_pos": [0, 0]}, "image_pos"),
        ({"image_shape": [1, 10]}, "image_shape must be 3 numbers"),
        ({"detector_size": [1.0, np.nan]}, "detector_size"),
        ({"image_shape": [1, -10, 20]}, "positive"),
    ],
)
def test_invalid_values_are_reported(change, message):
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


def test_parallel_beam_needs_no_distances():
    geometry = Geometry(
        mode="parallel",
        image_shape=SHAPE,
        image_size=SIZE,
        detector_shape=DET_SHAPE,
        detector_size=DET_SIZE,
        angles=np.linspace(0, np.pi, 4),
    )
    check_geometry(geometry)


def test_operator_checks_the_geometry_first():
    """Before tomosipo is needed, so this runs without it."""
    with pytest.raises(ValueError, match="Invalid CT geometry"):
        CTProjectionOp(Geometry())


# --------------------------------------------------------------------------
# Saving and loading


def test_saving_checks_the_geometry(tmp_path):
    with pytest.raises(ValueError, match="Invalid CT geometry"):
        Geometry().save(tmp_path / "geo.json")


def test_saved_file_has_only_the_values_that_were_set(tmp_path):
    geometry = fan(
        image_shape=SHAPE,
        voxel_size=SPACING,
        detector_shape=DET_SHAPE,
        detector_size=DET_SIZE,
    )
    geometry.save(tmp_path / "geo.json")
    saved = json.loads((tmp_path / "geo.json").read_text())
    assert {"image_shape", "voxel_size", "detector_shape", "detector_size"} <= set(
        saved
    )
    assert "image_size" not in saved and "pixel_size" not in saved


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


def old_geometry_file_text():
    """A 2DeteCT geometry file with every value stored, and voxel and pixel
    sizes that disagree with its sizes and shapes."""
    d = {
        "mode": "fan",
        "image_size": [1, 1024, 1024],
        "image_shape": [1, 1024, 1024],
        "voxel_size": [0.5859375, 0.5859375, 0.5859375],
        "image_pos": [0, -1, -1],
        "detector_shape": [1, 956],
        "detector_size": [1.3146214844668342, 1256.7781391502936],
        "pixel_size": [1.0, 1.0],
        "dso": 3787.621241805204,
        "dsd": 4648.632398517645,
        "angles": np.linspace(0, 2 * np.pi, 10, endpoint=False).tolist(),
    }
    return json.dumps(d)


def old_geometry_file(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(old_geometry_file_text())
    return path


@pytest.mark.parametrize("how", ["Geometry.load", "init_from_parameter"])
def test_old_file_with_out_of_date_sizes_loads(tmp_path, how):
    path = old_geometry_file(tmp_path)
    with pytest.warns(UserWarning, match="computed again"):
        if how == "Geometry.load":
            loaded = Geometry()
            loaded.load(path)
        else:
            parameter = LIONParameter()
            parameter.load(path)
            loaded = Geometry.init_from_parameter(parameter)
    check_geometry(loaded)
    np.testing.assert_allclose(loaded.voxel_size, [1, 1, 1])
    np.testing.assert_allclose(loaded.pixel_size, [1.3146214844668342] * 2)


def test_old_file_with_consistent_sizes_loads_silently(tmp_path):
    geometry = Geometry.default_parameters()
    d = {k: np.asarray(v).tolist() for k, v in vars(geometry).items()}
    d["voxel_size"] = geometry.voxel_size.tolist()
    d["pixel_size"] = geometry.pixel_size.tolist()
    (tmp_path / "old.json").write_text(json.dumps(d))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = Geometry()
        loaded.load(tmp_path / "old.json")
    np.testing.assert_allclose(loaded.voxel_size, geometry.voxel_size)


def test_old_pickle_with_out_of_date_sizes_loads():
    """A pickled geometry (e.g. inside a model's .pt file) whose stored voxel
    size disagrees with its size and shape."""
    state = {
        k: np.array(v)
        for k, v in json.loads(old_geometry_file_text()).items()
        if k != "mode"
    }
    state["mode"] = "fan"
    old = Geometry.__new__(Geometry)
    old.__dict__.update(state)  # as unpickling an old geometry gives
    data = pickle.dumps(old)
    with pytest.warns(UserWarning, match="computed again"):
        loaded = pickle.loads(data)
    check_geometry(loaded)
    np.testing.assert_allclose(loaded.voxel_size, [1, 1, 1])


def test_detect_geometry_sizes():
    """The 2DeteCT geometry's voxel and pixel sizes come from its own sizes and
    shapes."""
    geometry = deteCT.get_default_geometry()
    np.testing.assert_allclose(geometry.voxel_size, [1, 1, 1])
    np.testing.assert_allclose(
        geometry.pixel_size, geometry.detector_size / geometry.detector_shape
    )
    assert geometry.pixel_size[1] == pytest.approx(1.3146214844668342)
