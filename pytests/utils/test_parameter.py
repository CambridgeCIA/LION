"""Tests for LIONParameter and Geometry (de)serialisation."""

import pathlib

import numpy as np

from LION.CTtools.ct_geometry import Geometry
from LION.utils.parameter import LIONParameter


def test_serialize_does_not_modify_the_parameter():
    p = LIONParameter(a=1)
    p.inner = LIONParameter(b=2)
    p.path = pathlib.Path("some/folder")

    d = p.serialize()

    assert d == {"a": 1, "inner": {"b": 2}, "path": "some/folder"}
    assert isinstance(p.inner, LIONParameter)
    assert isinstance(p.path, pathlib.Path)


def test_save_load_round_trip(tmp_path):
    p = LIONParameter(a=1, inner={"b": [1, 2]})
    p.save(tmp_path / "params.json")
    q = LIONParameter()
    q.load(tmp_path / "params.json")
    assert q.a == 1
    assert isinstance(q.inner, LIONParameter) and q.inner.b == [1, 2]


def test_geometry_round_trip_keeps_image_pos(tmp_path):
    geo = Geometry.default_parameters()
    geo.image_pos = np.array([0.0, 10.0, -5.0])
    geo.save(tmp_path / "geo.json")

    loaded = LIONParameter()
    loaded.load(tmp_path / "geo.json")
    restored = Geometry.init_from_parameter(loaded)

    np.testing.assert_allclose(restored.image_pos, geo.image_pos)
    np.testing.assert_allclose(restored.angles, geo.angles)
    assert restored.mode == geo.mode
