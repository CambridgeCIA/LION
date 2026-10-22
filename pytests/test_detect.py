"""Regression tests for deteCT geometry-to-sinogram angle mapping."""

import os
import tempfile

os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np
import pytest

from LION.data_loaders.deteCT import deteCT


@pytest.fixture
def full_angles():
    return deteCT.get_default_geometry().angles


def test_sparse_angles_map_to_their_exact_raw_view_indices(full_angles):
    expected_indices = np.arange(0, len(full_angles), 72)
    requested_angles = full_angles[expected_indices]

    assert deteCT._get_angle_indices(full_angles, requested_angles) == list(
        expected_indices
    )


def test_angle_mapping_accepts_small_rounding_error(full_angles):
    expected_index = 1234
    requested_angle = full_angles[expected_index] + 5e-9

    assert deteCT._get_angle_indices(full_angles, [requested_angle]) == [expected_index]


def test_angle_mapping_rejects_angles_outside_the_native_grid(full_angles):
    non_native_angle = full_angles[1234] + 1e-5

    with pytest.raises(ValueError, match="not part of the existing angles"):
        deteCT._get_angle_indices(full_angles, [non_native_angle])
