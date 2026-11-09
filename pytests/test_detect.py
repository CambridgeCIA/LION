# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Shanglei Chai
# =============================================================================
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
