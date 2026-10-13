"""Tests for the sHSI reference measurement format."""

import numpy as np
import pytest
from LION.data_loaders.shsi.data import load_shsi_npz


def test_load_shsi_npz_preserves_shapes_and_values(tmp_path):
    measurement = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    mask = (measurement % 3 == 0).astype(np.uint8)
    ground_truth = measurement + 1
    path = tmp_path / "input.npz"
    np.savez(path, measurement=measurement, mask=mask, sigma=0.03, gt=ground_truth)

    actual = load_shsi_npz(path)

    np.testing.assert_array_equal(actual.measurement, measurement)
    np.testing.assert_array_equal(actual.mask, mask)
    np.testing.assert_array_equal(actual.ground_truth, ground_truth)
    assert actual.mask.dtype == np.bool_
    assert actual.sigma == 0.03


def test_load_shsi_npz_without_ground_truth(tmp_path):
    path = tmp_path / "input.npz"
    np.savez(path, measurement=np.ones((4, 7)), mask=np.ones((4, 7)), sigma=0.1)

    actual = load_shsi_npz(path)

    assert actual.ground_truth is None
    assert actual.measurement.shape == (4, 7)


@pytest.mark.parametrize(
    ("measurement", "mask", "sigma", "message"),
    [
        (np.ones((2, 3)), np.ones((3, 2)), 1.0, "same shape"),
        (np.ones((2, 3)), np.ones((2, 3)), 0.0, "positive"),
        (np.ones((2, 3)), np.ones((2, 3)) * 0.5, 1.0, "0 or 1"),
    ],
)
def test_load_shsi_npz_rejects_invalid_data(
    tmp_path, measurement, mask, sigma, message
):
    path = tmp_path / "bad.npz"
    np.savez(path, measurement=measurement, mask=mask, sigma=sigma)

    with pytest.raises(ValueError, match=message):
        load_shsi_npz(path)


def test_load_shsi_npz_requires_keys(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez(path, measurement=np.zeros((2, 3)))

    with pytest.raises(ValueError, match="Missing sHSI NPZ keys"):
        load_shsi_npz(path)
