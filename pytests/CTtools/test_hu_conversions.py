"""Tests for HU conversions and forward_projection in LION.CTtools.ct_utils."""

import numpy as np
import pytest
import torch

from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import (
    forward_projection,
    from_HU_to_material_id,
    make_operator,
)

HU = [-1000, -950, -800, -750, -400, -150, -50, 0, 100, 150, 200, 300, 500]
IDS = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]


def test_material_id_numpy():
    img = np.array(HU, dtype=np.float32)
    original = img.copy()
    ids = from_HU_to_material_id(img)
    assert ids.dtype == np.uint8
    np.testing.assert_array_equal(ids, IDS)
    np.testing.assert_array_equal(img, original)  # input not modified


@pytest.mark.parametrize("dtype", [torch.float32, torch.int16])
def test_material_id_torch(dtype):
    img = torch.tensor(HU, dtype=dtype)
    ids = from_HU_to_material_id(img)
    assert ids.dtype == torch.uint8
    assert ids.tolist() == IDS


@pytest.mark.tomosipo
@pytest.mark.parametrize(
    "geometry",
    [Geometry.default_parameters(), Geometry.parallel_default_parameters([1, 64, 64])],
)
def test_forward_projection_matches_operator(geometry):
    image = torch.rand(*geometry.image_shape[1:])
    if torch.cuda.is_available():
        image = image.cuda()
    expected = make_operator(geometry)(image.unsqueeze(0))[0]
    torch.testing.assert_close(forward_projection(image, geometry), expected)
