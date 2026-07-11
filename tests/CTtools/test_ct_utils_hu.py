import numpy as np
import torch

from LION.CTtools import ct_utils as ct


def test_normal_hu_conversion_round_trips_torch_values():
    hu = torch.tensor((-1000.0, -500.0, 0.0, 1000.0, 2000.0))

    recovered = ct.from_normal_to_HU(ct.from_HU_to_normal(hu))

    torch.testing.assert_close(recovered, hu)


def test_normal_hu_conversion_round_trips_numpy_values():
    hu = np.asarray((-1000.0, -500.0, 0.0, 1000.0, 2000.0))

    recovered = ct.from_normal_to_HU(ct.from_HU_to_normal(hu))

    np.testing.assert_allclose(recovered, hu)


def test_normal_to_hu_does_not_hide_reconstruction_overshoot():
    normal = torch.tensor((-0.1, 1.1))

    converted = ct.from_normal_to_HU(normal)

    torch.testing.assert_close(converted, torch.tensor((-1300.0, 2300.0)))
