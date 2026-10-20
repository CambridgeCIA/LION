"""ItNet step size handling (no CT operator needed)."""

import numpy as np
import pytest
import torch

from LION.CTtools.ct_geometry import Geometry
from LION.models.iterative_unrolled.ItNet import ItNet
from LION.models.LIONmodel import LIONmodel


@pytest.fixture(autouse=True)
def no_operator(monkeypatch):
    monkeypatch.setattr(LIONmodel, "_make_operator", lambda self: None)


def params(step_size, learned_step):
    p = ItNet.default_parameters()
    p.n_iters = 3
    p.step_size = step_size
    p.learned_step = learned_step
    p.Unet_params.baseDim = 4
    p.Unet_params.noGrp = 2
    return p


@pytest.mark.parametrize("learned_step", [True, False])
@pytest.mark.parametrize(
    "step_size, expected",
    [(0.5, [0.5, 0.5, 0.5]), ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]), (None, [1, 1, 1])],
)
def test_step_size_per_iteration(step_size, expected, learned_step):
    model = ItNet(Geometry.default_parameters(), params(step_size, learned_step))
    steps = [float(model.step_size[i].detach()) for i in range(3)]
    np.testing.assert_allclose(steps, expected, rtol=1e-6)


def test_wrong_number_of_steps():
    with pytest.raises(ValueError):
        ItNet(Geometry.default_parameters(), params([0.1, 0.2], True))


def test_fixed_steps_move_with_the_model():
    model = ItNet(Geometry.default_parameters(), params(0.5, False))
    assert "step_size" in dict(model.named_buffers())
    model.to(torch.float64)
    assert model.step_size.dtype == torch.float64
