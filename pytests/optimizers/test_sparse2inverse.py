"""Sparse2Inverse sub-operators for any number of angular splits."""

import sys
import types

import numpy as np
import pytest
import torch

import LION.CTtools.ct_utils as ct_utils
import LION.optimizers.LIONsolver as lion_solver
from LION.CTtools.ct_geometry import Geometry
from LION.models.LIONmodel import LIONmodel
from LION.optimizers.Sparse2InverseSolver import Sparse2InverseSolver
from pytests.optimizers._helpers import small_dncnn

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def fake_ct(monkeypatch):
    """Operators are replaced by the angles they were built with."""
    fake = types.ModuleType("tomosipo.torch_support")
    fake.to_autograd = lambda op, **kwargs: op
    monkeypatch.setitem(sys.modules, "tomosipo.torch_support", fake)
    monkeypatch.setitem(sys.modules, "tomosipo", types.ModuleType("tomosipo"))
    monkeypatch.setattr(ct_utils, "make_operator", lambda g: np.asarray(g.angles))
    monkeypatch.setattr(lion_solver, "make_operator", lambda g: None)
    monkeypatch.setattr(LIONmodel, "_make_operator", lambda self: None)


def make_solver(splits, geometry):
    params = Sparse2InverseSolver.default_parameters()
    params.sino_split_count = splits
    model = small_dncnn()
    return Sparse2InverseSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.nn.MSELoss(),
        solver_params=params,
        geometry=geometry,
        device=torch.device("cpu"),
        verbose=False,
    )


@pytest.mark.parametrize("splits", [3, 4, 5])
def test_projectors_for_every_target_set(fake_ct, splits):
    geometry = Geometry.default_parameters()  # 360 angles
    solver = make_solver(splits, geometry)
    angles = np.asarray(geometry.angles)
    assert len(solver.sub_ops) == splits
    for i, sub_op in enumerate(solver.sub_ops):
        np.testing.assert_array_equal(sub_op, angles[i::splits])

    for combo in solver.split_combinations:
        target = tuple(s for s in range(splits) if s not in combo)
        projector_angles = solver.combo_ops_autograd[target]
        expected = sorted(i for s in target for i in range(s, len(angles), splits))
        np.testing.assert_array_equal(projector_angles, angles[expected])
    # the solver's own geometry is not modified
    assert len(solver.geometry.angles) == len(angles)


def test_needs_at_least_three_splits(fake_ct):
    with pytest.raises(ValueError, match="sino_split_count"):
        make_solver(2, Geometry.default_parameters())
