"""Sparse2Inverse sub-operators for any number of angular splits."""

import itertools
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


def make_solver(splits, geometry, inputs=None, recon_fn=None):
    params = Sparse2InverseSolver.default_parameters()
    params.sino_split_count = splits
    if inputs is None:
        del params.input_split_count  # parameter files from before it existed
    else:
        params.input_split_count = inputs
    if recon_fn is not None:
        params.recon_fn = recon_fn
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


def test_defaults_match_the_paper_setup():
    params = Sparse2InverseSolver.default_parameters()
    assert (params.sino_split_count, params.input_split_count) == (4, 2)


@pytest.mark.parametrize(
    "splits, inputs", [(3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (5, 2), (5, 3)]
)
def test_inputs_and_target_projectors(fake_ct, splits, inputs):
    geometry = Geometry.default_parameters()  # 360 angles
    solver = make_solver(splits, geometry, inputs)
    angles = np.asarray(geometry.angles)

    # the single input sinogram is split into interleaved angle subsets
    assert len(solver.sub_ops) == splits
    for i, sub_op in enumerate(solver.sub_ops):
        np.testing.assert_array_equal(sub_op, angles[i::splits])

    # every combination of `inputs` splits is a network input...
    assert solver.split_combinations == list(
        itertools.combinations(range(splits), inputs)
    )
    # ...and the target is the projection onto all the remaining angles
    for combo in solver.split_combinations:
        target = tuple(s for s in range(splits) if s not in combo)
        expected = sorted(i for s in target for i in range(s, len(angles), splits))
        np.testing.assert_array_equal(
            solver.combo_ops_autograd[target], angles[expected]
        )
    # the solver's own geometry is not modified
    assert len(solver.geometry.angles) == len(angles)


def test_old_parameter_files_use_pairs(fake_ct):
    solver = make_solver(4, Geometry.default_parameters(), inputs=None)
    assert solver.input_split_count == 2
    assert solver.split_combinations == solver.two_two_strategy(4)


@pytest.mark.parametrize("splits, inputs", [(2, 2), (4, 4), (4, 0), (1, 1)])
def test_at_least_one_split_left_as_target(fake_ct, splits, inputs):
    with pytest.raises(ValueError, match="input_split_count"):
        make_solver(splits, Geometry.default_parameters(), inputs)


@pytest.mark.parametrize("splits, inputs", [(4, 2), (5, 3)])
def test_training_step_uses_each_combination(fake_ct, monkeypatch, splits, inputs):
    """One mini-batch: p sub-reconstructions per combination, averaged, then
    compared with the held-out projections."""
    # projectors return a constant sinogram over their angles, depending on the image
    monkeypatch.setitem(
        sys.modules["tomosipo.torch_support"].__dict__,
        "to_autograd",
        lambda op, **kw: (lambda img: img.mean() * torch.ones(1, 1, len(op), 8)),
    )
    calls = []

    def recon_fn(sub_sino, op):
        calls.append(sub_sino.shape[2])
        return sub_sino.mean() * torch.ones(sub_sino.shape[0], 1, 8, 8)

    geometry = Geometry.default_parameters()  # 360 angles
    solver = make_solver(splits, geometry, inputs, recon_fn=recon_fn)
    sinos = torch.rand(2, 1, 360, 8)
    loss = solver.mini_batch_step(sinos, None)
    assert torch.isfinite(loss) and loss > 0
    n_combinations = len(list(itertools.combinations(range(splits), inputs)))
    assert len(calls) == n_combinations * inputs
    assert all(n == 360 // splits for n in calls)
