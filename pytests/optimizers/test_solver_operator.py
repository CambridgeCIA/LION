"""Solvers build the CT operator once."""

import pytest

import LION.optimizers.LIONsolver as lion_solver
from pytests.optimizers._helpers import make_solver

pytestmark = pytest.mark.filterwarnings("ignore")


def test_supervised_solver_builds_one_operator(monkeypatch):
    built = []
    monkeypatch.setattr(lion_solver, "make_operator", lambda g: built.append(g) or "op")
    solver = make_solver()
    assert built and len(built) == 1
    assert solver.op == "op"
