"""Fixtures for solver tests that run on CPU without tomosipo.

The solvers always build a CT operator from a geometry and run FDK for
image-domain models. Neither matters for these tests, so both are replaced:
the operator by ``None`` and FDK by the identity.
"""

import pytest

import LION.optimizers.LIONsolver as lion_solver
import LION.optimizers.SupervisedSolver as supervised_solver


@pytest.fixture(autouse=True)
def no_ct(monkeypatch):
    for module in (lion_solver, supervised_solver):
        monkeypatch.setattr(module, "make_operator", lambda geometry: None)
        monkeypatch.setattr(module, "fdk", lambda sino, op: sino)
