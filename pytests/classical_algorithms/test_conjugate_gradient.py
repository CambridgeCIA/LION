"""Tests for the conjugate gradient solver."""

import pytest
import torch

from LION.classical_algorithms.conjugate_gradient import conjugate_gradient


@pytest.fixture
def spd_system():
    torch.manual_seed(0)
    n = 100
    M = torch.randn(n, n, dtype=torch.float64)
    A = torch.eye(n, dtype=torch.float64) + M @ M.T / n
    b = torch.randn(n, dtype=torch.float64)
    return A, b


class Counter:
    def __init__(self, A):
        self.A, self.calls = A, 0

    def __call__(self, v):
        self.calls += 1
        return self.A @ v


@pytest.mark.parametrize("tol", [1e-2, 1e-6, 1e-10])
def test_stops_at_requested_relative_residual(spd_system, tol):
    A, b = spd_system
    x = conjugate_gradient(Counter(A), b, torch.zeros_like(b), max_iter=500, tol=tol)
    assert torch.norm(b - A @ x) / torch.norm(b) < tol


def test_looser_tolerance_needs_fewer_iterations(spd_system):
    A, b = spd_system
    loose, tight = Counter(A), Counter(A)
    conjugate_gradient(loose, b, torch.zeros_like(b), max_iter=500, tol=1e-2)
    conjugate_gradient(tight, b, torch.zeros_like(b), max_iter=500, tol=1e-10)
    assert loose.calls < tight.calls


def test_solution_matches_direct_solve(spd_system):
    A, b = spd_system
    x = conjugate_gradient(
        lambda v: A @ v, b, torch.zeros_like(b), max_iter=500, tol=1e-12
    )
    torch.testing.assert_close(x, torch.linalg.solve(A, b))


def test_zero_rhs_and_exact_start(spd_system):
    A, b = spd_system
    zero = torch.zeros_like(b)
    torch.testing.assert_close(
        conjugate_gradient(lambda v: A @ v, zero, b, 10, 1e-6), zero
    )
    exact = torch.linalg.solve(A, b)
    calls = Counter(A)
    conjugate_gradient(calls, b, exact, 10, 1e-6)
    assert calls.calls == 1  # only the initial residual
