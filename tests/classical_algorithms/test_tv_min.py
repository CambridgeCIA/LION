"""Tests for TV-minimization batch handling.

These tests verify that the TV-min wrapper correctly handles batched and
unbatched sinogram inputs, including proper per-sample indexing.

The bug: tv_min.py previously passed the full batched ``sino`` tensor to
ts_algorithms.tv_min2d instead of ``sino[i]``, causing a shape mismatch when
``B > 1`` (ts_algorithms expects a single sample, not a batch).
"""
import sys
from unittest.mock import MagicMock

sys.modules["tomosipo"] = MagicMock()
sys.modules["tomosipo.torch_support"] = MagicMock()
sys.modules["ts_algorithms"] = MagicMock()


class Geometry:
    pass


ct_geom = MagicMock()
ct_geom.Geometry = Geometry
sys.modules["LION.CTtools.ct_geometry"] = ct_geom

sys.modules["LION.CTtools.ct_utils"] = MagicMock()
sys.modules["LION.CTtools.ct_utils"].make_operator = lambda op: op

sys.modules["LION.operators"] = MagicMock()
sys.modules["LION.operators.CTProjectionOp"] = MagicMock()


class NoDataException(Exception):
    pass


exceptions = MagicMock()
exceptions.NoDataException = NoDataException
sys.modules["LION.exceptions"] = MagicMock()
sys.modules["LION.exceptions.exceptions"] = exceptions

import importlib.util
import torch

_spec = importlib.util.spec_from_file_location(
    "LION.classical_algorithms.tv_min",
    "LION/classical_algorithms/tv_min.py",
)
_tv_mod = importlib.util.module_from_spec(_spec)
sys.modules["LION.classical_algorithms.tv_min"] = _tv_mod
_spec.loader.exec_module(_tv_mod)

tv_min = _tv_mod.tv_min
ts_tv_min = _tv_mod.ts_tv_min

import pytest


class MockOp:
    def __init__(self, domain_shape=(1, 16, 16), range_shape=(1, 20, 20)):
        self.domain_shape = domain_shape
        self.range_shape = range_shape


@pytest.fixture(autouse=True)
def mock_ts_tv_min():
    """Replace the real ts_algorithms.tv_min2d with a deterministic mock."""
    _tv_mod.ts_tv_min = MagicMock()
    _tv_mod.ts_tv_min.side_effect = lambda op, y, *a, **kw: torch.zeros(
        op.domain_shape
    )
    yield
    _tv_mod.ts_tv_min = ts_tv_min


def test_tv_min_3d_input_unsqueezed(mock_ts_tv_min):
    """3D input (Z, A, D) should be unsqueezed to (1, Z, A, D)."""
    sino = torch.rand(1, 20, 20)
    op = MockOp()
    recon = tv_min(sino, op, lam=0.1, num_iterations=5)
    assert recon.shape == (1, 16, 16)
    args, _ = _tv_mod.ts_tv_min.call_args
    assert args[1].dim() == 3


def test_tv_min_4d_batch_calls_per_sample(mock_ts_tv_min):
    """B=2: ts_tv_min should be called twice with sino[0] and sino[1]."""
    B = 2
    sino = torch.rand(B, 1, 20, 20)
    op = MockOp()
    tv_min(sino, op, lam=0.1, num_iterations=5)
    assert _tv_mod.ts_tv_min.call_count == B
    args_0 = _tv_mod.ts_tv_min.call_args_list[0][0]
    args_1 = _tv_mod.ts_tv_min.call_args_list[1][0]
    assert args_0[1].shape == (1, 20, 20)
    assert args_1[1].shape == (1, 20, 20)


def test_tv_min_batch_different_inputs(mock_ts_tv_min):
    """Batch elements should not receive the same sino tensor reference."""
    sino = torch.rand(2, 1, 20, 20)
    op = MockOp()
    tv_min(sino, op, lam=0.1, num_iterations=5)
    args_0 = _tv_mod.ts_tv_min.call_args_list[0][0]
    args_1 = _tv_mod.ts_tv_min.call_args_list[1][0]
    assert args_0[1].data_ptr() != args_1[1].data_ptr()


def test_tv_min_batch_b1_output_shape(mock_ts_tv_min):
    """B=1 batch input should produce shape (1, *domain)."""
    op = MockOp()
    sino = torch.rand(1, 1, 20, 20)
    recon = tv_min(sino, op, lam=0.1, num_iterations=5)
    assert recon.shape == (1, 16, 16)


def test_tv_min_remove_batch(mock_ts_tv_min):
    """3D input should produce 3D output (remove_batch=True)."""
    op = MockOp()
    sino = torch.rand(1, 20, 20)
    recon = tv_min(sino, op, lam=0.1, num_iterations=5)
    assert recon.dim() == 3
    assert recon.shape == (1, 16, 16)


def test_tv_min_zero_batch_raises():
    """B=0 batch input should raise NoDataException."""
    op = MockOp()
    sino = torch.rand(0, 1, 20, 20)
    with pytest.raises(NoDataException):
        tv_min(sino, op, lam=0.1, num_iterations=5)


def test_tv_min_3d_zero_batch_raises():
    """3D input with dim 0 = 0 should raise NoDataException."""
    op = MockOp()
    sino = torch.rand(0, 20, 20)
    with pytest.raises(NoDataException):
        tv_min(sino, op, lam=0.1, num_iterations=5)
