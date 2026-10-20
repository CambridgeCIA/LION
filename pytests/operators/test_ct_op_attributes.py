"""CTProjectionOp attribute delegation, without tomosipo."""

import copy
import types

import pytest

from LION.operators import CTProjectionOp


def bare_op(ts_op=None):
    op = CTProjectionOp.__new__(CTProjectionOp)
    CTProjectionOp.__bases__[0].__init__(op)  # Operator/nn.Module init only
    if ts_op is not None:
        op._ts = ts_op
    return op


def test_missing_attribute_before_ts_is_set():
    op = CTProjectionOp.__new__(CTProjectionOp)  # nothing initialised
    assert not hasattr(op, "anything")  # AttributeError, not RecursionError


def test_delegates_to_tomosipo_operator():
    op = bare_op(types.SimpleNamespace(T="transpose", domain_shape=(1, 2, 2)))
    assert op.T == "transpose"
    with pytest.raises(AttributeError):
        op.not_there


def test_deepcopy():
    op = bare_op(types.SimpleNamespace(T="transpose"))
    assert copy.deepcopy(op).T == "transpose"
