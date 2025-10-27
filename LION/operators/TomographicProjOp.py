"""
This module defines a linear tomographic projection operator by wrapping
tomosipo's operator class.
"""

import tomosipo as ts
import torch

from LION.operators.Operator import Operator


class TomographicProjOp(Operator):
    def __init__(self, ts_operator: ts.Operator.Operator):
        self._ts = ts_operator

    def __getattr__(self, name):
        return getattr(self._ts, name)

    @property
    def domain_shape(self):
        return self._ts.domain_shape

    @property
    def range_shape(self):
        return self._ts.range_shape

    def forward(self, x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        return self._ts._fp(x, out=out)

    def adjoint(self, y: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        return self._ts._bp(y, out=out)
