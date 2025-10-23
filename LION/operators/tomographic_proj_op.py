"""
This module defines a linear tomographic projection operator by wrapping
tomosipo's operator class.
"""

import tomosipo as ts

from LION.operators.operator import Operator


class TomographicProjOp(Operator):
    def __init__(self, ts_operator: ts.Operator.Operator):
        self._ts = ts_operator

    def __getattr__(self, name):
        return getattr(self._ts, name)

    @property
    def image_shape(self):
        return self._ts.domain_shape

    @property
    def data_shape(self):
        return self._ts.range_shape

    def forward(self, x):
        return self._ts._fp(x)

    def adjoint(self, y):
        return self._ts._bp(y)
