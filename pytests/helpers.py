"""Small models and operators shared by several test files."""

import torch

from LION.models.CNNs.dncnn import DnCNN
from LION.models.LIONmodel import ModelInputType
from LION.models.PnP.gradient_step_denoiser import GSD
from LION.models.PnP.gs_drunet import GSDRUNet
from LION.operators import Operator


class ImageMatrixOp(Operator):
    """y = A vec(x): a dense matrix as a LION operator, for tests without CT.

    Images have ``domain_shape``; measurements are vectors unless
    ``range_shape`` is given (e.g. a sinogram-shaped ``(1, angles, detectors)``).
    """

    def __init__(self, A, domain_shape, range_shape=None):
        super().__init__()
        self.A = A
        self._domain_shape = tuple(domain_shape)
        self._range_shape = (A.shape[0],) if range_shape is None else range_shape

    def __call__(self, x, out=None):
        return self.forward(x)

    def forward(self, x):
        return (self.A @ x.reshape(-1)).reshape(self._range_shape)

    def adjoint(self, y):
        return (self.A.T @ y.reshape(-1)).reshape(self._domain_shape)

    @property
    def domain_shape(self):
        return self._domain_shape

    @property
    def range_shape(self):
        return self._range_shape


def small_dncnn(input_type=ModelInputType.IMAGE):
    params = DnCNN.default_parameters()
    params.blocks = 3
    params.int_channels = 4
    params.model_input_type = input_type
    return DnCNN(params)


def small_gsd():
    params = GSD.default_parameters()
    params.backbone_params.int_channels = 4
    params.backbone_params.n_blocks = 1
    return GSD(params)


def small_gsdrunet():
    params = GSDRUNet.default_parameters()
    params.int_channels = 4
    params.n_blocks = 1
    return GSDRUNet(params)
