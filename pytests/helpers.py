"""Small models and operators shared by several test files."""

import torch

from LION.models.CNNs.dncnn import DnCNN
from LION.models.LIONmodel import ModelInputType
from LION.models.PnP.gradient_step_denoiser import GSD
from LION.models.torch_model import TorchModel
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

    def default_reconstruction(self, y):  # the adjoint, per sample
        if y.dim() == len(self._range_shape) + 1:
            return torch.stack([self.adjoint(s) for s in y])
        return self.adjoint(y)

    @property
    def domain_shape(self):
        return self._domain_shape

    @property
    def range_shape(self):
        return self._range_shape


class ScaleOp(Operator):
    """y = factor * x on images of ``shape``: an operator that keeps its
    arguments, so it can be written as a spec (unlike ImageMatrixOp's matrix)."""

    def __init__(self, factor, shape):
        super().__init__()
        self.factor = factor
        self.shape = tuple(shape)

    def __call__(self, x, out=None):
        return self.forward(x)

    def forward(self, x):
        return self.factor * x

    def adjoint(self, y):
        return self.factor * y

    def default_reconstruction(self, y):
        return self.adjoint(y)

    @property
    def domain_shape(self):
        return self.shape

    @property
    def range_shape(self):
        return self.shape


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


def small_gsd_dncnn():
    """GSD with a denoiser that takes no noise level."""
    return GSD(denoiser=small_dncnn())


class NoiseLevelConvolution(torch.nn.Module):
    """A plain torch denoiser that takes the noise level, as an extra input
    channel (as DRUNet does)."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, x, noise_level):
        return self.conv(torch.cat((x, noise_level * torch.ones_like(x)), dim=1))


def small_gsd_noise_channel():
    """GSD whose denoiser takes the noise level."""
    return GSD(denoiser=TorchModel(NoiseLevelConvolution))


def replace_fdk(monkeypatch, stand_in):
    """Make LION's ``fdk`` compute ``stand_in(sino, op)``, e.g. without the CT
    backend. Models keep ``fdk`` itself in their parameters (their initial
    reconstruction), so its inside is replaced, not the name."""
    import importlib

    fdk_module = importlib.import_module("LION.classical_algorithms.fdk")
    monkeypatch.setattr(
        fdk_module, "batched_reconstruction", lambda fn, sino, op: stand_in(sino, op)
    )


class CTStandIn:
    """A stand-in CT operator, for tests without tomosipo: its default
    reconstruction is LION's ``fdk`` (made a stand-in with ``replace_fdk``)."""

    def default_reconstruction(self, y):
        from LION.classical_algorithms.fdk import fdk

        return fdk(y, self)
