# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  :
# =============================================================================
"""Small models and operators shared by several test files."""

import torch

from LION.models.CNNs.dncnn import DnCNN
from LION.models.LIONmodel import ModelInputType
from LION.models.PnP.gradient_step_denoiser import GSD
from LION.models.torch_model import TorchModel
from LION.operators import Operator


def adjoint_reconstruction(y, op):
    """The adjoint of the measurements, per sample if batched: the default
    reconstruction of the stand-in operators here."""
    if y.dim() == len(op.range_shape) + 1:
        return torch.stack([op.adjoint(s) for s in y])
    return op.adjoint(y)


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

    def default_reconstruction_fn(self):  # the adjoint, per sample
        return adjoint_reconstruction

    def autograd(self):  # plain torch: torch differentiates it as it is
        return self, self.adjoint

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

    def default_reconstruction_fn(self):
        return adjoint_reconstruction

    def autograd(self):  # plain torch: torch differentiates it as it is
        return self, self.adjoint

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

    fdk_module = importlib.import_module("LION.operators.ct.fdk")
    monkeypatch.setattr(
        fdk_module, "batched_reconstruction", lambda fn, sino, op: stand_in(sino, op)
    )


class CTStandIn:
    """A stand-in CT operator, for tests without tomosipo: its default
    reconstruction is LION's ``fdk`` (made a stand-in with ``replace_fdk``),
    with the given domain and range shapes."""

    def __init__(self, domain_shape=None, range_shape=None):
        self.domain_shape = domain_shape
        self.range_shape = range_shape

    def default_reconstruction_fn(self):
        from LION.operators.ct.fdk import fdk

        return fdk

    def default_reconstruction(self, y):
        return self.default_reconstruction_fn()(y, self)


class IdentityStandIn(Operator):
    """A stand-in operator that is the identity on ``shape``, with autograd
    (for solvers whose gradients go through the operator), without CT."""

    def __init__(self, shape=(1, 16, 16)):
        super().__init__()
        self.shape = tuple(shape)

    def __call__(self, x, out=None):
        return x

    def forward(self, x):
        return x

    def adjoint(self, y):
        return y

    def autograd(self):
        return self, self.adjoint

    def default_reconstruction_fn(self):
        return identity_reconstruction

    @property
    def domain_shape(self):
        return self.shape

    @property
    def range_shape(self):
        return self.shape


def identity_reconstruction(y, op):
    """The measurements as they are: IdentityStandIn's default reconstruction."""
    return y


def dotproduct_adjointness_test(
    operator: Operator,
    u: torch.Tensor,
    v: torch.Tensor,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-5,
) -> None:
    """Test the adjointness of linear operator and operator.H.

    Test if
         <Operator(u),v> == <u, Operator^H(v)>
         for one u ∈ domain and one v ∈ range of Operator.
    and if the shapes match.

    Note: This property should hold for all u and v.
    Commonly, this function is called with two random vectors u and v.

    Parameters
    ----------
    operator
        linear operator
    u
        element of the domain of the operator
    v
        element of the range of the operator
    relative_tolerance
        default is pytorch's default for float16
    absolute_tolerance
        default is pytorch's default for float16

    Raises
    ------
    `AssertionError`
        if the adjointness property does not hold
    `AssertionError`
        if the shape of operator(u) and v does not match
        if the shape of u and operator.H(v) does not match

    """
    forward_u = operator(u)
    adjoint_v = operator.adjoint(v)
    if not isinstance(forward_u, torch.Tensor):
        forward_u = torch.as_tensor(forward_u)
    if not isinstance(adjoint_v, torch.Tensor):
        adjoint_v = torch.as_tensor(adjoint_v)

    # explicitly check the shapes, as flatten makes the dot product insensitive to wrong shapes
    assert forward_u.shape == v.shape
    assert adjoint_v.shape == u.shape

    dotproduct_range = torch.vdot(forward_u.flatten(), v.flatten())
    dotproduct_domain = torch.vdot(u.flatten().flatten(), adjoint_v.flatten())
    torch.testing.assert_close(
        dotproduct_range,
        dotproduct_domain,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
