"""FDK / FBP with tomosipo (needs astra with CUDA)."""

import numpy as np
import pytest
import torch

from LION.classical_algorithms import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator

pytestmark = pytest.mark.tomosipo


def fan_geometry():
    voxel = 300 / 64
    return Geometry(
        image_shape=[1, 64, 64],
        image_size=[voxel, 300, 300],
        detector_shape=[1, 128],
        detector_size=[voxel, 900],
        dso=575,
        dsd=1050,
        mode="fan",
        angles=np.linspace(0, 2 * np.pi, 90, endpoint=False),
    )


def parallel_geometry():
    geometry = Geometry.parallel_default_parameters(image_shape=[1, 64, 64])
    geometry.angles = np.linspace(0, np.pi, 90, endpoint=False)
    return geometry


GEOMETRIES = [fan_geometry(), parallel_geometry()]
IDS = ["fan", "parallel"]


def phantom(geometry):
    _, H, W = geometry.image_shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    disk = ((xx**2 + yy**2) < 0.5**2).float()
    return disk.reshape(1, 1, H, W).cuda()


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=IDS)
def test_reconstructs_both_geometries(geometry):
    x = phantom(geometry)
    op = make_operator(geometry)
    sino = op(x[0]).unsqueeze(0)
    recon = fdk(sino, op)
    assert recon.shape == x.shape
    assert torch.linalg.vector_norm(recon - x) / torch.linalg.vector_norm(x) < 0.3


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=IDS)
@pytest.mark.parametrize("clip", [False, True])
def test_differentiable_matches_default(geometry, clip):
    op = make_operator(geometry)
    sino = op(phantom(geometry)[0]).unsqueeze(0)
    sino = sino + 0.1 * torch.randn_like(sino)
    torch.testing.assert_close(
        fdk(sino, op, clip=clip, differentiable=True),
        fdk(sino, op, clip=clip),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=IDS)
def test_backward_is_the_adjoint(geometry):
    """FDK (unclipped) is linear, L(y); autograd must give L^T: <L y, x> = <y, L^T x>."""
    torch.manual_seed(0)
    op = make_operator(geometry)
    y = torch.randn(1, *op.range_shape, device="cuda", requires_grad=True)
    x = torch.randn(1, *op.domain_shape, device="cuda")
    Ly = fdk(y, op, clip=False, differentiable=True)
    (LTx,) = torch.autograd.grad(Ly, y, grad_outputs=x)
    lhs = torch.sum(Ly.detach() * x)
    rhs = torch.sum(y.detach() * LTx)
    assert LTx.abs().sum() > 0
    torch.testing.assert_close(lhs, rhs, rtol=1e-3, atol=1e-3)


def test_itnet_gradient_reaches_the_sinogram():
    from LION.models.iterative_unrolled.ItNet import ItNet

    geometry = fan_geometry()
    params = ItNet.default_parameters()
    params.n_iters, params.step_size = 2, [0.5, 0.5]
    params.Unet_params.baseDim, params.Unet_params.noGrp = 4, 2
    model = ItNet(geometry, params).cuda()
    op = make_operator(geometry)
    sino = op(phantom(geometry)[0]).unsqueeze(0).requires_grad_(True)
    (grad,) = torch.autograd.grad(model(sino).sum(), sino)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
