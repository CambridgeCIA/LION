"""CT tools, the CT projection operator and FDK/FBP.

Tests marked ``tomosipo`` need tomosipo/astra with CUDA and are skipped otherwise.
"""

import copy
import types

import numpy as np
import pytest
import torch
from LION.classical_algorithms.fdk import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import (
    forward_projection,
    from_HU_to_material_id,
    make_operator,
    sinogram_add_noise,
)
from LION.operators.CTProjectionOp import CTProjectionOp

from pytests.helper import dotproduct_adjointness_test

try:
    from tomosipo.torch_support import to_autograd
except Exception:  # the tests using it are skipped by the marker in that case
    to_autograd = None


# --------------------------------------------------------------------------
# Noise


@pytest.fixture
def clean_sino():
    torch.manual_seed(0)
    return 2.0 * torch.rand(1, 30, 40)


def test_sinogram_add_noise_on_cpu_tensor(clean_sino):
    noisy = sinogram_add_noise(clean_sino, I0=1000, sigma=5)
    assert noisy.device == clean_sino.device
    assert noisy.shape == clean_sino.shape
    assert torch.isfinite(noisy).all() and (noisy >= 0).all()
    assert not torch.equal(noisy, clean_sino)


def test_sinogram_add_noise_numpy_in_numpy_out(clean_sino):
    noisy = sinogram_add_noise(clean_sino.numpy(), I0=1000, sigma=5)
    assert isinstance(noisy, np.ndarray)
    assert noisy.shape == tuple(clean_sino.shape)


def test_less_noise_with_more_photons(clean_sino):
    torch.manual_seed(1)
    low = sinogram_add_noise(clean_sino, I0=1e3, sigma=0)
    high = sinogram_add_noise(clean_sino, I0=1e6, sigma=0)
    assert (high - clean_sino).abs().mean() < (low - clean_sino).abs().mean()


@pytest.fixture
def deterministic_noise(monkeypatch):
    """No random counts: Poisson returns its mean, electronic noise is zero,
    so only the cross-talk blur and the scaling of the line integrals remain."""
    monkeypatch.setattr(torch, "poisson", lambda x: x.clone())
    monkeypatch.setattr(torch, "randn", lambda *shape, **kw: torch.zeros(*shape, **kw))


@pytest.mark.parametrize("legacy", [False, True])
def test_crosstalk_is_along_the_detector(deterministic_noise, legacy):
    sino = torch.zeros(1, 9, 9)
    sino[0, 4, 4] = 1.0  # one bright detector pixel in one projection
    noisy = sinogram_add_noise(sino, I0=1e4, sigma=0, sigma_blur=1.0, legacy=legacy)
    assert noisy[0, 4, 3] > 1e-3  # neighbouring detector pixel, same angle
    other_angle = noisy[0, 3, 4]
    if legacy:  # the old model also blurred into the neighbouring angle
        assert other_angle > 1e-3
    else:
        assert other_angle < 1e-5  # (float32 rounding of log(exp(0)))


@pytest.mark.parametrize("legacy", [False, True])
def test_noise_of_a_sample_does_not_depend_on_its_batch(deterministic_noise, legacy):
    torch.manual_seed(0)
    a, b = torch.rand(1, 20, 30), 5 * torch.rand(1, 20, 30)  # b has a larger maximum
    alone = sinogram_add_noise(a.unsqueeze(0), I0=1e3, sigma=0, legacy=legacy)[0]
    in_batch = sinogram_add_noise(torch.stack([a, b]), I0=1e3, sigma=0, legacy=legacy)[
        0
    ]
    if legacy:  # scaled by the maximum of the whole batch
        assert not torch.allclose(alone, in_batch)
    else:
        torch.testing.assert_close(alone, in_batch)


# --------------------------------------------------------------------------
# HU conversions

HU = [-1000, -950, -800, -750, -400, -150, -50, 0, 100, 150, 200, 300, 500]
MATERIAL_IDS = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]


def test_material_id_numpy():
    img = np.array(HU, dtype=np.float32)
    original = img.copy()
    ids = from_HU_to_material_id(img)
    assert ids.dtype == np.uint8
    np.testing.assert_array_equal(ids, MATERIAL_IDS)
    np.testing.assert_array_equal(img, original)  # input not modified


@pytest.mark.parametrize("dtype", [torch.float32, torch.int16])
def test_material_id_torch(dtype):
    img = torch.tensor(HU, dtype=dtype)
    ids = from_HU_to_material_id(img)
    assert ids.dtype == torch.uint8
    assert ids.tolist() == MATERIAL_IDS


# --------------------------------------------------------------------------
# CTProjectionOp


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


@pytest.mark.tomosipo
@pytest.mark.parametrize(
    "geometry",
    [Geometry.default_parameters(), Geometry.parallel_default_parameters([1, 64, 64])],
)
def test_forward_projection_matches_operator(geometry):
    image = torch.rand(*geometry.image_shape[1:])
    if torch.cuda.is_available():
        image = image.cuda()
    expected = make_operator(geometry)(image.unsqueeze(0))[0]
    torch.testing.assert_close(forward_projection(image, geometry), expected)


@pytest.mark.tomosipo
def test_ct_autograd_op_forward_and_backward():
    """Check that to_autograd wraps the CT operator correctly and supports backprop."""
    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)
    autograd_operator = to_autograd(operator)

    torch.manual_seed(0)
    input_tensor = torch.randn(*geometry.image_shape, requires_grad=True)

    output_tensor = autograd_operator(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.requires_grad
    assert input_tensor.grad is None

    output_tensor.mean().backward()

    assert input_tensor.grad is not None
    assert isinstance(input_tensor.grad, torch.Tensor)
    assert input_tensor.grad.shape == input_tensor.shape
    assert torch.isfinite(input_tensor.grad).all()


@pytest.mark.tomosipo
def test_ct_autograd_op_matches_original_operator():
    """Check that the autograd wrapper produces the same output as the original operator."""
    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)
    autograd_operator = to_autograd(operator)

    torch.manual_seed(1)
    input_tensor = torch.randn(*geometry.image_shape, requires_grad=True)

    input_np = input_tensor.detach().cpu().numpy()
    output_np = operator(input_np)
    output_autograd = autograd_operator(input_tensor).detach().cpu().numpy()

    np.testing.assert_allclose(output_autograd, output_np, rtol=1e-5, atol=1e-5)


@pytest.mark.tomosipo
def test_original_op_does_not_propagate_grad():
    """Check that the original operator output does not require gradients and backward fails."""
    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)

    torch.manual_seed(2)
    input_tensor = torch.randn(*geometry.image_shape, requires_grad=True)

    # Original operator is not autograd aware
    output_tensor = operator(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.requires_grad is False

    # Backprop should fail with the expected autograd error
    with pytest.raises(
        RuntimeError, match="does not require grad and does not have a grad_fn"
    ):
        output_tensor.mean().backward()


@pytest.mark.tomosipo
def test_ct_op_adjointness():
    """Test CT operator adjoint property."""
    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)

    # Check the default operator shapes
    assert operator.domain_shape == (1, 512, 512)
    assert operator.range_shape == (1, 360, 900)

    test_volume = torch.rand(*operator.domain_shape)
    test_projection = torch.rand(*operator.range_shape)
    # Note: tomosipo implementation uses an accelerated method that is not
    # exactly adjoint, so we use looser tolerances here.
    dotproduct_adjointness_test(
        operator,
        test_volume,
        test_projection,
        relative_tolerance=1e-1,
        absolute_tolerance=1e-1,
    )


@pytest.mark.tomosipo
def test_ct_op_backward_compatibility_with_tomosipo():
    geometry = Geometry.default_parameters()
    operator = make_operator(geometry=geometry)

    assert operator.domain_shape == (1, 512, 512)
    assert operator.range_shape == (1, 360, 900)

    test_volume = torch.rand(*operator.domain_shape)
    test_projection = torch.rand(*operator.range_shape)

    # Make sure that tomosipo Operator's attributes are accessible
    assert operator._fp(volume=test_volume, out=None) is not None
    assert operator._bp(projection=test_projection, out=None) is not None
    assert operator.transpose() is not None
    assert operator.T is not None
    assert operator.domain is not None
    assert operator.range is not None


# --------------------------------------------------------------------------
# FDK / FBP


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
GEOMETRY_IDS = ["fan", "parallel"]


def phantom(geometry):
    _, H, W = geometry.image_shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    disk = ((xx**2 + yy**2) < 0.5**2).float()
    return disk.reshape(1, 1, H, W).cuda()


@pytest.mark.tomosipo
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=GEOMETRY_IDS)
def test_reconstructs_both_geometries(geometry):
    x = phantom(geometry)
    op = make_operator(geometry)
    sino = op(x[0]).unsqueeze(0)
    recon = fdk(sino, op)
    assert recon.shape == x.shape
    assert torch.linalg.vector_norm(recon - x) / torch.linalg.vector_norm(x) < 0.3


@pytest.mark.tomosipo
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=GEOMETRY_IDS)
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


@pytest.mark.tomosipo
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=GEOMETRY_IDS)
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


@pytest.mark.tomosipo
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


@pytest.mark.tomosipo
def test_deepfusionbp_single_view_backprojections_match_the_operator():
    """The per-angle backprojectors are built from the LION geometry: summed over
    the angles they give the CT operator's backprojection."""
    from LION.models.learned_fbp.DeepFusionBP import single_back_projections

    geometry = fan_geometry()
    geometry.angles = geometry.angles[::10]  # sparse view: 9 angles
    op = make_operator(geometry)
    sino = torch.rand(2, 1, len(geometry.angles), geometry.detector_shape[1]).cuda()
    views = single_back_projections(geometry)(sino)  # (B, angles, H, W)
    assert views.shape == (2, len(geometry.angles), *geometry.image_shape[1:])
    expected = torch.stack([op.T(s) for s in sino])  # (B, 1, H, W)
    torch.testing.assert_close(
        views.sum(dim=1, keepdim=True), expected, rtol=1e-3, atol=1e-3
    )
