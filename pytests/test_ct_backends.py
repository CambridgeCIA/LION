# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""CT backends: the library-wide choice, recorded in the geometry and used
again on load; TIGRE's operator in LION's layout (with a stand-in for TIGRE, a
matrix, so it runs anywhere); and TIGRE against tomosipo (where both are
installed, with CUDA)."""

import sys
import types

import numpy as np
import pytest
import torch

from LION.operators.ct import CTProjectionOp, Geometry, fdk, set_ct_backend
from LION.operators.ct import backend as ct_backend_module
from LION.utils import spec


def small_geometry(mode="fan", n=16, n_angles=12, n_detectors=24):
    return Geometry(
        mode=mode,
        image_shape=[1, n, n],
        image_size=[1.0, float(n), float(n)],
        detector_shape=[1, n_detectors],
        detector_size=[1.0, 1.5 * n_detectors],
        dso=4.0 * n,
        dsd=8.0 * n,
        angles=np.linspace(0, 2 * np.pi, n_angles, endpoint=False),
    )


@pytest.fixture(autouse=True)
def restore_ct_backend():
    backend = ct_backend_module.ct_backend()
    yield
    set_ct_backend(backend)


@pytest.fixture
def stand_in_tigre(monkeypatch):
    """A ``tigre`` whose Ax is a fixed random matrix (per geometry and angles)
    and whose Atb is its transpose, in TIGRE's layout."""
    matrices = {}

    def matrix(geo, angles):
        key = (tuple(geo.nVoxel), tuple(geo.nDetector), len(angles))
        if key not in matrices:
            n_in = int(np.prod(geo.nVoxel))
            n_out = len(angles) * int(np.prod(geo.nDetector))
            generator = torch.Generator().manual_seed(0)
            matrices[key] = torch.rand(n_out, n_in, generator=generator)
        return matrices[key]

    def Ax(img, geo, angles, **kwargs):
        x = torch.as_tensor(img)
        out = matrix(geo, angles) @ x.reshape(-1)
        out = out.reshape(len(angles), *[int(n) for n in geo.nDetector])
        return out.numpy() if isinstance(img, np.ndarray) else out

    def Atb(proj, geo, angles, backprojection_type="FDK", **kwargs):
        y = torch.as_tensor(proj)
        assert y.shape == (len(angles), *[int(n) for n in geo.nDetector])
        out = (matrix(geo, angles).T @ y.reshape(-1)).reshape(
            *[int(n) for n in geo.nVoxel]
        )
        return out.numpy() if isinstance(proj, np.ndarray) else out

    tigre = types.ModuleType("tigre")
    tigre.Ax, tigre.Atb = Ax, Atb
    utilities = types.ModuleType("tigre.utilities")
    geometry = types.ModuleType("tigre.utilities.geometry")
    geometry.Geometry = type("Geometry", (), {})
    for name, module in (
        ("tigre", tigre),
        ("tigre.utilities", utilities),
        ("tigre.utilities.geometry", geometry),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return tigre


def test_backend_is_recorded_and_used_again(stand_in_tigre):
    geometry = small_geometry()
    assert geometry.backend == "default"
    set_ct_backend("tigre")
    assert geometry.spec_kwargs()["backend"] == "tigre"
    op = geometry.make_operator()
    assert op.backend == geometry.backend == "tigre"

    # a record of it makes a TIGRE operator whatever the library-wide choice
    record = spec.encode(geometry)
    set_ct_backend("tomosipo")
    assert spec.build(record).make_operator().backend == "tigre"

    with pytest.raises(ValueError, match="not supported"):
        set_ct_backend("astra")
    geometry.backend = "astra"
    with pytest.raises(ValueError, match="backend"):
        CTProjectionOp(geometry)


@pytest.mark.parametrize("mode", ["fan", "parallel"])
def test_tigre_operator_in_lion_layout(stand_in_tigre, mode):
    op = CTProjectionOp(small_geometry(mode), backend="tigre")
    x = torch.rand(op.domain_shape, dtype=torch.float64).float()
    y = torch.rand(op.range_shape)
    assert op(x).shape == op.range_shape == (1, 12, 24)
    assert op.adjoint(y).shape == op.domain_shape
    # the backprojection is the adjoint, through LION's layout
    assert torch.allclose(
        (op(x) * y).sum(), (x * op.adjoint(y)).sum(), rtol=1e-4, atol=1e-3
    )

    # batches through autograd, with the adjoint as gradient
    A, AT = op.autograd()
    xb = torch.rand(3, *op.domain_shape, requires_grad=True)
    yb = A(xb)
    assert yb.shape == (3, *op.range_shape)
    weights = torch.rand_like(yb)
    (yb * weights).sum().backward()
    assert torch.allclose(xb.grad, AT(weights), rtol=1e-4, atol=1e-4)

    recon = fdk(yb.detach(), op, clip=False)
    assert recon.shape == xb.shape
    sino = yb.detach().requires_grad_()
    fdk(sino, op, differentiable=True).sum().backward()
    assert sino.grad.shape == sino.shape


@pytest.mark.tigre
@pytest.mark.tomosipo
@pytest.mark.parametrize("mode", ["fan", "parallel"])
def test_tigre_matches_tomosipo(mode):
    geometry = small_geometry(mode, n=64, n_angles=90, n_detectors=96)
    tomosipo_op = CTProjectionOp(geometry, backend="tomosipo")
    tigre_op = CTProjectionOp(geometry, backend="tigre")
    x = torch.zeros(tigre_op.domain_shape, device="cuda")
    x[0, 16:48, 20:40] = 1.0
    x[0, 24:32, 40:52] = 0.5

    ts_sino, tigre_sino = tomosipo_op(x), tigre_op(x)
    assert torch.linalg.norm(tigre_sino - ts_sino) < 0.02 * torch.linalg.norm(ts_sino)

    y = torch.rand(tigre_op.range_shape, device="cuda")
    lhs, rhs = (tigre_op(x) * y).sum(), (x * tigre_op.adjoint(y)).sum()
    assert torch.isclose(lhs, rhs, rtol=1e-2)

    ts_recon, tigre_recon = fdk(ts_sino, tomosipo_op), fdk(ts_sino, tigre_op)
    assert torch.linalg.norm(tigre_recon - ts_recon) < 0.05 * torch.linalg.norm(
        ts_recon
    )
