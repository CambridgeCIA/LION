"""The tools of scripts/noise_study (comparing the relative and physical noise
models, and measuring noise in real data), on CPU with synthetic data."""

import importlib.util
import math
import pathlib

import numpy as np
import pytest
import torch

from LION.CTtools import ct_utils as ct
from LION.CTtools.ct_geometry import Geometry

_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "noise_study" / "noise_tools.py"
_spec = importlib.util.spec_from_file_location("noise_tools", _PATH)
nt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nt)

MU_RHO = ct.WATER_MASS_ATTENUATION[60]


def ramp_sinogram(top, n_angles=4000, n_det=12):
    """(1, angles, detectors): the same line integrals in every view, rising
    across the detector from 0.1 * top to top."""
    p = torch.linspace(0.1 * top, top, n_det, dtype=torch.float64)
    return p.reshape(1, 1, n_det).repeat(1, n_angles, 1)


@pytest.mark.parametrize("model", ["relative", "physical"])
@pytest.mark.parametrize("sigma_blur", [0.3015, 1.0])
def test_predicted_variance_matches_lions_simulation(model, sigma_blur):
    torch.manual_seed(0)
    I0, sigma = 5000, 5.0
    if model == "physical":
        p = ramp_sinogram(150.0)  # density * mm: up to ~4.5 attenuation
        scale, kwargs = nt.physical_scale(MU_RHO), dict(mass_attenuation=MU_RHO)
    else:
        p = ramp_sinogram(2.0)
        scale, kwargs = nt.relative_scale(p), dict(model="relative")
    noisy = ct.sinogram_add_noise(
        p.float(), I0=I0, sigma=sigma, sigma_blur=sigma_blur, **kwargs
    ).double()
    # the blur mixes neighbouring detectors: compare away from the borders
    measured = noisy.var(dim=1)[0, 2:-2]
    w = nt.blur_variance_factor(sigma_blur)
    predicted = nt.predicted_variance(p[0, 0, 2:-2], I0, scale, sigma=sigma, w=w)
    # 4000 samples per column: the variance is known to ~2 sqrt(2/4000) = 4.5 %
    torch.testing.assert_close(measured, predicted, rtol=0.07, atol=0)


def test_blur_factor():
    assert nt.blur_variance_factor(0.3015) == pytest.approx(0.984, abs=1e-3)
    assert nt.blur_variance_factor(1e-3) == pytest.approx(1.0)
    assert nt.blur_variance_factor(1.0) < 0.5


def test_water_cylinder_sinogram():
    g = Geometry.default_parameters()
    p = nt.water_cylinder_sinogram(300.0, g)
    assert p.shape == (1, len(g.angles), g.detector_shape[-1])
    assert p.max().item() == pytest.approx(300 * 1.0137, rel=1e-4)  # central ray
    assert p[0, 0, 0] == 0 and p[0, 0, -1] == 0  # outside the cylinder
    torch.testing.assert_close(p[0, 0], p[0, -1])  # same in every view


def test_equivalent_I0_is_I0_when_the_two_scales_agree():
    """If the sample's maximum is exactly the physical scale, the two models
    are the same model, so the equivalent physical I0 is the relative I0."""
    s = nt.physical_scale(MU_RHO)
    p = ramp_sinogram(s, n_angles=3)
    assert nt.equivalent_physical_I0(p, 3500, MU_RHO).item() == pytest.approx(3500)


def test_equivalent_I0_grows_with_the_object():
    """Relative: every object gets I0/e at its darkest ray. The same image
    noise with a fixed physical I0 needs more photons the bigger the object."""
    g = Geometry.default_parameters()
    p = torch.stack(
        [nt.water_cylinder_sinogram(d, g)[:, :3] for d in (100.0, 200.0, 300.0, 400.0)]
    )
    eq = nt.equivalent_physical_I0(p, 10000, MU_RHO)
    assert eq.shape == (4,)
    assert torch.all(eq[1:] > eq[:-1])
    # and it equalises the total variance it was asked to
    s = nt.physical_scale(MU_RHO)
    for k in range(4):
        inside = p[k] > 0  # rays that miss the object are left out
        relative = nt.predicted_variance(p[k], 10000, nt.relative_scale(p[k]))
        physical = nt.predicted_variance(p[k], eq[k], s)
        relative, physical = relative[inside].sum(), physical[inside].sum()
        assert physical.item() == pytest.approx(relative.item(), rel=1e-9)


# --------------------------------------------------------------------------
# Measuring noise: synthetic "measured" data, physical noise (log-transformed
# counts, scale 1), angles finely sampled as in 2DeteCT


def measured_slice(top, I0, sigma, n_angles=3600, n_det=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = torch.linspace(-1, 1, n_det, dtype=torch.float64)
    a = torch.linspace(0, 2 * math.pi, n_angles, dtype=torch.float64)
    # an off-centre ellipse-like object whose sinogram changes with the angle
    p = top * torch.clamp(1 - (d[None, :] - 0.2 * torch.sin(a[:, None])) ** 2 / 0.6, 0)
    counts = torch.poisson(I0 * torch.exp(-p), generator=g)
    counts = counts + sigma * torch.randn(counts.shape, generator=g, dtype=p.dtype)
    return -torch.log(torch.clamp(counts, min=1e-6) / I0), p


def measure(noisy, edges):
    r, guide = nt.angular_residual(noisy)
    return nt.binned_noise_variance(guide, r, edges)


def test_angular_estimator_recovers_I0_and_sigma():
    I0, sigma = 2000, 8.0
    noisy, _ = measured_slice(3.0, I0, sigma)
    centres, var, counts = measure(noisy, np.linspace(0, 3.0, 16))
    fit = nt.fit_noise_model(centres, var, counts, scale=1.0)
    assert fit["I0"] == pytest.approx(I0, rel=0.03)
    assert fit["sigma"] == pytest.approx(sigma, rel=0.2)  # a small term: rougher
    assert fit["rel_rms"] < 0.05


def test_physical_fit_is_consistent_across_slices_and_relative_is_not():
    """Measured data follow var ~ 1/(I0 T) with one I0 for every slice. The
    relative model must give each slice its own I0 (it depends on the
    slice's maximum), and fits worse."""
    I0 = 2000
    fits = {"physical": [], "relative": []}
    for top in (1.5, 3.0):
        noisy, clean = measured_slice(top, I0, sigma=0.0, seed=int(top * 10))
        centres, var, counts = measure(noisy, np.linspace(0, top, 16))
        fits["physical"].append(nt.fit_noise_model(centres, var, counts, scale=1.0))
        s = clean.amax().item()  # what the relative model would use
        fits["relative"].append(nt.fit_noise_model(centres, var, counts, scale=s))
    phys = [f["I0"] for f in fits["physical"]]
    rel = [f["I0"] for f in fits["relative"]]
    assert phys[0] == pytest.approx(phys[1], rel=0.03)
    assert phys[0] == pytest.approx(I0, rel=0.03)
    assert abs(rel[0] / rel[1] - 1) > 0.3
    assert all(
        r["rel_rms"] > p["rel_rms"] for p, r in zip(fits["physical"], fits["relative"])
    )


def test_fit_drops_photon_starved_bins():
    """At I0 = 200 (2DeteCT's low dose) the thickest rays get ~2 photons,
    where the first-order formula fails; the fit must not be pulled by them."""
    noisy, _ = measured_slice(4.5, 200, sigma=3.0)
    centres, var, counts = measure(noisy, np.linspace(0.3, 4.5, 30))
    assert nt.fit_noise_model(centres, var, counts)["I0"] == pytest.approx(
        200, rel=0.05
    )
    naive = nt.fit_noise_model(centres, var, counts, min_photons=0)
    assert naive["I0"] > 1.2 * 200
