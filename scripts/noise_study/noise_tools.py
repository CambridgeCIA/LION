# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Tools for comparing LION's two sinogram noise models, and for measuring the
noise of real data. Used by the scripts in this folder; tested (on CPU, with
synthetic data) in pytests/test_noise_study.py.

The two models of ``LION.CTtools.ct_utils.sinogram_add_noise`` both compute
transmission T = exp(-p / s) of a line integral p, simulate counts
N = blur(Poisson(I0 T)) + N(0, sigma^2) and return p_noisy = -s log(N / I0).
They differ only in the scale s:

- relative (default): s = max(p) of each sample, so the darkest ray of every
  sample keeps I0 / e photons whatever the object;
- physical (``mass_attenuation``): s = 1 / (mu/rho * length_unit_cm), so T is
  the real transmission, and I0 is photons per detector pixel in vacuum.

To first order (the delta method) the noise variance of p_noisy is

    var(p) = s^2 (w I0 T + sigma^2) / (I0 T)^2 = s^2 (w / (I0 T) + sigma^2 / (I0 T)^2)

with w = sum of the squared cross-talk kernel weights (the blur along the
detector lowers the per-pixel variance). Everything below uses this formula.
It is accurate while I0 T is more than ~10 photons; below that (photon
starvation) the clipping of N makes the real noise non-Gaussian, which only an
actual simulation (see calibrate_lidc.py) shows.
"""

import math

import numpy as np
import torch

# --------------------------------------------------------------------------
# The variance of each model


def blur_variance_factor(sigma_blur=0.3015, ks_value=3):
    """
    w = sum of squares of the cross-talk kernel that sinogram_add_noise uses
    (the same Gaussian kernel as torchvision's gaussian_blur), so that blurred
    Poisson noise of variance v has per-pixel variance w v.
    """
    ks = max(int(sigma_blur * ks_value) * 2 + 1, 3)
    x = torch.linspace(-(ks - 1) / 2, (ks - 1) / 2, ks, dtype=torch.float64)
    k = torch.exp(-0.5 * (x / sigma_blur) ** 2)
    k = k / k.sum()
    return float(torch.sum(k**2))


def relative_scale(p):
    """The scale s of the relative model: the maximum of each sample.
    p is (C, A, D) (returns a 0-d tensor) or (B, C, A, D) (returns (B, 1, 1, 1))."""
    if p.dim() == 4:
        s = torch.amax(p, dim=(1, 2, 3), keepdim=True)
    else:
        s = torch.amax(p)
    return torch.where(s <= 0, torch.ones_like(s), s)


def physical_scale(mass_attenuation, length_unit_cm=0.1):
    """The scale s of the physical model: line integral units per unit of attenuation."""
    return 1.0 / (mass_attenuation * length_unit_cm)


def predicted_variance(p, I0, scale, sigma=5.0, w=1.0):
    """
    First-order noise variance of every ray of p, in the units of p, for
    I0 photons per pixel in vacuum and scale s (see the module docstring).
    I0 and scale can be tensors that broadcast against p.
    """
    counts = I0 * torch.exp(-p / scale)
    return scale**2 * (w * counts + sigma**2) / counts**2


def equivalent_physical_I0(
    p,
    I0_relative,
    mass_attenuation,
    length_unit_cm=0.1,
    sigma=5.0,
    w=1.0,
    min_line_integral=0.0,
):
    """
    For each sample of p (density line integrals, (C, A, D) or (B, C, A, D)),
    the physical I0 whose total sinogram noise variance (summed over the
    rays through the object, p > min_line_integral) equals that of the
    relative model at I0_relative. (Rays that miss the object are left out:
    sinogram_add_noise clips negative line integrals to 0, which halves their
    noise in a way the first-order formula does not see.)

    For filtered backprojection with uncorrelated noise, the total sinogram
    variance is proportional to the noise power of the whole reconstruction,
    so this is the physical dose that gives the same average image noise.
    Solved by bisection in log I0 (the variance decreases with I0).
    Returns a tensor of shape (B,) (or 0-d for one sample).
    """
    p = p.double()
    batched = p.dim() == 4
    if not batched:
        p = p.unsqueeze(0)
    dims = (1, 2, 3)
    through = (p > min_line_integral).double()
    target = (
        predicted_variance(p, I0_relative, relative_scale(p), sigma=sigma, w=w)
        * through
    ).sum(dim=dims)
    s_phys = physical_scale(mass_attenuation, length_unit_cm)

    def total(log_I0):
        I0 = torch.exp(log_I0).reshape(-1, 1, 1, 1)
        var = predicted_variance(p, I0, s_phys, sigma=sigma, w=w)
        return (var * through).sum(dim=dims)

    lo = torch.full_like(target, math.log(1e-3))
    hi = torch.full_like(target, math.log(1e15))
    for _ in range(200):  # 200 halvings of a 41-wide log interval: exact
        mid = (lo + hi) / 2
        too_noisy = total(mid) > target
        lo = torch.where(too_noisy, mid, lo)
        hi = torch.where(too_noisy, hi, mid)
    out = torch.exp((lo + hi) / 2)
    return out if batched else out[0]


# --------------------------------------------------------------------------
# Analytic phantoms


def water_cylinder_sinogram(diameter_mm, geometry, density=1.0137):
    """
    Density line integrals (g/cm^3 * mm) of a centred water cylinder of the
    given diameter, for a LION fan-beam geometry, shape (1, angles, detectors).
    Exact (no discretisation): chord length times density.
    """
    n_det = int(geometry.detector_shape[-1])
    du = float(geometry.detector_size[-1]) / n_det
    u = (torch.arange(n_det, dtype=torch.float64) - (n_det - 1) / 2) * du
    # distance of each ray from the rotation axis
    d = float(geometry.dso) * torch.sin(torch.atan(u / float(geometry.dsd)))
    r = diameter_mm / 2
    chord = 2 * torch.sqrt(torch.clamp(r**2 - d**2, min=0))
    n_angles = len(geometry.angles)
    return (density * chord).reshape(1, 1, n_det).repeat(1, n_angles, 1)


# --------------------------------------------------------------------------
# Measuring the noise of a sinogram (simulated or real)


def angular_residual(sino):
    """
    r = p[a] - (p[a-1] + p[a+1]) / 2 along the angle axis (second to last),
    and a guide value of the line integral at the same place: the mean of
    views a-3, a-2, a+2, a+3. With finely sampled angles (e.g. the 3600 of
    2DeteCT) the object barely changes between neighbouring views, so r is
    almost pure noise, with var(r) = 1.5 var(p) for noise independent between
    views. Cross-talk along the detector does not change this.

    Bin r by the guide, not by p[a]: the guide's noise is independent of r,
    whereas binning by the noisy p[a] would truncate the noise of r and
    underestimate the variance.
    """
    n = sino.shape[-2]
    centre = sino[..., 3 : n - 3, :]
    r = centre - 0.5 * (sino[..., 2 : n - 4, :] + sino[..., 4 : n - 2, :])
    guide = 0.25 * (
        sino[..., 0 : n - 6, :]
        + sino[..., 1 : n - 5, :]
        + sino[..., 5 : n - 1, :]
        + sino[..., 6:n, :]
    )
    return r, guide


def binned_noise_variance(p, r, edges, r_variance_factor=1.5, min_count=200):
    """
    Robust per-bin noise variance of p, from residuals r (same shape as p),
    binning by the values of p (e.g. by line integral, or by transmission).
    Uses the median absolute deviation, so object edges (where r also holds
    some signal) do not inflate it. Returns (bin centres, variance, counts)
    for the bins with at least min_count values; the centre of a bin is the
    mean of its values of p.
    """
    p = p.detach().reshape(-1).double().cpu().numpy()
    r = r.detach().reshape(-1).double().cpu().numpy()
    which = np.digitize(p, edges) - 1
    centres, variances, counts = [], [], []
    for b in range(len(edges) - 1):
        rb = r[which == b]
        if rb.size < min_count:
            continue
        mad = np.median(np.abs(rb - np.median(rb)))
        centres.append(p[which == b].mean())  # where the values are, not the bin middle
        variances.append((1.4826 * mad) ** 2 / r_variance_factor)
        counts.append(rb.size)
    return np.array(centres), np.array(variances), np.array(counts)


def fit_noise_model(p, variance, counts, scale=1.0, w=1.0, min_photons=20.0):
    """
    Fit var(p) = s^2 (w / (I0 T) + sigma^2 / (I0 T)^2), T = exp(-p / s), for
    I0 and sigma, to binned variances (from binned_noise_variance).
    Linear least squares in a = w / I0 and b = sigma^2 / I0^2, weighted by
    the relative error of each bin's variance (sqrt(2 / count)).

    The formula is first order: bins where the fitted model expects fewer
    than min_photons photons (I0 T) are dropped and the fit repeated.

    scale=1 is the physical model of measured, log-transformed data; scale =
    max of the sample is the relative model. Returns a dict with I0, sigma
    and rel_rms, the RMS relative misfit of the model to the bins used (a
    model that is wrong in shape cannot fit well whatever I0 is).
    """
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(variance, dtype=np.float64) / scale**2
    n = np.asarray(counts, dtype=np.float64)
    keep = np.ones(p.shape, dtype=bool)
    for _ in range(2):
        a, b = _fit_ab(p[keep], v[keep], n[keep], scale)
        enough = (w / a) * np.exp(-p / scale) >= min_photons
        if np.array_equal(enough, keep) or enough.sum() < 3:
            break
        keep = enough
    I0 = w / a
    sigma = math.sqrt(b) * I0
    inv_T = np.exp(p[keep] / scale)
    fit = a * inv_T + b * inv_T**2
    rel_rms = float(np.sqrt(np.mean(((fit - v[keep]) / v[keep]) ** 2)))
    return {"I0": float(I0), "sigma": float(sigma), "rel_rms": rel_rms}


def _fit_ab(p, v, n, scale):
    inv_T = np.exp(p / scale)
    X = np.stack([inv_T, inv_T**2], axis=1)
    weight = np.sqrt(n / 2) / v  # 1 / std of v
    (a, b), *_ = np.linalg.lstsq(X * weight[:, None], v * weight, rcond=None)
    if a <= 0 or b < 0:  # a negative term is unphysical: refit with photon noise only
        a = np.sum(weight**2 * v * inv_T) / np.sum(weight**2 * inv_T**2)
        b = 0.0
    return a, b
