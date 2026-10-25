# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================

from __future__ import annotations

# math/science imports
import numpy as np
import torch
import torchvision.transforms.functional as TF

# AItomotools imports
from LION.CTtools.ct_geometry import Geometry
from LION.operators.CTProjectionOp import CTProjectionOp
from LION.utils.device import default_device


def from_HU_to_normal(img):
    """
    Converts image in Hounsfield Units (air-> -1000, bone->500) into a [0-1] image.
    Comercial scanners use a piecewise linear function. Check STIR for real values. (https://raw.githubusercontent.com/UCL/STIR/85cc1940c297b1749cf44a9fba937d7cefdccd47/src/utilities/share/ct_slopes.json)
    """
    if isinstance(img, np.ndarray):
        return np.minimum(np.maximum((img.astype(np.float32) + 1000) / 3000, 0), 1)
    elif isinstance(img, torch.Tensor):
        return torch.clip((img.float() + 1000) / 3000, min=0, max=1)
    else:
        raise NotImplementedError


# Mass attenuation coefficient of water, mu/rho in cm^2/g (NIST XCOM), at a few
# effective energies (keV) of a CT spectrum.
WATER_MASS_ATTENUATION = {50: 0.2269, 60: 0.2059, 80: 0.1837, 100: 0.1707}
# Default of the physical noise model: water at 80 keV, close to the effective
# energy of a 120 kV clinical beam after the patient (decided 25 Oct 2026).
DEFAULT_MASS_ATTENUATION = WATER_MASS_ATTENUATION[80]


def from_HU_to_density(img):
    """
    Converts an image in Hounsfield Units into (approximate) mass density in
    g/cm^3, linearly: air (-1000 HU) -> 0.0012, water (0 HU) -> 1.0137,
    500 HU -> 1.52. Real tissue is closer to bilinear (bone is denser per HU);
    this is the conversion LION uses for its simulated datasets.
    """
    if isinstance(img, np.ndarray):
        return np.maximum(
            ((1.52 - 0.0012) / (500 + 1000)) * (img.astype(np.float32) + 1000) + 0.0012,
            0,
        )
    elif isinstance(img, torch.Tensor):
        return torch.clip(
            ((1.52 - 0.0012) / (500 + 1000)) * (img.float() + 1000) + 0.0012, min=0.0
        )
    else:
        raise NotImplementedError


def from_HU_to_mu(img, mass_attenuation=None, length_unit_cm=0.1):
    """
    Linear attenuation coefficient mu = density * (mu/rho), per length unit
    (default 1/mm, as LION geometries are in mm), with ``mass_attenuation``
    the mass attenuation coefficient mu/rho in cm^2/g (e.g.
    ``WATER_MASS_ATTENUATION[60]``: water-equivalent, monochromatic).

    Without ``mass_attenuation`` it returns the density (g/cm^3) of
    ``from_HU_to_density``, as it did before October 2026 (when its docstring
    called that the attenuation).
    """
    density = from_HU_to_density(img)
    if mass_attenuation is None:
        return density
    return density * (mass_attenuation * length_unit_cm)


NOISE_MODELS = ("physical", "relative")
DETECTORS = ("energy_integrating", "photon_counting")


def sinogram_add_noise(
    proj,
    I0=1000,
    sigma=5,
    sigma_blur=0.3015,
    ks_value=3,
    flat_field=None,
    dark_field=None,
    enable_gradients=False,
    legacy=False,
    model="physical",
    detector="energy_integrating",
    mass_attenuation=DEFAULT_MASS_ATTENUATION,
    length_unit_cm=0.1,
    min_counts=1.0,
):
    """
    Adds realistic noise to sinograms (see _sinogram_add_noise), with or
    without gradients.

    ``model`` says how the line integrals ``proj`` become attenuation, i.e.
    what ``I0`` means:

    - ``"physical"`` (default): ``proj`` are line integrals of a density image
      (g/cm^3, e.g. from ``from_HU_to_density``) in ``length_unit_cm`` (0.1:
      mm, as LION geometries), and the transmission is
      exp(-mass_attenuation * proj * length_unit_cm), with ``mass_attenuation``
      the mu/rho of water in cm^2/g (default ``DEFAULT_MASS_ATTENUATION``, water
      at 80 keV; monochromatic, water-equivalent). ``I0`` is the number of photons per
      detector pixel in vacuum, as in XCIST, and larger objects get noisier
      data, as in a scanner.
    - ``"relative"``: ``proj`` is divided by the maximum of each sample, so the
      most attenuating ray always keeps e^-1 of I0, whatever the object and its
      units. LION's model before October 2026 (``mass_attenuation`` unused).

    The returned sinogram has the units of ``proj`` in both cases.

    ``detector``:

    - ``"energy_integrating"`` (default; most clinical scanners): Poisson
      photon counts, detector cross-talk (a Gaussian blur of std
      ``sigma_blur`` pixels along the detector) and Gaussian electronic noise
      of std ``sigma`` photons, as in XCIST.
    - ``"photon_counting"``: the Poisson photon counts themselves. The energy
      threshold rejects electronic noise, so ``sigma`` and ``sigma_blur`` are
      not used, and the counts are integers >= 0. Charge sharing, pulse
      pile-up and energy resolution are not modelled.

    ``min_counts``: the detector signal (photons plus electronic noise, the
    dark field subtracted) can be 0 or negative when a ray gets (almost) no
    photons, and the log needs a positive value. Signals below ``min_counts``
    (in photons; default 1) are set to it, so a starved ray reads as at most
    log(I0 / min_counts) of attenuation. None: LION's old rule, signals <= 0
    become 1e-6, i.e. log(I0) + 13.8, far beyond any real attenuation, which
    turns starved rays into strong streaks. It only matters when rays are
    starved, so in practice only with the physical model. A photon-counting
    detector's signal is never negative, but it can be 0.

    legacy=True reproduces the noise of LION before October 2026 exactly: the
    relative model, with detector cross-talk blurred across neighbouring angles
    as well as detectors, the attenuation scaled by the maximum of the whole
    input (so in a batch, each sample's noise depended on the others) and the
    old rule for starved rays, energy-integrating (``model``, ``detector``,
    ``mass_attenuation`` and ``min_counts`` unused).
    """
    if model not in NOISE_MODELS:
        raise ValueError(f"model must be one of {NOISE_MODELS}, got {model!r}")
    if detector not in DETECTORS:
        raise ValueError(f"detector must be one of {DETECTORS}, got {detector!r}")
    if legacy:
        model, detector, min_counts = "relative", "energy_integrating", None
    kwargs = dict(
        I0=I0,
        sigma=sigma,
        sigma_blur=sigma_blur,
        ks_value=ks_value,
        flat_field=flat_field,
        dark_field=dark_field,
        legacy=legacy,
        mass_attenuation=mass_attenuation if model == "physical" else None,
        photon_counting=detector == "photon_counting",
        length_unit_cm=length_unit_cm,
        min_counts=min_counts,
    )
    if enable_gradients:
        return _sinogram_add_noise(proj, **kwargs)
    if torch.is_tensor(proj):
        proj = proj.detach()
    with torch.no_grad():
        return _sinogram_add_noise(proj, **kwargs)


def _sinogram_add_noise(
    proj,
    I0=1000,
    sigma=5,
    sigma_blur=0.3015,
    ks_value=3,
    flat_field=None,
    dark_field=None,
    legacy=False,
    mass_attenuation=None,
    length_unit_cm=0.1,
    min_counts=None,
    photon_counting=False,
):
    """
    Adds realistic noise to sinograms, (C, angles, detectors) or batched
    (B, C, angles, detectors).
    - Poisson noise, with I0 counts in a scanner with no sample (bigger value==less noise)
    - Gaussian noise of zero mean and sigma std
    - Detector crosstalk: a Gaussian blur (std sigma_blur pixels) along the detector,
      within each projection.
    - Add a flat_field to add even more realistic noise (computed at non-corrected flat fields)
    ``mass_attenuation=None`` is the relative model, otherwise the physical one;
    photon_counting: only the Poisson counts (no cross-talk, no electronic
    noise). See sinogram_add_noise for the models, the detectors,
    ``min_counts`` and ``legacy``.
    """
    if torch.is_tensor(proj):
        istorch = True
        dev = proj.device
    elif isinstance(proj, np.ndarray):
        istorch = False
        dev = default_device()
        proj = torch.from_numpy(proj).to(dev)
    else:
        raise ValueError("numpy or torch tensor expected")
    if dark_field is None:
        dark_field = torch.zeros(proj.shape, device=dev)
    if flat_field is None:
        flat_field = torch.ones(proj.shape, device=dev)
    # Scale of the line integrals: attenuation = proj / max_val
    if mass_attenuation is not None:
        # physical: mu/rho [cm^2/g] * density line integral [g/cm^3 * length unit]
        max_val = torch.tensor(1.0 / (mass_attenuation * length_unit_cm), device=dev)
    else:
        # relative: the maximum of each sample (alternatively the highest power
        # of 2 close to this value, but lets leave it as is)
        if proj.dim() == 4 and not legacy:
            max_val = torch.amax(proj, dim=(1, 2, 3), keepdim=True)
        else:  # one sample (or, legacy, the whole batch)
            max_val = torch.amax(proj)
        max_val = torch.where(max_val <= 0, torch.ones_like(max_val), max_val)
    Im = I0 * torch.exp(-proj / max_val)

    # Uncorrect the flat fields
    Im = Im * (flat_field - dark_field) + dark_field

    # Add Poisson noise
    Im = torch.poisson(Im)

    if not photon_counting:  # energy-integrating detector
        # Detector cross talk
        ks = int(sigma_blur * ks_value) * 2 + 1
        if ks < 3:
            ks = 3
        if legacy:  # blurred across neighbouring angles too
            Im = TF.gaussian_blur(
                Im, kernel_size=[ks, ks], sigma=[sigma_blur, sigma_blur]
            )
        else:  # kernel [width, height]: along the detector (last axis) only
            Im = TF.gaussian_blur(Im, kernel_size=[ks, 1], sigma=[sigma_blur, 1.0])

        # Electronic noise:
        Im = Im + sigma * torch.randn(Im.shape, device=dev)

    if min_counts is None:  # LION's old rule
        Im[Im <= 0] = 1e-6
    else:  # starved rays: at most log(I0 / min_counts) of attenuation
        Im = torch.clamp(Im, min=min_counts)
    # Correct flat fields
    Im = (Im - dark_field) / (flat_field - dark_field)
    proj = -torch.log(Im / I0) * max_val
    proj[proj < 0] = 0
    if istorch:
        return proj
    else:
        return proj.cpu().detach().numpy()


# Upper HU limit of each material class; values >= the last limit are bone.
_MATERIAL_HU_LIMITS = [-950, -750, -150, 0, 150, 300]
# 0: air, 1: lung, 2: bronchi, 3: fat, 4: muscle, 5: bone marrow, 6: bone


def from_HU_to_material_id(img):
    """
    Converts an image in Hounsfield units into a material index:
    0 air (< -950), 1 lung [-950, -750), 2 bronchi [-750, -150), 3 fat [-150, 0),
    4 muscle [0, 150), 5 bone marrow [150, 300), 6 bone (>= 300).
    Returns a new uint8 array/tensor; the input is not modified.
    May require some image filtering preprocessing
    """
    if isinstance(img, torch.Tensor):
        limits = torch.tensor(_MATERIAL_HU_LIMITS, dtype=img.dtype, device=img.device)
        return torch.bucketize(img, limits, right=True).to(torch.uint8)
    return np.digitize(np.asarray(img), _MATERIAL_HU_LIMITS).astype(np.uint8)


def make_operator(geometry: Geometry) -> CTProjectionOp:
    return CTProjectionOp(geometry)


def forward_projection(
    image: np.ndarray | torch.Tensor,
    geometry: Geometry,
    backend: str = "tomosipo",
) -> torch.Tensor:
    """
    Produces a noise free forward projection of a 2D image with the given geometry
    (fan or parallel beam, as set in geometry.mode).
    May support other backends than tomosipo
    """
    if backend != "tomosipo":
        raise ValueError("Only tomosipo backend for CT supported")

    if isinstance(image, np.ndarray):
        device = default_device()
        image = torch.from_numpy(image).float().to(device)
    if len(image.shape) == 3:
        if image.shape[0] > 1:  # there is no reason to have this constraint
            raise ValueError("Image must be 2D")
    elif len(image.shape) == 2:
        image = torch.unsqueeze(image, 0)
    else:
        raise ValueError("Image must be 2D")

    # Same operator as everywhere else in LION (respects mode and image_pos)
    A = CTProjectionOp(geometry, backend=backend)
    sino = A(image)[0]
    return sino
