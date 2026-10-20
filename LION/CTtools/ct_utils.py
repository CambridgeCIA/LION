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


def from_HU_to_mu(img):
    """
    Converts image in Hounsfield Units (air-> -1000, bone->500) into linear attenuation coefficient (air-> 0.0012,
    bone->1.52 g/cm^3). Approximate.
    Comercial scanners use a piecewise linear function. Check STIR for real values. (https://raw.githubusercontent.com/UCL/STIR/85cc1940c297b1749cf44a9fba937d7cefdccd47/src/utilities/share/ct_slopes.json)
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


def sinogram_add_noise(
    proj,
    I0=1000,
    sigma=5,
    sigma_blur=0.3015,
    ks_value=3,
    flat_field=None,
    dark_field=None,
    enable_gradients=False,
):
    """
    Wraper for _sinogram_add_noise to support gradients
    """
    if enable_gradients:
        sino = _sinogram_add_noise(
            proj, I0, sigma, sigma_blur, ks_value, flat_field, dark_field
        )
    else:
        if torch.is_tensor(proj):
            proj = proj.detach()
        with torch.no_grad():
            sino = _sinogram_add_noise(
                proj, I0, sigma, sigma_blur, ks_value, flat_field, dark_field
            )
    return sino


def _sinogram_add_noise(
    proj,
    I0=1000,
    sigma=5,
    sigma_blur=0.3015,
    ks_value=3,
    flat_field=None,
    dark_field=None,
):
    """
    Adds realistic noise to sinograms.
    - Poisson noise, with I0 counts in a scanner with no sample (bigger value==less noise)
    - Gaussian noise of zero mean and sigma std
    - Detector crosstalk in % of the signal of adjacent pixels.
    - Add a flat_field to add even more realistic noise (computed at non-corrected flat fields)
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
    max_val = torch.amax(
        proj
    )  # alternatively the highest power of 2 close to this value, but lets leave it as is.
    if max_val <= 0:
        max_val = 1.0
    Im = I0 * torch.exp(-proj / max_val)

    # Uncorrect the flat fields
    Im = Im * (flat_field - dark_field) + dark_field

    # Add Poisson noise
    Im = torch.poisson(Im)

    # Detector cross talk
    ks = int(sigma_blur * ks_value) * 2 + 1
    if ks < 3:
        ks = 3
    Im = TF.gaussian_blur(Im, kernel_size=[ks, ks], sigma=[sigma_blur, sigma_blur])

    # Electronic noise:
    Im = Im + sigma * torch.randn(Im.shape, device=dev)

    Im[Im <= 0] = 1e-6
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
