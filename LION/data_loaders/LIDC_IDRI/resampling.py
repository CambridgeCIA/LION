# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""Resampling of LIDC-IDRI slices to one pixel spacing, for the pre-processing.

LIDC-IDRI scans are 512x512 slices with each scanner's own pixel spacing,
0.46 to 0.98 mm (fields of view of 236 to 500 mm). The pre-processing
resamples every slice to ``PIXEL_SPACING_MM``, the largest spacing in the
dataset, on a 512x512 grid: a 500 mm field of view, so no scan is cropped,
and every image has the same physical size (which the physical noise model
of LION's CT experiments needs). Slices with a smaller spacing are shrunk
about their centre and surrounded by air.

Only numpy and scipy are used, so that this runs in the pre-processing
environment (pre_process_lidc_idri_environment.yml, Python 3.7).
"""

import numpy as np
from scipy import ndimage

PIXEL_SPACING_MM = 500.0 / 512.0  # 0.9765625 mm, the largest in LIDC-IDRI
IMAGE_SHAPE = (512, 512)
AIR_HU = -1000.0


def resample_slice(
    image,
    pixel_spacing,
    new_spacing=PIXEL_SPACING_MM,
    shape=IMAGE_SHAPE,
    fill=AIR_HU,
):
    """Resample a 2D slice to another pixel spacing, keeping its centre.

    Linear interpolation; when shrinking, the slice is first smoothed with a
    Gaussian of std (new_spacing / pixel_spacing - 1) / 2 input pixels, to avoid
    aliasing (as skimage.transform.rescale's anti_aliasing). Points outside the
    slice get ``fill``.

    Args:
        image: 2D array, square pixels of ``pixel_spacing`` mm.
        pixel_spacing: pixel spacing of ``image`` in mm.
        new_spacing: pixel spacing of the output in mm.
        shape: shape of the output.
        fill: value outside the slice (air, in HU).

    Returns:
        A float32 array of ``shape`` with pixels of ``new_spacing`` mm.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("expected a 2D slice, got shape {}".format(image.shape))
    # input pixels per output pixel
    step = float(new_spacing) / float(pixel_spacing)
    if step > 1:
        image = ndimage.gaussian_filter(image, sigma=(step - 1) / 2, mode="nearest")
    centre_in = (np.array(image.shape) - 1) / 2.0
    centre_out = (np.array(shape) - 1) / 2.0
    # input index = step * (output index - centre_out) + centre_in
    return ndimage.affine_transform(
        image,
        matrix=[step, step],
        offset=centre_in - step * centre_out,
        output_shape=tuple(shape),
        order=1,
        mode="constant",
        cval=fill,
    ).astype(np.float32)


def resample_ct_slice(image, pixel_spacing):
    """A slice in HU, resampled to PIXEL_SPACING_MM on 512x512, as int16 HU."""
    check_no_cropping(pixel_spacing)
    resampled = resample_slice(image, pixel_spacing)
    return np.round(resampled).astype(np.int16)


def resample_mask(mask, pixel_spacing):
    """A binary nodule mask, resampled as the slices: a pixel is in the
    resampled mask if at least half of it was."""
    check_no_cropping(pixel_spacing)
    resampled = resample_slice(mask, pixel_spacing, fill=0.0)
    return (resampled >= 0.5).astype(np.int16)


def check_no_cropping(pixel_spacing):
    """Raise if a slice with this spacing would not fit in 512 pixels."""
    if pixel_spacing > PIXEL_SPACING_MM * (1 + 1e-6):
        raise ValueError(
            "Pixel spacing {} mm is larger than LION's {} mm: the slice would be "
            "cropped".format(pixel_spacing, PIXEL_SPACING_MM)
        )
