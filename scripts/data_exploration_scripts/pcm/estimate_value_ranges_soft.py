# %%
from pprint import pprint

import numpy as np

# Medians?
#   CIGS        : [-1.970e-03, 2.821e-03] (using 512x512 data)
#   Organic     : [-3.791e-03, 1.485e-03] (using 512x512 data)
#   Perovskite  : [-1.674e-05, -1.466e-05] (using 256x256 data)
#   Reference   : [-2.501e-05, 5.145e-06] (using 512x512 data)
#   Silicone    : [-1.018e-01, 1.294e-01] (using 512x512 data)

# Questions:
#   - [Needs NPL or Tameem]: PNG images vs. TXT maps.
#     - PNG images are not aligned with TXT maps (e.g., sample 100_P2).
#       Probably centered the TXT maps?
#     - Normalization process seems to be different for each sample (e.g. sample 512_CIGS1).
#       Probably some manual tuning so that object is brighter?
#   - [Needs NPL] Why does the range gets smaller when the resolution decreases?
#     Shouldn't it be bigger because it covers more area?
#   - [Needs NPL] Could I depend on the acquisition process to normalize the data so that the current map is [0, 1]?
#     It could be slightly outside [0, 1] if there's noise.
#     - [Needs NPL] If acquisition can normalize data, then for the training we may need an updated normalized version
#       of the noise generation parameters?
#   - [Simplest?] Perhaps we can simply normalize the data using the min and max values and hope for the best?
#     - The thing is, all synthetic ground-truth images have at least one `0` and one `1`.
#       After simulating noise, they are likely to have values outside [0, 1].
#       If in application, we blindly normalize noisy images to [0, 1],
#       will it affect performance? Will the model be biased?
#         - Perhaps a workaround is to normalize again after adding synthetic noise during training,
#           so that all noisy images are also in [0, 1]?
#           Then use this as input to the denoiser.
#           Remember to also apply this to the ground-truth before computing the loss.
#             - Or to reverse-normalize the output of the model and keep the ground-truth?
#               This might be better in the sense that the things are more "normalized"?
#               Loss is always computed on the same scale?
#               But is it ok for the backpropagation? Maybe just accept a bit out-of-scale and not reverse-normalize.
#               But probably no problem for backpropagation? It's just some multiplication and addition.

# Plan:
#   - Training pipeline:
#     - Use the current noise generation implementation
#     - First: Keep the image in [0, 1].
#       - Run forward, add noise (keep current noise implementation parameters).
#     - After: Use a same scaling for all images, e.g. add 0 to measurements and multiply measurements by 1e-5
#       - The scaling happens after the measurements are simulated. I.e., scaling on the measurements.
#       - If noise is added to image rather than measurements, run forward then add noise.
#         Otherwise add noise to the measurements.
#       - After adding noise, run forward to get image if not already image,
#         and rescale to get images in ~[0, 1] before inputting to the models/denoisers.
#     - Augmentation: x8 rotations+flips, random translation (at most 128 pixels in each direction)
#   - Inference pipeline:
#     - Input: measurements, addition term, multiplication term.
#       - Or simply measurements, if assuming already scaled so that image ~[0. 1].
#         - In practice, perhaps:
#           - takes some coarse measurements,
#           - run FWHT,
#           - if image contrast doesn't look good, choose clipping points and whether to flip sign
#             (use some sliders to clip the min and max, toggle the sign flip option, and manually decide once)
#           - automatically compute the addition term and multiplication term for the image,
#             convert them to measurement scale,
#             then scale every measurement before passing to the model
#     - Output: image/current map
#   - Sanity check: Run inference on the real data example.
#     - Scale the real data myself: Just use the min and max values to scale to [0, 1].
DEFAULT_PCM_RANGES = {
    "CIGS": {
        100: [
            (-4.780e-7, -3.440e-7),
        ],
        512: [
            # [min, max]: [-2.068e-2, 6.995e-2]~[-2e-2, 7e-2]  ~  adds 5e-2 then times 5 --> becomes ~[0.15, 0.60]
            # [min, max]: [-2.068e-2, 6.995e-2]  ~  [-1e-1, 1e-1] or [-5e-2, 5e-2] ?
            # [max of min, min of max]: [-1.977e-4, 1.768e-3]  ~  times 5e2 ?
            (-5.077e-4, 6.995e-2),
            (-1.977e-4, 3.633e-3),
            (-2.068e-2, 2.839e-3),
            (-1.778e-2, 2.803e-3),
            (-3.803e-4, 1.768e-3),
            (-3.433e-3, 1.795e-3),
        ],
    },
    "Organic": {
        512: [
            # [min, max]: [-5.962e-3, 3.235e-3]~[-6e-3, 3e-3]  ~  adds 1e-2 then times 50 --> becomes ~[0.2, 0.65]
            # [min, max]: [-5.962e-3, 3.235e-3]  ~  [-1e-2, 1e-2] or [-5e-3, 5e-3] ?
            (-5.339e-3, 1.068e-3),
            (-5.962e-3, 1.902e-3),
            (-2.244e-3, 3.235e-3),
            (-1.046e-3, 2.174e-4),
        ],
    },
    "Perovskite": {
        100: [  # [min, max]: [-2.870e-7, 2.380e-7]  ~  [-1e-6, 1e-6] or [-5e-7, 5e-7] ?
            (4.100e-8, 2.380e-7),
            (1.800e-8, 1.210e-7),
            (2.100e-8, 1.290e-7),
            (1.000e-8, 1.140e-7),
            (-2.870e-7, -1.320e-7),
        ],
        256: [
            (-1.674e-5, -1.466e-5),
        ],
    },
    "Reference": {
        512: [  # [min, max]: [-2.534e-5, 6.429e-6]  ~  [-1e-4, 1e-5] or [-5e-5, 5e-6] ?  Min and max have different magnitudes?
            (-2.501e-5, 6.429e-6),
            (-1.249e-5, 2.117e-6),
            (-2.534e-5, 5.145e-6),
        ],
    },
    "Silicone": {
        512: [  # [min, max]: [-1.359e-1, 1.364e-1]  ~  [-1, 1] or [-1e-1, 1e-1] ?
            (-1.359e-1, 1.364e-1),
            (-1.540e-2, 1.294e-1),
            (-1.018e-1, 9.083e-3),
        ],
    },
}


class SoftPCMScaler:
    r"""
    Soft scaler from normalized synthetic images to approximate PCM values.

    Each reconstructed pixel is assumed to represent its contribution to the
    total photocurrent.

    If a real PCM range was measured at resolution N_ref x N_ref and the
    synthetic image has resolution N x N, then the per-pixel current scales as

        s = (N_ref / N)^2.

    Hence,

        x_min_N = s * x_min_ref,
        x_max_N = s * x_max_ref.

    A soft perturbation is then applied. For a range [x_min, x_max],

        c = (x_min + x_max) / 2,
        h = (x_max - x_min) / 2,

    and

        c_tilde = c + eps_c * h,
        h_tilde = h * exp(eps_h),

    where

        eps_c ~ N(0, center_jitter^2),
        eps_h ~ N(0, range_jitter^2).

    Finally, for normalized image z in [0, 1],

        x = x_min_tilde + z * (x_max_tilde - x_min_tilde).

    An RNG passed to __call__ takes priority over the RNG stored in the object.
    """

    def __init__(
        self,
        ranges,
        center_jitter=0.10,
        range_jitter=0.15,
        rng=None,
    ):
        self.ranges = ranges
        self.center_jitter = float(center_jitter)
        self.range_jitter = float(range_jitter)
        self.rng = np.random.default_rng() if rng is None else rng

    def _select_reference_resolution(self, target_size):
        r"""
        Select the available real-data resolution closest to target_size:

            N_ref = argmin_N |log2(N / target_size)|.
        """
        available = self.ranges.keys()

        return min(available, key=lambda n: abs(np.log2(n / target_size)))

    def _sample_range(self, target_size, rng):
        r"""
        Sample a physical PCM range for a target resolution.

        First select a real range measured at N_ref x N_ref. Then account for
        pixel area using

            scale = (N_ref / N_target)^2.

        Therefore,

            [x_min, x_max]
                -> scale * [x_min, x_max].

        A small random perturbation of the centre and width is applied after
        this resolution correction.
        """
        reference_size = self._select_reference_resolution(target_size)

        samples = np.asarray(self.ranges[reference_size], dtype=float)

        # Pick one observed real range.
        idx = rng.integers(len(samples))
        x_min, x_max = samples[idx]

        # Pixel contribution scales with pixel area.
        resolution_scale = (reference_size / target_size) ** 2

        x_min *= resolution_scale
        x_max *= resolution_scale

        # Softly perturb the selected range.
        center = 0.5 * (x_min + x_max)
        half_range = 0.5 * (x_max - x_min)

        center += rng.normal(0.0, self.center_jitter) * half_range

        half_range *= np.exp(rng.normal(0.0, self.range_jitter))

        x_min = center - half_range
        x_max = center + half_range

        return x_min, x_max, reference_size, resolution_scale

    def __call__(self, image, rng=None, clip=False, return_info=False):
        r"""
        Convert a normalized image to approximate physical PCM values.

        The image resolution N is inferred directly from image.shape.

        For z in [0, 1],

            x = x_min + z * (x_max - x_min).

        Parameters
        ----------
        image : array-like
            Square normalized PCM image.

        rng : numpy.random.Generator, optional
            RNG for this call. Takes priority over the stored RNG.

        clip : bool
            Clip image values to [0, 1] before scaling.

        return_info : bool
            Also return information about the sampled physical range.
        """
        image = np.asarray(image, dtype=float)

        if image.shape[-2] != image.shape[-1]:
            raise ValueError(
                "Expected a square PCM image, " f"but received shape {image.shape}."
            )

        target_size = image.shape[-1]
        active_rng = self.rng if rng is None else rng

        if clip:
            image = np.clip(image, 0.0, 1.0)

        x_min, x_max, reference_size, resolution_scale = self._sample_range(
            target_size, active_rng
        )

        scaled = x_min + image * (x_max - x_min)

        if return_info:
            return scaled, {
                "target_resolution": target_size,
                "reference_resolution": reference_size,
                "resolution_scale": resolution_scale,
                "x_min": x_min,
                "x_max": x_max,
            }

        return scaled


# %%
# cigs_scaler = SoftPCMScaler("CIGS", rng=np.random.default_rng(123))

for category in DEFAULT_PCM_RANGES:
    scaler = SoftPCMScaler(
        ranges=DEFAULT_PCM_RANGES[category],
        center_jitter=0.10,
        range_jitter=0.15,
        rng=np.random.default_rng(123),
    )

    png_image = np.random.rand(256, 256)
    # png_image = np.random.rand(512, 512)

    scaled_image, info = scaler(png_image, return_info=True)

    print(f"Category: {category}")
    pprint(info)
    print(f"Scaled image range: [{scaled_image.min():.2e}, {scaled_image.max():.2e}]")
    print(f"Scaled image shape: {scaled_image.shape}")
    print()

    call_rng = np.random.default_rng(999)
    scaled_image, info = scaler(png_image, rng=call_rng, return_info=True)

# Category: CIGS
# Scaled image range: [-3.72e-02, 3.05e-01]
# Scaled image shape: (256, 256)

# Category: Organic
# Scaled image range: [-2.46e-02, 6.53e-03]
# Scaled image shape: (256, 256)

# Category: Perovskite
# Scaled image range: [-1.68e-05, -1.48e-05]
# Scaled image shape: (256, 256)

# Category: Reference
# Scaled image range: [-1.16e-04, 3.68e-05]
# Scaled image shape: (256, 256)

# Category: Silicone
# Scaled image range: [-6.80e-01, 6.42e-01]
# Scaled image shape: (256, 256)
