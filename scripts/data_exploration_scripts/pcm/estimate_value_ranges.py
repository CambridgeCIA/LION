# %%
import numpy as np

PCM_RANGES = {
    "CIGS": {
        100: [
            (-4.780e-7, -3.440e-7),
        ],
        512: [
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
            (-5.339e-3, 1.068e-3),
            (-5.962e-3, 1.902e-3),
            (-2.244e-3, 3.235e-3),
            (-1.046e-3, 2.174e-4),
        ],
    },
    "Perovskite": {
        100: [
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
        512: [
            (-2.501e-5, 6.429e-6),
            (-1.249e-5, 2.117e-6),
            (-2.534e-5, 5.145e-6),
        ],
    },
    "Silicone": {
        512: [
            (-1.359e-1, 1.364e-1),
            (-1.540e-2, 1.294e-1),
            (-1.018e-1, 9.083e-3),
        ],
    },
}


def estimate_pcm_range(category, target_size=256):
    r"""
    Estimate a rough physical photocurrent range for a synthetic PCM image.

    For category c and target resolution N, choose the available real-data
    resolution closest on a log2 scale:

        r* = argmin_r |log2(r / N)|.

    Using samples at resolution r*, estimate the lower and upper bounds as

        x_min_hat = median_i(x_min_i),

        x_max_hat = median_i(x_max_i).

    The median is used rather than the mean because some observed PCM ranges
    differ strongly between samples.

    Parameters
    ----------
    category : str
        One of "CIGS", "Organic", "Perovskite", "Reference", or "Silicone".

    target_size : int
        Width/height of the synthetic image. Default is 256.

    Returns
    -------
    x_min_hat : float
        Estimated physical minimum.

    x_max_hat : float
        Estimated physical maximum.

    source_size : int
        Real-data resolution used for the estimate.
    """
    ranges_by_size = PCM_RANGES[category]

    source_size = min(
        ranges_by_size,
        key=lambda r: abs(np.log2(r / target_size)),
    )

    ranges = np.asarray(ranges_by_size[source_size])

    x_min_hat = np.median(ranges[:, 0])
    x_max_hat = np.median(ranges[:, 1])

    return x_min_hat, x_max_hat, source_size


def normalized_to_pcm(image, category):
    r"""
    Convert a normalized synthetic image z in [0, 1] to an approximate
    physical PCM scale.

    The affine transformation is

        x = x_min_hat + z * (x_max_hat - x_min_hat),

    where x_min_hat and x_max_hat are estimated from real PCM samples of
    the corresponding category.

    Parameters
    ----------
    image : ndarray
        Normalized synthetic image with values in [0, 1].

    category : str
        PCM material category.

    Returns
    -------
    ndarray
        Approximate physical photocurrent values.
    """
    x_min, x_max, _ = estimate_pcm_range(category, target_size=256)

    return x_min + image * (x_max - x_min)


# %%
if __name__ == "__main__":
    categories = [
        "CIGS",
        "Organic",
        "Perovskite",
        "Reference",
        "Silicone",
    ]

    for category in categories:
        x_min, x_max, source_size = estimate_pcm_range(category, target_size=256)

        print(
            f"{category:12s}: "
            f"[{x_min:.3e}, {x_max:.3e}] "
            f"(using {source_size}x{source_size} data)"
        )

# CIGS        : [-1.970e-03, 2.821e-03] (using 512x512 data)
# Organic     : [-3.791e-03, 1.485e-03] (using 512x512 data)
# Perovskite  : [-1.674e-05, -1.466e-05] (using 256x256 data)
# Reference   : [-2.501e-05, 5.145e-06] (using 512x512 data)
# Silicone    : [-1.018e-01, 1.294e-01] (using 512x512 data)
