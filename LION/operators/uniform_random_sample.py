from __future__ import annotations

import numpy as np

from LION.operators.sampling_utils import get_random_indices_up_to_last_coarse_level


def uniform_random_sample(
    *,
    J: int,
    num_samples: int,
    coarse_J: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    assert J > 0, f"J must be positive, got {J}"
    num_pixels = 1 << (2 * J)
    assert (
        0 < num_samples <= num_pixels
    ), f"Number of samples must be in (0, {num_pixels}], got {num_samples}"
    assert (
        coarse_J <= J
    ), f"coarse_J must be less than or equal to J, got coarse_J={coarse_J}, J={J}"
    num_coarse_samples = 1 << (2 * coarse_J)

    if num_coarse_samples > num_samples:
        return get_random_indices_up_to_last_coarse_level(
            J=J,
            num_samples=num_samples,
            coarse_J=coarse_J,
            num_coarse_samples=num_coarse_samples,
            rng=rng,
        )

    num_random_samples = num_samples - num_coarse_samples
    if num_random_samples > 0:
        if rng is None:
            rng = np.random.default_rng()
        random_tail_indices = (
            rng.choice(
                num_pixels - num_coarse_samples, size=num_random_samples, replace=False
            )
            + num_coarse_samples
        )
        sampled_pattern_indices = np.concatenate(
            [
                np.arange(num_coarse_samples, dtype=np.int64),
                random_tail_indices.astype(np.int64),
            ]
        )
    else:
        sampled_pattern_indices = np.arange(num_coarse_samples, dtype=np.int64)

    return sampled_pattern_indices
