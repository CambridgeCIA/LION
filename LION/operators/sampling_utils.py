import numpy as np


def get_random_indices_up_to_last_coarse_level(
    *,
    J: int,
    num_samples: int,
    coarse_J: int,
    num_coarse_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    assert num_coarse_samples > num_samples, (
        "num_coarse_samples must be greater than num_samples to use this function. "
        f"Got num_coarse_samples={num_coarse_samples}, num_samples={num_samples}."
    )
    assert coarse_J > 0, (
        "coarse_J must be positive if num_coarse_samples > num_samples. "
        f"Got coarse_J={coarse_J}"
    )
    num_full_coarse_samples = 1 << (2 * (coarse_J - 1))
    if num_full_coarse_samples >= num_samples:
        raise ValueError(
            f"Number of full coarse samples must be less than num_samples. "
            f"Got num_full_coarse_samples={num_full_coarse_samples}, num_samples={num_samples}. "
            f"Consider reducing coarse_J (currently coarse_J={coarse_J} with J={J} and {num_samples} samples). "
        )
    remaining_coarse_indices_pool_size = num_coarse_samples - num_full_coarse_samples
    remaining_coarse_indices_pool = np.arange(remaining_coarse_indices_pool_size, dtype=np.int64) + num_full_coarse_samples
    num_remaining_coarse_samples = num_samples - num_full_coarse_samples
    if rng is None:
        rng = np.random.default_rng()
    selected_remaining_coarse_indices = rng.choice(
        remaining_coarse_indices_pool,
        size=num_remaining_coarse_samples,
        replace=False,
    )
    sampled_pattern_indices = np.concatenate(
        [np.arange(num_full_coarse_samples, dtype=np.int64), selected_remaining_coarse_indices.astype(np.int64)]
    )
    return sampled_pattern_indices
