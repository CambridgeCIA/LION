"""Sampling utilities for photocurrent mapping (PCM) reconstruction."""

from __future__ import annotations

import numpy as np


def get_random_indices_up_to_last_coarse_level(
    *,
    J: int,
    num_samples: int,
    coarse_J: int,
    num_coarse_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Helper function to get random indices up to the last coarse level."""
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
    remaining_coarse_indices_pool = (
        np.arange(remaining_coarse_indices_pool_size, dtype=np.int64)
        + num_full_coarse_samples
    )
    num_remaining_coarse_samples = num_samples - num_full_coarse_samples
    if rng is None:
        rng = np.random.default_rng()
    selected_remaining_coarse_indices = rng.choice(
        remaining_coarse_indices_pool,
        size=num_remaining_coarse_samples,
        replace=False,
    )
    sampled_pattern_indices = np.concatenate(
        [
            np.arange(num_full_coarse_samples, dtype=np.int64),
            selected_remaining_coarse_indices.astype(np.int64),
        ]
    )
    return sampled_pattern_indices


def multilevel_sample(
    *,
    J: int,
    num_samples: int,
    coarse_J: int = 0,
    alpha: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""
    Sample indices from N = 4^J patterns arranged in coarse-to-fine blocks.

    Block sizes are:
        level 0: 1
        level 1: 3
        level 2: 12
        level 3: 48
        ...
    i.e. size_l = 3 * 4^(l-1) for l>=1, and size_0 = 1.

    Sampling:
        - allocate samples per level with weight 2^(-alpha*l) so larger alpha favours coarse levels
        - sample uniformly within each level without replacement

    Parameters
    ----------
    J:
        Image is (2^J)-by-(2^J), so total patterns N = 4^J.
    num_samples:
        Number of indices to return, 0 <= num_samples <= N.
    coarse_J:
        Keep (fully include) the first 2^coarse_J-by-2^coarse_J block of indices in the
        coarse-to-fine ordering, i.e. the first 4^coarse_J indices. Equivalently, fully
        sample levels 0..coarse_J.
    alpha:
        Decay exponent in weight 2^(-alpha*l); larger means stronger bias toward coarse scales.
    rng:
        Random number generator. If None, a new default generator is created.

    Returns
    -------
    indices:
        Array of chosen indices in [0, 4^J - 1].

    Notes
    -----
    This routine implements a simple multilevel (coarse-to-fine) variable-density
    sampling rule. Levels grow geometrically in size (approximately |Omega_l| ~ 4^l),
    and an expected sample count per level is allocated as

        q_l ∝ 2^(-alpha * l) * |Omega_l|,

    followed by uniform sampling without replacement within each level.

    Notes of the choice of ``alpha``:

    - alpha = 0:
    Gives approximately uniform sampling over all coefficients/patterns, since the
    per-coefficient inclusion probability satisfies pi_l ≈ q_l / |Omega_l| ∝ 2^(-alpha*l),
    hence pi_l is constant in l when alpha = 0.

    - alpha = 2:
    Gives approximately the same expected number of samples per level, since
    q_l ∝ 2^(-alpha*l) * 4^l = 2^((2 - alpha) * l), which is roughly constant in l
    when alpha = 2.

    Relation to the literature:

    - Krahmer and Ward (2014) analyse power-law (variable-density) sampling strategies
    for compressive imaging and highlight an inverse-square style decay as a
    theory-backed choice, which corresponds to alpha ≈ 2 in the parametrisation used
    here.

    - Adcock et al. (2017) formalise multilevel random sampling by specifying level
    boundaries and per-level sample counts, rather than prescribing a single default
    exponent alpha.

    - Tsaig and Donoho (2006) discuss multiscale ideas for compressed sensing with
    structured transforms, without prescribing a single default alpha.

    Examples
    --------
    Consider ``J = 4`` so the total number of patterns is ``N = 4^J = 256`` and take
    ``num_samples = 32``. The multilevel partition (coarse to fine) has sizes::

        level 0: 1
        level 1: 3
        level 2: 12
        level 3: 48
        level 4: 192

    Case ``alpha = 0`` (no coarse bias)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    With ``alpha = 0``, the per-level weights satisfy

    .. math::

        q_\ell \propto 2^{-\alpha \ell} |\Omega_\ell| = |\Omega_\ell|,

    so every coefficient/pattern has the same inclusion probability

    .. math::

        \pi = \frac{32}{256} = 0.125.

    The expected samples per level (proportional to level size) are

    .. math::

        \mathbb{E}[m_\ell] = 32 \cdot \frac{|\Omega_\ell|}{256}
        = [0.125,\ 0.375,\ 1.5,\ 6,\ 24].

    A typical integer allocation is therefore close to ``[0, 0, 2, 6, 24]``.

    Case ``alpha = 2`` (coarse-favouring)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    With ``alpha = 2``, the per-level weights satisfy

    .. math::

        q_\ell \propto 2^{-2\ell} |\Omega_\ell|.

    Since :math:`|\Omega_\ell| \approx 4^\ell`, this makes

    .. math::

        q_\ell \propto 2^{-2\ell} 4^\ell = 2^{(2 - 2)\ell} = 1,

    so (before integer rounding and capacity constraints) the allocation tends to be
    roughly balanced across levels. Because the coarsest levels are very small (size
    1 and 3), no-replacement sampling often fully includes them (1 and 3 samples),
    and the remaining samples are distributed across the finer levels.

    References
    ----------
    .. [1] Y. Tsaig and D. L. Donoho, "Extensions of compressed sensing,"
        Signal Processing, 86(3):549-571, 2006. doi:10.1016/j.sigpro.2005.05.029.
    .. [2] F. Krahmer and R. Ward, "Stable and Robust Sampling Strategies for
        Compressive Imaging," IEEE Transactions on Image Processing, 23(2):612-622,
        2014. doi:10.1109/TIP.2013.2288004.
    .. [3] B. Adcock, A. C. Hansen, C. Poon, and B. Roman, "Breaking the coherence
        barrier: A new theory for compressed sensing," Forum of Mathematics, Sigma,
        5:e4, 2017. doi:10.1017/fms.2016.32.
    """
    if J < 0:
        raise ValueError("J must be non-negative.")
    if not (0 <= coarse_J <= J):
        raise ValueError(f"coarse_J must satisfy 0 <= coarse_J <= {J}, got {coarse_J}.")
    N = 4**J
    if not (0 <= num_samples <= N):
        raise ValueError(f"num_samples must be in [0, {N}].")
    if num_samples == 0:
        return np.empty((0,), dtype=int)

    if rng is None:
        rng = np.random.default_rng()

    keep = 4**coarse_J
    if keep > num_samples:
        return get_random_indices_up_to_last_coarse_level(
            J=J,
            num_samples=num_samples,
            coarse_J=coarse_J,
            num_coarse_samples=keep,
            rng=rng,
        )

    # Level sizes for l = 0..J, summing to 4^J
    sizes = np.array([1] + [3 * (4 ** (l - 1)) for l in range(1, J + 1)], dtype=int)

    # Level start offsets
    starts = np.concatenate(([0], np.cumsum(sizes[:-1])))

    # Force-include levels 0..coarse_J
    k = np.zeros(J + 1, dtype=int)
    k[: coarse_J + 1] = sizes[: coarse_J + 1]

    # Allocate remaining counts per level using decaying weights, only for levels > coarse_J
    remaining = num_samples - int(k.sum())
    if remaining > 0:
        levels = np.arange(J + 1)
        w = (2.0 ** (-alpha * levels)) * sizes
        w[: coarse_J + 1] = 0.0

        q = remaining * (w / w.sum())  # desired (non-integer) extra counts per level

        extra = np.floor(q).astype(int)
        extra = np.minimum(extra, sizes - k)
        k += extra

        # Fix rounding so sum(k) == num_samples (only adjust levels > coarse_J)
        remaining2 = num_samples - int(k.sum())
        if remaining2 > 0:
            frac = q - np.floor(q)
            order = np.argsort(-frac)
            for l in order:
                if l <= coarse_J:
                    continue
                if remaining2 == 0:
                    break
                if k[l] < sizes[l]:
                    k[l] += 1
                    remaining2 -= 1
        elif remaining2 < 0:
            order = np.argsort(levels)[::-1]  # fine to coarse
            for l in order:
                if l <= coarse_J:
                    continue
                if remaining2 == 0:
                    break
                take = min(k[l], -remaining2)
                k[l] -= take
                remaining2 += take

    # Sample within each level
    out = []
    for l in range(J + 1):
        if k[l] == 0:
            continue
        if l <= coarse_J:
            offsets = np.arange(sizes[l], dtype=int)  # keep all
        else:
            offsets = rng.choice(sizes[l], size=k[l], replace=False)
        out.append(starts[l] + offsets)

    indices = np.concatenate(out).astype(int)
    indices.sort()
    return indices


def uniform_random_sample(
    *,
    J: int,
    num_samples: int,
    coarse_J: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample indices uniformly at random from the first 4^J patterns, with a fully included coarse block."""
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
