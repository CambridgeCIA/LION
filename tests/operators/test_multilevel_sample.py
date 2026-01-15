from __future__ import annotations

import numpy as np
import pytest

# Adjust this import to match the actual module path
from LION.operators.multilevel_sample import multilevel_sample


def test_basic_properties() -> None:
    J = 4
    num_samples = 32
    rng = np.random.default_rng(123)

    idx = multilevel_sample(J=J, num_samples=num_samples, coarse_J=0, alpha=1.0, rng=rng)

    assert isinstance(idx, np.ndarray)
    assert idx.shape == (num_samples,)
    assert np.issubdtype(idx.dtype, np.integer)

    N = 4**J
    assert int(idx.min()) >= 0
    assert int(idx.max()) < N
    assert np.unique(idx).size == num_samples


def test_coarse_block_is_fully_included() -> None:
    J = 5
    coarse_J = 2
    keep = 4**coarse_J  # first 2^coarse_J-by-2^coarse_J indices in coarse-to-fine ordering
    num_samples = keep + 20

    rng = np.random.default_rng(0)
    idx = multilevel_sample(J=J, num_samples=num_samples, coarse_J=coarse_J, alpha=1.0, rng=rng)

    assert np.isin(np.arange(keep), idx).all()


def test_num_samples_equal_keep_returns_exact_coarse_block() -> None:
    J = 6
    coarse_J = 3
    keep = 4**coarse_J
    num_samples = keep

    rng = np.random.default_rng(0)
    idx = multilevel_sample(J=J, num_samples=num_samples, coarse_J=coarse_J, alpha=1.0, rng=rng)

    assert np.array_equal(np.sort(idx), np.arange(keep, dtype=int))


def test_coarse_J_equal_J_can_return_all_indices() -> None:
    J = 4
    coarse_J = 4
    N = 4**J
    num_samples = N

    rng = np.random.default_rng(0)
    idx = multilevel_sample(J=J, num_samples=num_samples, coarse_J=coarse_J, alpha=2.0, rng=rng)

    assert np.array_equal(np.sort(idx), np.arange(N, dtype=int))


def test_num_samples_zero_returns_empty_array() -> None:
    rng = np.random.default_rng(0)
    idx = multilevel_sample(J=4, num_samples=0, coarse_J=0, alpha=1.0, rng=rng)

    assert isinstance(idx, np.ndarray)
    assert idx.size == 0
    assert np.issubdtype(idx.dtype, np.integer)


def test_deterministic_given_same_seed() -> None:
    J = 6
    num_samples = 200

    idx1 = multilevel_sample(J=J, num_samples=num_samples, coarse_J=2, alpha=1.0, rng=np.random.default_rng(42))
    idx2 = multilevel_sample(J=J, num_samples=num_samples, coarse_J=2, alpha=1.0, rng=np.random.default_rng(42))

    assert np.array_equal(idx1, idx2)


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        multilevel_sample(J=-1, num_samples=10, coarse_J=0)

    with pytest.raises(ValueError):
        multilevel_sample(J=4, num_samples=-1, coarse_J=0)

    with pytest.raises(ValueError):
        multilevel_sample(J=4, num_samples=300, coarse_J=0)  # 4**4 = 256

    with pytest.raises(ValueError):
        multilevel_sample(J=4, num_samples=10, coarse_J=-1)

    with pytest.raises(ValueError):
        multilevel_sample(J=4, num_samples=10, coarse_J=5)  # coarse_J > J

    with pytest.raises(ValueError):
        # If keep = 4**2 = 16 but only 10 requested, it is still OK
        #   because 4**(2-1) = 4 <= 10.
        #   The remaining 6 samples are drawn from level 2.
        # But if keep = 4**3 = 64 and only 10 requested, it is invalid
        #   because 4**(3-1) = 16 > 10.
        multilevel_sample(J=4, num_samples=10, coarse_J=3)
