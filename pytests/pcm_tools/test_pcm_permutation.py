"""Tests for PCM WHT ordering permutations."""

import numpy as np
import pytest
from LION.operators.PhotocurrentMapOp import normal_to_dyadic_permutation
from LION.pcm_tools.noise_utils import generate_order_sorting_matrix


@pytest.mark.parametrize("j_order", range(1, 9))
def test_square_order_sorting_matches_normal_to_dyadic(j_order: int) -> None:
    """The two PCM permutation implementations agree for square 2^J x 2^J images."""
    expected = normal_to_dyadic_permutation(j_order)
    actual = generate_order_sorting_matrix(
        o_col=j_order,
        o_row=j_order,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("j_order", "expected"),
    [
        (1, np.array([0, 2, 1, 3], dtype=np.int64)),
        (
            2,
            np.array(
                [
                    0,
                    8,
                    2,
                    10,
                    4,
                    12,
                    6,
                    14,
                    1,
                    9,
                    3,
                    11,
                    5,
                    13,
                    7,
                    15,
                ],
                dtype=np.int64,
            ),
        ),
    ],
)
def test_normal_to_dyadic_known_values(
    j_order: int,
    expected: np.ndarray,
) -> None:
    """Check small permutations against explicit known values."""
    np.testing.assert_array_equal(
        normal_to_dyadic_permutation(j_order),
        expected,
    )
    np.testing.assert_array_equal(
        generate_order_sorting_matrix(
            o_col=j_order,
            o_row=j_order,
        ),
        expected,
    )


@pytest.mark.parametrize("j_order", range(1, 9))
def test_square_order_sorting_is_a_permutation(j_order: int) -> None:
    """The square PCM ordering contains every index exactly once."""
    permutation = generate_order_sorting_matrix(
        o_col=j_order,
        o_row=j_order,
    )
    expected = np.arange(1 << (2 * j_order), dtype=np.int64)

    np.testing.assert_array_equal(
        np.sort(permutation),
        expected,
    )
