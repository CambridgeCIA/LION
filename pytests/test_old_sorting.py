from __future__ import annotations

import numpy as np
from LION.operators.PhotocurrentMapOp import normal_to_dyadic_permutation
from LION.pcm_tools.hadamard_matrix import (
    build_reference_natural_matrix,
    hadamard_matrix,
    reorder_natural_to_sorted,
    reorder_sorted_to_natural,
)
from LION.pcm_tools.noise_utils import generate_order_sorting_matrix

from pcm_helper import compare_rows_up_to_sign


def test_permutation_equivalence(max_J: int = 8) -> None:
    """Compare the two permutation generators for square sizes.

    Args:
        max_J: Maximum J to test.
    """
    for J in range(max_J + 1):
        perm_old = generate_order_sorting_matrix(o_col=J, o_row=J)
        perm_new = normal_to_dyadic_permutation(J)

        same = np.array_equal(perm_old, perm_new)

        print(f"J={J:2d}  n={perm_old.size:6d}  exact_match={same}")

        if not same:
            mismatch = np.where(perm_old != perm_new)[0]
            print(f"  mismatch indices: {mismatch.tolist()}")
            print(f"  old values      : {perm_old[mismatch].tolist()}")
            print(f"  new values      : {perm_new[mismatch].tolist()}")
            raise AssertionError(f"Mismatch found for J={J}")

    print("All square-case permutation tests passed.")


def build_sorted_matrix_from_perm(perm: np.ndarray, J: int) -> np.ndarray:
    """Build the sorted-order matrix from a permutation.

    Args:
        perm: Permutation mapping sorted_index -> natural_index.
        J: Base-2 image order.

    Returns:
        Sorted-order matrix whose rows are flattened 2D patterns.
    """
    H = hadamard_matrix(J)
    n_side = 1 << J
    n_patterns = n_side * n_side

    S = np.empty((n_patterns, n_patterns), dtype=np.int8)

    for sorted_index in range(n_patterns):
        natural_index = int(perm[sorted_index])
        row_index = natural_index // n_side
        col_index = natural_index % n_side

        pattern_2d = np.outer(H[row_index], H[col_index]).astype(np.int8, copy=False)
        S[sorted_index] = pattern_2d.reshape(-1)

    return S


def run_square_case(J: int, verbose: bool = False) -> None:
    """Run all tests for one square case.

    Args:
        J: Base-2 order in both directions.
        verbose: Whether to print detailed output.
    """
    perm_old = generate_order_sorting_matrix(o_col=J, o_row=J)
    perm_new = normal_to_dyadic_permutation(J)

    if verbose:
        print(f"\nTesting J={J}, n={(1 << (2 * J))}")
        print("  permutation exact match:", np.array_equal(perm_old, perm_new))

    np.testing.assert_array_equal(perm_old, perm_new)

    S_old = build_sorted_matrix_from_perm(perm_old, J)
    S_new = build_sorted_matrix_from_perm(perm_new, J)
    np.testing.assert_array_equal(S_old, S_new)

    N_old = reorder_sorted_to_natural(S_old, perm_old)
    N_new = reorder_sorted_to_natural(S_new, perm_new)
    N_ref = build_reference_natural_matrix(o_col=J, o_row=J)

    np.testing.assert_array_equal(N_old, N_new)

    ok, signs = compare_rows_up_to_sign(N_old, N_ref)
    if verbose:
        print("  natural vs kron exact match:", np.array_equal(N_old, N_ref))
        print("  natural vs kron up to sign :", bool(np.all(ok)))
        if np.all(ok):
            neg_rows = np.where(signs == -1)[0]
            print("  negative rows vs kron      :", neg_rows.tolist())

    np.testing.assert_array_equal(ok, np.ones_like(ok, dtype=bool))

    S_roundtrip = reorder_natural_to_sorted(N_old, perm_old)
    np.testing.assert_array_equal(S_roundtrip, S_old)


def test_many_square_cases(max_J: int = 3) -> None:
    """Test many square cases.

    Args:
        max_J: Maximum J to test.
    """
    for J in range(max_J + 1):
        run_square_case(J, verbose=True)
    print("\nAll square-case tests passed.")
