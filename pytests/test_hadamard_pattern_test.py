import numpy as np
from LION.pcm_tools.hadamard_matrix import (
    build_reference_natural_matrix,
    build_sorted_matrix_from_patterns,
    reorder_natural_to_sorted,
    reorder_sorted_to_natural,
)

from pcm_helper import check_orthogonality, compare_rows_up_to_sign


def run_test(o_row: int, o_col: int) -> None:
    """
    Run all consistency tests.

    Parameters
    ----------
    o_row : int
        Base-2 order for rows.
    o_col : int
        Base-2 order for columns.
    """
    print(f"Testing o_row={o_row}, o_col={o_col}")

    S, perm = build_sorted_matrix_from_patterns(o_row=o_row, o_col=o_col)
    N_ref = build_reference_natural_matrix(o_row=o_row, o_col=o_col)

    n = S.shape[0]
    ar = np.arange(n, dtype=np.int64)

    # 1. Permutation / inverse-permutation checks
    inv_perm = np.argsort(perm)
    print("\nPermutation checks:")
    print("  perm is a permutation:", np.array_equal(np.sort(perm), ar))
    print("  inv_perm[perm] == arange(n):", np.array_equal(inv_perm[perm], ar))
    print("  perm[inv_perm] == arange(n):", np.array_equal(perm[inv_perm], ar))

    # 2. Round-trip checks
    N_from_S = reorder_sorted_to_natural(S, perm)
    S_roundtrip = reorder_natural_to_sorted(N_from_S, perm)

    print("\nRound-trip checks:")
    print("  sorted -> natural -> sorted:", np.array_equal(S_roundtrip, S))

    # 3. Compare against Kronecker reference
    print("\nComparison against Kronecker reference:")
    exact_match = np.array_equal(N_from_S, N_ref)
    print("  exact matrix match:", exact_match)

    ok_up_to_sign, signs = compare_rows_up_to_sign(N_from_S, N_ref)
    print("  rowwise match up to global sign:", bool(np.all(ok_up_to_sign)))

    if not exact_match:
        mismatch_rows = np.where(~np.all(N_from_S == N_ref, axis=1))[0]
        print("  exact mismatch rows:", mismatch_rows.tolist())

    if not np.all(ok_up_to_sign):
        bad_rows = np.where(~ok_up_to_sign)[0]
        print("  rows that do not match even up to sign:", bad_rows.tolist())
    else:
        neg_rows = np.where(signs == -1)[0]
        print("  rows that match only after multiplying by -1:", neg_rows.tolist())

    # 4. Hadamard orthogonality checks
    print("\nOrthogonality checks:")
    print("  sorted matrix is Hadamard:", check_orthogonality(S.astype(np.int64)))
    print(
        "  natural-from-sorted is Hadamard:",
        check_orthogonality(N_from_S.astype(np.int64)),
    )
    print(
        "  Kronecker reference is Hadamard:",
        check_orthogonality(N_ref.astype(np.int64)),
    )

    # 5. Inspect the suspicious rows
    for idx in [14, 15]:
        if idx < n:
            print(f"\nInspecting natural row {idx}:")
            same = np.array_equal(N_from_S[idx], N_ref[idx])
            neg = np.array_equal(N_from_S[idx], -N_ref[idx])
            print("  exact match:", same)
            print("  negative match:", neg)
            print("  generated:", N_from_S[idx].tolist())
            print("  reference:", N_ref[idx].tolist())


if __name__ == "__main__":
    run_test(o_row=2, o_col=2)
