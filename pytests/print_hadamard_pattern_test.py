import numpy as np
from LION.pcm_tools.hadamard_matrix import (
    build_reference_natural_matrix,
    build_sorted_matrix_from_patterns,
    reorder_natural_to_sorted,
    reorder_sorted_to_natural,
)

from pcm_helper import check_orthogonality, compare_rows_up_to_sign


def format_plus_minus_matrix(M: np.ndarray, plus: str = "+", minus: str = "-") -> str:
    """
    Format a {-1, +1} matrix for readable terminal printing.

    Parameters
    ----------
    M : np.ndarray
        Input matrix with entries in {-1, +1}.
    plus : str, optional
        Symbol for +1.
    minus : str, optional
        Symbol for -1.

    Returns
    -------
    str
        Formatted multiline string.
    """
    rows = []
    for row in M:
        rows.append(" ".join(plus if v > 0 else minus for v in row))
    return "\n".join(rows)


def print_full_matrices(S: np.ndarray, N: np.ndarray, N_ref: np.ndarray) -> None:
    """
    Print the sorted, reordered-natural, and reference-natural matrices.

    Parameters
    ----------
    S : np.ndarray
        Sorted-order matrix.
    N : np.ndarray
        Natural-order matrix obtained by reordering S.
    N_ref : np.ndarray
        Natural-order Kronecker reference matrix.
    """
    print("\n=== Sorted-order matrix S ===")
    print(format_plus_minus_matrix(S))

    print("\n=== Natural-order matrix N = reorder_sorted_to_natural(S) ===")
    print(format_plus_minus_matrix(N))

    print("\n=== Reference natural-order matrix N_ref = kron(H_row, H_col) ===")
    print(format_plus_minus_matrix(N_ref))


def run_test(o_row: int, o_col: int, *, print_matrices: bool = True) -> None:
    """
    Run consistency tests and optionally print all matrices.

    Parameters
    ----------
    o_row : int
        Base-2 order for rows.
    o_col : int
        Base-2 order for columns.
    print_matrices : bool, optional
        If True, print the full matrices.
    """
    print(f"Testing o_row={o_row}, o_col={o_col}")

    S, perm = build_sorted_matrix_from_patterns(o_row=o_row, o_col=o_col)
    N = reorder_sorted_to_natural(S, perm)
    N_ref = build_reference_natural_matrix(o_row=o_row, o_col=o_col)

    n = S.shape[0]
    ar = np.arange(n, dtype=np.int64)
    inv_perm = np.argsort(perm)

    print("\nPermutation checks:")
    print("  perm is a permutation:", np.array_equal(np.sort(perm), ar))
    print("  inv_perm[perm] == arange(n):", np.array_equal(inv_perm[perm], ar))
    print("  perm[inv_perm] == arange(n):", np.array_equal(perm[inv_perm], ar))

    print("\nRound-trip checks:")
    print(
        "  sorted -> natural -> sorted:",
        np.array_equal(reorder_natural_to_sorted(N, perm), S),
    )
    print(
        "  natural -> sorted -> natural:",
        np.array_equal(
            reorder_sorted_to_natural(reorder_natural_to_sorted(N_ref, perm), perm),
            N_ref,
        ),
    )

    print("\nMatrix checks:")
    print("  exact match N == N_ref:", np.array_equal(N, N_ref))

    ok_up_to_sign, signs = compare_rows_up_to_sign(N, N_ref)
    print("  rowwise match up to sign:", bool(np.all(ok_up_to_sign)))

    if np.all(ok_up_to_sign):
        neg_rows = np.where(signs == -1)[0]
        print("  rows matching only after multiplying by -1:", neg_rows.tolist())
    else:
        bad_rows = np.where(~ok_up_to_sign)[0]
        print("  rows not matching even up to sign:", bad_rows.tolist())

    print("\nOrthogonality checks:")
    print("  S is Hadamard:", check_orthogonality(S))
    print("  N is Hadamard:", check_orthogonality(N))
    print("  N_ref is Hadamard:", check_orthogonality(N_ref))

    if print_matrices:
        print_full_matrices(S, N, N_ref)

    for idx in [14, 15]:
        if idx < n:
            print(f"\nInspecting natural row {idx}:")
            print("  generated :", " ".join("+" if v > 0 else "-" for v in N[idx]))
            print("  reference :", " ".join("+" if v > 0 else "-" for v in N_ref[idx]))
            print("  exact     :", np.array_equal(N[idx], N_ref[idx]))
            print("  negative  :", np.array_equal(N[idx], -N_ref[idx]))


if __name__ == "__main__":
    run_test(o_row=2, o_col=2, print_matrices=True)
