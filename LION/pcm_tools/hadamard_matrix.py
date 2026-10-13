from __future__ import annotations

from typing import Literal

import numpy as np

from LION.pcm_tools.noise_utils import generate_order_sorting_matrix


def hadamard_matrix(order: int) -> np.ndarray:
    """Return the Sylvester Hadamard matrix of size 2**order.

    Args:
        order: Base-2 order.

    Returns:
        Matrix of shape (2**order, 2**order) with entries in {-1, +1}.
    """
    H = np.array([[1]], dtype=np.int8)
    for _ in range(order):
        H = np.block([[H, H], [H, -H]])
    return H.astype(np.int8, copy=False)


def build_reference_natural_matrix(o_row: int, o_col: int) -> np.ndarray:
    """Build the natural-order 2D Hadamard matrix using a Kronecker product.

    Convention used here:
    - basis index order is row-major:
        `natural_index = row_index * n_cols + col_index`
    - pattern flattening is also row-major

    Then the full natural matrix is `kron(H_row, H_col)`
    where `kron` is the Kronecker product, and `H_row` and `H_col` are the 1D Hadamard matrices for rows and columns.

    Args:
        o_row: Base-2 order for rows.
        o_col: Base-2 order for columns.

    Returns:
        Matrix of shape (n_patterns, n_pixels).
    """
    H_row = hadamard_matrix(o_row)
    H_col = hadamard_matrix(o_col)
    return np.kron(H_row, H_col).astype(np.int8, copy=False)


def build_sorted_matrix_from_patterns(
    o_row: int, o_col: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build the sorted-order matrix by generating each sorted pattern explicitly.

    Args:
        o_row: Base-2 order for rows.
        o_col: Base-2 order for columns.

    Returns:
        sorted_matrix, perm

        where
        - sorted_matrix[s] is the flattened pattern at sorted index s
        - perm[s] is the corresponding natural Hadamard index
    """
    perm = generate_order_sorting_matrix(o_col=o_col, o_row=o_row)

    H_row = hadamard_matrix(o_row)
    H_col = hadamard_matrix(o_col)

    n_rows = 2**o_row
    n_cols = 2**o_col
    n_patterns = n_rows * n_cols

    S = np.empty((n_patterns, n_patterns), dtype=np.int8)

    for sorted_index in range(n_patterns):
        natural_index = int(perm[sorted_index])

        row_index = natural_index // n_cols
        col_index = natural_index % n_cols

        pattern_2d = np.outer(H_row[row_index], H_col[col_index]).astype(
            np.int8, copy=False
        )
        S[sorted_index] = pattern_2d.reshape(-1)

    return S, perm


def reorder_sorted_to_natural(
    sorted_matrix: np.ndarray, perm: np.ndarray
) -> np.ndarray:
    """Convert a sorted-order matrix into natural order.

    Args:
        sorted_matrix: Matrix with rows indexed by sorted order.
        perm:  Mapping sorted_index -> natural_index.

    Returns:
        Matrix with rows in natural order.
    """
    inverse_perm = np.argsort(perm)
    return sorted_matrix[inverse_perm]


def reorder_natural_to_sorted(
    natural_matrix: np.ndarray, perm: np.ndarray
) -> np.ndarray:
    """Convert a natural-order matrix into sorted order.

    Parameters
    ----------
    natural_matrix : np.ndarray
        Matrix with rows indexed by natural order.
    perm : np.ndarray
        Mapping sorted_index -> natural_index.

    Returns:
    -------
    np.ndarray
        Matrix with rows in sorted order.
    """
    return natural_matrix[perm]


def hadamard_row_plus_minus(row_index: int, order: int) -> np.ndarray:
    """Generate one row of a Sylvester Hadamard matrix with entries in {-1, +1}.

    We could generate using
    ``H[row_index, j] = (-1) ** bitwise_count(row_index & j)``.

    However, here we use
    ``H[row_index, j] = 1 - 2 * (bitwise_count(row_index & j) % 2)``.

    Args:
        row_index: Row index in the 1D Hadamard matrix.
        order: Base-2 order, so the row length is ``2**order``.

    Returns:
        Array of shape ``(2**order,)`` with values in ``{-1, +1}``.
    """
    n = 1 << order
    j = np.arange(n, dtype=np.uint32)
    parity = np.bitwise_count(np.uint32(row_index) & j) % np.uint8(2)
    return (1 - 2 * parity.astype(np.int8)).astype(np.int8)


def hadamard_pattern_from_sorted_index(
    sorted_index: int,
    o_col: int,
    o_row: int,
    *,
    flatten_order: Literal["row-major", "col-major"] = "row-major",
) -> dict[str, int | np.ndarray]:
    """Generate a 2D Hadamard pattern from its sorted-order index.


    Args:
        sorted_index: Index in the sorted order.
        o_col: Base-2 order for the number of columns.
        o_row: Base-2 order for the number of rows.
        flatten_order: Convention used to decode the natural Hadamard index into 2D row/column basis indices.

    Returns:
        Dictionary containing:
        - ``sorted_index``
        - ``hadamard_index``
        - ``row_index``
        - ``col_index``
        - ``pattern_pm``      : values in {-1, +1}
        - ``pattern_01``      : values in {0, 1}
        - ``positive_mask``   : 1 where pattern_pm == +1
        - ``negative_mask``   : 1 where pattern_pm == -1
    """
    n_rows = 1 << o_row
    n_cols = 1 << o_col
    n_patterns = n_rows * n_cols

    if not (0 <= sorted_index < n_patterns):
        raise ValueError(
            f"sorted_index must lie in [0, {n_patterns}), got {sorted_index}."
        )

    perm = generate_order_sorting_matrix(o_col=o_col, o_row=o_row)
    hadamard_index = int(perm[sorted_index])

    if flatten_order == "row-major":
        row_index = hadamard_index // n_cols
        col_index = hadamard_index % n_cols
    elif flatten_order == "col-major":
        row_index = hadamard_index % n_rows
        col_index = hadamard_index // n_rows
    else:
        raise ValueError(f"Unsupported flatten_order: {flatten_order}")

    row_vec = hadamard_row_plus_minus(row_index, o_row)
    col_vec = hadamard_row_plus_minus(col_index, o_col)

    pattern_pm = np.outer(row_vec, col_vec).astype(np.int8, copy=False)
    pattern_01 = ((pattern_pm + 1) // 2).astype(np.uint8)
    positive_mask = (pattern_pm > 0).astype(np.uint8)
    negative_mask = (pattern_pm < 0).astype(np.uint8)

    return {
        "sorted_index": sorted_index,
        "hadamard_index": hadamard_index,
        "row_index": row_index,
        "col_index": col_index,
        "pattern_pm": pattern_pm,
        "pattern_01": pattern_01,
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
    }
