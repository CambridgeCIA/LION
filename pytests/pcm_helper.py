import numpy as np


def compare_rows_up_to_sign(
    A: np.ndarray, B: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compare two matrices rowwise, allowing a global sign flip per row.

    Parameters
    ----------
    A : np.ndarray
        First matrix.
    B : np.ndarray
        Second matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ok, signs

        - ok[i] is True if row i matches exactly or up to sign
        - signs[i] is
            +1 if exact match,
            -1 if negative match,
             0 otherwise
    """
    same = np.all(A == B, axis=1)
    neg = np.all(A == -B, axis=1)

    ok = same | neg
    signs = np.zeros(A.shape[0], dtype=np.int8)
    signs[same] = 1
    signs[neg] = -1
    return ok, signs


def check_orthogonality(M: np.ndarray) -> bool:
    """
    Check whether M M^T = n I exactly.

    Parameters
    ----------
    M : np.ndarray
        Matrix with entries in {-1, +1}.

    Returns
    -------
    bool
        True if orthogonal up to the Hadamard scaling.
    """
    n = M.shape[1]
    G = M @ M.T
    return np.array_equal(G, n * np.eye(M.shape[0], dtype=G.dtype))
