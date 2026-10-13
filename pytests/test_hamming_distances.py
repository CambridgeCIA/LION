from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_pcm_data(filepath: str | Path) -> pd.DataFrame:
    """
    Read a PCM measurement file.

    Args:
    filepath: Path to the PCM text file.

    Returns:
        DataFrame with columns ``PatternIndex`` and ``MeasuredCurrentA``.
    """
    return pd.read_csv(
        filepath,
        sep=r"\t",
        skiprows=3,
        header=None,
        names=["PatternIndex", "MeasuredCurrentA"],
        usecols=[0, 1],
        engine="python",
    )


def normal_to_dyadic_permutation(J: int) -> np.ndarray:
    """
    Compute the permutation mapping dyadic/sorted order to normal order.

    Parameters
    ----------
    J : int
        Image order. The image size is ``2**J x 2**J``.

    Returns
    -------
    np.ndarray
        Permutation array ``perm`` such that

        ``perm[sorted_index] = natural_index``.
    """
    nbits = 2 * J
    n = 1 << nbits

    k = np.arange(n, dtype=np.uint32)
    shifts = np.arange(nbits, dtype=np.uint32)
    v = ((k[:, None] >> shifts) & np.uint32(1)).astype(np.uint8)

    p = np.r_[np.arange(0, nbits, 2), np.arange(1, nbits, 2)]
    v = v[:, p]

    weights = np.uint32(1) << np.arange(nbits - 1, -1, -1, dtype=np.uint32)
    permutation = (v @ weights).astype(np.int64)
    return permutation


def hadamard_row_pm(index: int, order: int) -> np.ndarray:
    """
    Generate one row of a Sylvester Hadamard matrix in ``{-1, +1}``.

    Parameters
    ----------
    index : int
        Row index.
    order : int
        Base-2 order. The row length is ``2**order``.

    Returns
    -------
    np.ndarray
        Row vector with entries in ``{-1, +1}``.
    """
    n = 1 << order
    j = np.arange(n, dtype=np.uint32)
    parity = np.bitwise_count(np.uint32(index) & j) % np.uint8(2)
    return (1 - 2 * parity.astype(np.int8)).astype(np.int8)


def pm_pattern_from_sorted_index(
    sorted_index: int, J: int, perm: np.ndarray | None = None
) -> np.ndarray:
    """
    Generate the 2D plus-minus Hadamard pattern for one sorted-order index.

    Parameters
    ----------
    sorted_index : int
        Sorted-order index.
    J : int
        Image order. The image size is ``2**J x 2**J``.
    perm : np.ndarray | None, optional
        Permutation mapping sorted index to natural index. If omitted, it is
        computed internally.

    Returns
    -------
    np.ndarray
        2D pattern with entries in ``{-1, +1}``.
    """
    n_side = 1 << J
    n_patterns = n_side * n_side

    if not (0 <= sorted_index < n_patterns):
        raise ValueError(
            f"sorted_index must lie in [0, {n_patterns}), got {sorted_index}."
        )

    if perm is None:
        perm = normal_to_dyadic_permutation(J)

    natural_index = int(perm[sorted_index])

    row_index = natural_index // n_side
    col_index = natural_index % n_side

    row_vec = hadamard_row_pm(row_index, J)
    col_vec = hadamard_row_pm(col_index, J)

    return np.outer(row_vec, col_vec).astype(np.int8, copy=False)


def resolve_display_signs(pattern_index_values: np.ndarray) -> np.ndarray:
    """
    Resolve the display sign sequence from the PatternIndex column.

    The file uses two zero entries at the beginning, which are ambiguous under
    ``np.sign(0)``. This function follows the same convention as the existing
    decoding code:
    - first zero is treated as ``+0``
    - second zero is treated as ``-0``

    Parameters
    ----------
    pattern_index_values : np.ndarray
        PatternIndex values as read from file.

    Returns
    -------
    np.ndarray
        Array of signs in ``{-1, +1}``.
    """
    rounded = np.round(pattern_index_values).astype(np.int64)
    signs = np.sign(rounded).astype(np.int8)

    zero_positions = np.flatnonzero(rounded == 0)
    if zero_positions.size >= 1:
        signs[zero_positions[0]] = 1
    if zero_positions.size >= 2:
        signs[zero_positions[1]] = -1
    if zero_positions.size > 2:
        raise ValueError(
            "More than two zero PatternIndex entries found; the sign convention is ambiguous."
        )

    if np.any(signs == 0):
        raise ValueError("Some PatternIndex signs remain unresolved.")

    return signs


def displayed_binary_pattern_from_signed_sorted_index(
    sorted_index: int,
    display_sign: int,
    J: int,
    perm: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate the displayed binary DMD pattern for one signed sorted-order index.

    Convention:
    - if ``display_sign == +1``, ones correspond to ``+1`` entries of the
      plus-minus Hadamard pattern
    - if ``display_sign == -1``, ones correspond to ``-1`` entries of the
      plus-minus Hadamard pattern

    Parameters
    ----------
    sorted_index : int
        Sorted-order pattern index.
    display_sign : int
        Either ``+1`` or ``-1``.
    J : int
        Image order. The image size is ``2**J x 2**J``.
    perm : np.ndarray | None, optional
        Permutation mapping sorted index to natural index.

    Returns
    -------
    np.ndarray
        2D binary pattern with entries in ``{0, 1}``.
    """
    if display_sign not in (-1, 1):
        raise ValueError(f"display_sign must be -1 or +1, got {display_sign}.")

    pattern_pm = pm_pattern_from_sorted_index(sorted_index, J, perm=perm)
    pattern_01 = ((display_sign * pattern_pm + 1) // 2).astype(np.uint8)
    return pattern_01


def compute_consecutive_hamming_distances(
    filepath: str | Path,
    J: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute Hamming distances between consecutive displayed patterns in file order.

    Parameters
    ----------
    filepath : str | Path
        Path to the PCM measurement file.
    J : int
        Image order. The image size is ``2**J x 2**J``.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        A tuple ``(result_df, signs)`` where

        - ``result_df`` contains one row per consecutive pair
        - ``signs`` is the resolved sign sequence for the file
    """
    df = read_pcm_data(filepath)

    raw_pattern_index = df["PatternIndex"].to_numpy(dtype=np.float64)
    sorted_indices = np.round(np.abs(raw_pattern_index)).astype(np.int64)
    display_signs = resolve_display_signs(raw_pattern_index)

    n_side = 1 << J
    n_pixels = n_side * n_side
    perm = normal_to_dyadic_permutation(J)

    if np.any(sorted_indices >= n_pixels):
        bad = sorted_indices[sorted_indices >= n_pixels]
        raise ValueError(
            f"Found sorted indices outside [0, {n_pixels - 1}]: {bad.tolist()}"
        )

    rows: list[dict[str, int | float]] = []
    prev_pattern: np.ndarray | None = None
    prev_sorted_index: int | None = None
    prev_display_sign: int | None = None

    for i, (sorted_index, display_sign) in enumerate(
        zip(sorted_indices, display_signs)
    ):
        pattern = displayed_binary_pattern_from_signed_sorted_index(
            sorted_index=int(sorted_index),
            display_sign=int(display_sign),
            J=J,
            perm=perm,
        )

        if prev_pattern is not None:
            hamming_distance = int(np.count_nonzero(prev_pattern != pattern))
            rows.append(
                {
                    "prev_file_row": i - 1,
                    "curr_file_row": i,
                    "prev_sorted_index": int(prev_sorted_index),
                    "curr_sorted_index": int(sorted_index),
                    "prev_display_sign": int(prev_display_sign),
                    "curr_display_sign": int(display_sign),
                    "hamming_distance": hamming_distance,
                    "normalised_hamming_distance": hamming_distance / n_pixels,
                }
            )

        prev_pattern = pattern
        prev_sorted_index = int(sorted_index)
        prev_display_sign = int(display_sign)

    result_df = pd.DataFrame(rows)
    return result_df, display_signs


def build_displayed_pattern_stack(
    filepath: str | Path,
    J: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the full stack of displayed binary patterns in file order.

    Parameters
    ----------
    filepath : str | Path
        Path to the PCM measurement file.
    J : int
        Image order. The image size is ``2**J x 2**J``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(pattern_stack, sorted_indices, display_signs)`` where

        - ``pattern_stack`` has shape ``(num_measurements, 2**J, 2**J)``
        - ``sorted_indices`` are the absolute sorted-order indices
        - ``display_signs`` are the resolved display signs
    """
    df = read_pcm_data(filepath)

    raw_pattern_index = df["PatternIndex"].to_numpy(dtype=np.float64)
    sorted_indices = np.round(np.abs(raw_pattern_index)).astype(np.int64)
    display_signs = resolve_display_signs(raw_pattern_index)

    perm = normal_to_dyadic_permutation(J)
    patterns = [
        displayed_binary_pattern_from_signed_sorted_index(
            sorted_index=int(idx),
            display_sign=int(sign),
            J=J,
            perm=perm,
        )
        for idx, sign in zip(sorted_indices, display_signs)
    ]

    return np.stack(patterns, axis=0), sorted_indices, display_signs


def plot_consecutive_hamming_distances(
    distances_df: pd.DataFrame,
    *,
    normalised: bool = False,
    figsize: tuple[float, float] = (10.0, 4.0),
) -> None:
    """
    Plot consecutive Hamming distances in file order.

    Parameters
    ----------
    distances_df : pd.DataFrame
        Output from ``compute_consecutive_hamming_distances``.
    normalised : bool, optional
        If True, plot the normalised Hamming distance instead of the raw count.
    figsize : tuple[float, float], optional
        Figure size.
    """
    if distances_df.empty:
        raise ValueError("distances_df is empty.")

    y_col = "normalised_hamming_distance" if normalised else "hamming_distance"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        distances_df["curr_file_row"].to_numpy(),
        distances_df[y_col].to_numpy(),
        marker="o",
        linewidth=1.0,
        markersize=3.0,
    )
    ax.set_xlabel("File row")
    ax.set_ylabel("Normalised Hamming distance" if normalised else "Hamming distance")
    ax.set_title("Consecutive pattern Hamming distances")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_first_displayed_patterns(
    pattern_stack: np.ndarray,
    sorted_indices: np.ndarray,
    display_signs: np.ndarray,
    *,
    num_patterns: int = 12,
    ncols: int = 4,
    figsize_per_panel: tuple[float, float] = (2.5, 2.5),
) -> None:
    """
    Plot the first few displayed binary patterns in file order.

    Parameters
    ----------
    pattern_stack : np.ndarray
        Array of shape ``(num_measurements, H, W)`` with binary patterns.
    sorted_indices : np.ndarray
        Absolute sorted-order indices for each displayed pattern.
    display_signs : np.ndarray
        Display signs in ``{-1, +1}``.
    num_patterns : int, optional
        Number of patterns to display.
    ncols : int, optional
        Number of subplot columns.
    figsize_per_panel : tuple[float, float], optional
        Width and height per subplot panel.
    """
    if pattern_stack.ndim != 3:
        raise ValueError(
            f"pattern_stack must have shape (num_patterns, H, W), got {pattern_stack.shape}."
        )

    n_total = pattern_stack.shape[0]
    n_show = min(num_patterns, n_total)
    nrows = math.ceil(n_show / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n_show:
            ax.imshow(
                pattern_stack[i], cmap="gray", interpolation="nearest", vmin=0, vmax=1
            )
            sign_str = "+" if int(display_signs[i]) > 0 else "-"
            ax.set_title(f"{sign_str}{int(sorted_indices[i])}")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    fig.suptitle("Displayed binary patterns in file order")
    fig.tight_layout()
    plt.show()


def plot_pattern_pair_and_difference(
    pattern_a: np.ndarray,
    pattern_b: np.ndarray,
    *,
    title_a: str = "Pattern A",
    title_b: str = "Pattern B",
) -> None:
    """
    Plot two binary patterns and their pixelwise difference.

    Parameters
    ----------
    pattern_a : np.ndarray
        First binary pattern.
    pattern_b : np.ndarray
        Second binary pattern.
    title_a : str, optional
        Title for the first pattern.
    title_b : str, optional
        Title for the second pattern.
    """
    if pattern_a.shape != pattern_b.shape:
        raise ValueError(
            f"pattern_a and pattern_b must have the same shape, got {pattern_a.shape} and {pattern_b.shape}."
        )

    diff = (pattern_a != pattern_b).astype(np.uint8)
    hamming_distance = int(np.count_nonzero(diff))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))

    axes[0].imshow(pattern_a, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axes[0].set_title(title_a)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(pattern_b, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axes[1].set_title(title_b)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(diff, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axes[2].set_title(f"Difference, Hamming = {hamming_distance}")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.tight_layout()
    plt.show()


def plot_average_distance_by_previous_index(
    distances_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (10.0, 4.0),
) -> None:
    """
    Plot the average consecutive Hamming distance grouped by previous sorted index.

    Parameters
    ----------
    distances_df : pd.DataFrame
        Output from ``compute_consecutive_hamming_distances``.
    figsize : tuple[float, float], optional
        Figure size.
    """
    grouped = (
        distances_df.groupby("prev_sorted_index", as_index=False)["hamming_distance"]
        .mean()
        .sort_values("prev_sorted_index")
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        grouped["prev_sorted_index"].to_numpy(),
        grouped["hamming_distance"].to_numpy(),
        width=0.8,
    )
    ax.set_xlabel("Previous sorted index")
    ax.set_ylabel("Average Hamming distance")
    ax.set_title("Average switching distance by previous pattern index")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    filepath = "pytests/Si_2_256_short.TXT"
    J = 4  # for a 4 x 4 pattern set
    # J = 5  # for a 32 x 32 pattern set

    distances_df, display_signs = compute_consecutive_hamming_distances(filepath, J)

    print(distances_df.head(20))

    # Sanity check:
    # consecutive rows with the same absolute index but opposite signs should be complements,
    # so their Hamming distance should equal the total number of pixels.
    n_pixels = (1 << J) ** 2
    same_index_opposite_sign = distances_df[
        (distances_df["prev_sorted_index"] == distances_df["curr_sorted_index"])
        & (distances_df["prev_display_sign"] != distances_df["curr_display_sign"])
    ]
    print()
    print("Complement-pair sanity check:")
    print(
        same_index_opposite_sign[
            [
                "prev_file_row",
                "curr_file_row",
                "prev_sorted_index",
                "curr_sorted_index",
                "hamming_distance",
            ]
        ]
    )
    print(
        "All complement-pair distances equal n_pixels:",
        bool((same_index_opposite_sign["hamming_distance"] == n_pixels).all()),
    )

    pattern_stack, sorted_indices, display_signs = build_displayed_pattern_stack(
        filepath, J
    )

    plot_consecutive_hamming_distances(distances_df, normalised=False)
    plot_consecutive_hamming_distances(distances_df, normalised=True)

    plot_first_displayed_patterns(
        pattern_stack, sorted_indices, display_signs, num_patterns=12, ncols=4
    )

    # Inspect one consecutive pair, for example rows 2 and 3 in file order
    i = 2
    plot_pattern_pair_and_difference(
        pattern_stack[i],
        pattern_stack[i + 1],
        title_a=f"{'+' if display_signs[i] > 0 else '-'}{int(sorted_indices[i])}",
        title_b=f"{'+' if display_signs[i + 1] > 0 else '-'}{int(sorted_indices[i + 1])}",
    )

    plot_average_distance_by_previous_index(distances_df)

    plot_first_displayed_patterns(
        pattern_stack, sorted_indices, display_signs, num_patterns=8, ncols=4
    )
    plot_consecutive_hamming_distances(distances_df, normalised=False)
