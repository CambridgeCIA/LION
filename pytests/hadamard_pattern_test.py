from __future__ import annotations

import argparse

import numpy as np
from LION.pcm_tools.hadamard_matrix import hadamard_pattern_from_sorted_index


def print_pattern_summary(result: dict[str, int | np.ndarray]) -> None:
    """
    Print a short summary of a generated pattern.

    Parameters
    ----------
    result : dict[str, int | np.ndarray]
        Output from ``hadamard_pattern_from_sorted_index``.
    """
    pattern_pm = result["pattern_pm"]
    assert isinstance(pattern_pm, np.ndarray)

    print(f"sorted_index   = {result['sorted_index']}")
    print(f"hadamard_index = {result['hadamard_index']}")
    print(f"row_index      = {result['row_index']}")
    print(f"col_index      = {result['col_index']}")
    print(f"shape          = {pattern_pm.shape}")
    print(f"unique values  = {np.unique(pattern_pm)}")
    print()
    print("pattern_pm:")
    print(pattern_pm)


def maybe_plot_pattern(pattern: np.ndarray, title: str) -> None:
    """
    Plot a 2D pattern if matplotlib is available.

    Parameters
    ----------
    pattern : np.ndarray
        2D array to display.
    title : str
        Plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return

    plt.figure(figsize=(6, 6))
    plt.imshow(pattern, cmap="gray", interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Run a simple command-line test.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--o-row", type=int, default=3, help="Base-2 order of rows.")
    parser.add_argument("--o-col", type=int, default=3, help="Base-2 order of columns.")
    parser.add_argument(
        "--sorted-index",
        type=int,
        default=0,
        help="Index in the sorted order.",
    )
    parser.add_argument(
        "--flatten-order",
        type=str,
        default="row-major",
        choices=["row-major", "col-major"],
        help="Convention used to decode the natural Hadamard index.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the pattern using matplotlib.",
    )
    args = parser.parse_args()

    result = hadamard_pattern_from_sorted_index(
        sorted_index=args.sorted_index,
        o_col=args.o_col,
        o_row=args.o_row,
        flatten_order=args.flatten_order,
    )

    print_pattern_summary(result)

    pattern_pm = result["pattern_pm"]
    pattern_01 = result["pattern_01"]
    assert isinstance(pattern_pm, np.ndarray)
    assert isinstance(pattern_01, np.ndarray)

    print()
    print("pattern_01:")
    print(pattern_01)

    if args.plot:
        maybe_plot_pattern(
            pattern_pm,
            title=(
                f"sorted={result['sorted_index']}, "
                f"hadamard={result['hadamard_index']}, "
                f"row={result['row_index']}, "
                f"col={result['col_index']}"
            ),
        )


if __name__ == "__main__":
    main()
