"""Ordering utilities for PCM Walsh-Hadamard measurements."""

from __future__ import annotations

import numpy as np


def generate_order_sorting_matrix(o_col: int, o_row: int) -> np.ndarray:
    """Generate the permutation from coarse-to-fine ordered indices to Hadamard indices."""
    bit_swap_order = np.arange(1, o_col + o_row + 1, dtype=np.int64)[::-1]

    i1 = 2 * min(o_col, o_row)
    b1 = bit_swap_order[:i1]
    b2 = bit_swap_order[i1:]

    b11 = b1[::2]
    b12 = b1[1::2]

    b11 = np.concatenate([b11, b2])
    b12 = np.concatenate([b12, b2])

    bit_swap_order = np.concatenate([b11[:o_row], b12[:o_col]])

    width = o_col + o_row
    n = (2**o_col) * (2**o_row)

    ix = np.arange(n, dtype=np.uint32)
    iy = np.zeros(n, dtype=np.uint32)

    for out_pos, src_pos_1_based in enumerate(bit_swap_order):
        src_bit_from_lsb = width - int(src_pos_1_based)
        out_bit_from_lsb = width - 1 - out_pos
        iy |= ((ix >> np.uint32(src_bit_from_lsb)) & np.uint32(1)) << np.uint32(
            out_bit_from_lsb
        )

    return iy.astype(np.int64)


def normal_to_dyadic_permutation(j_order: int) -> np.ndarray:
    """Return the square-image normal-to-dyadic WHT permutation."""
    return generate_order_sorting_matrix(o_col=j_order, o_row=j_order)
