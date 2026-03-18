from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from turtle import pos
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from sympy import content


@dataclass(frozen=True)
class RawPCM:
    """Raw photocurrent mapping measurements.

    Attributes
    ----------
    order_signed :
        Signed pattern indices as integers (one per row in the TXT file).
    current_a :
        Measured currents in amperes (same length as `order_signed`).
    """

    order_signed: np.ndarray  # shape (M,), int
    current_a: np.ndarray  # shape (M,), float


def read_pcm_txt(txt_path: Path) -> RawPCM:
    """Read a PCM TXT file with (signed index, measured current)."""
    order_vals: list[int] = []
    current_vals: list[float] = []

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.split("%", 1)[0].strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                idx = int(round(float(parts[0])))
                cur = float(parts[1])
            except ValueError:
                continue

            order_vals.append(idx)
            current_vals.append(cur)

    if not order_vals:
        raise ValueError(f"No numeric data rows parsed from {txt_path}")

    return RawPCM(
        order_signed=np.asarray(order_vals, dtype=np.int64),
        current_a=np.asarray(current_vals, dtype=np.float64),
    )


def pair_pos_neg_to_differences(
    raw: RawPCM,
    *,
    positive_sign: Literal[+1, -1] = +1,
) -> np.ndarray:
    """Convert raw signed-index measurements into per-order differences.

    Assumption used (matches the example given):
    - For order k>0, one measurement is labeled +k and the other -k.
    - The difference is (positive_pattern - negative_pattern).
    - `positive_sign=+1` means +k is treated as "positive pattern" and -k as "negative pattern".
      If the reconstructed map looks inverted / wrong, try `positive_sign=-1`.

    For k=0, both rows often have index 0; the first occurrence is treated as positive,
    the second as negative.

    Parameters
    ----------
    raw :
        Parsed measurements.
    positive_sign :
        Which sign denotes the positive pattern.

    Returns
    -------
    diffs :
        Vector d[k] = I_pos(k) - I_neg(k), for k=0..Kmax, shape (Kmax+1,).
    """
    idx = raw.order_signed
    cur = raw.current_a

    abs_k = np.abs(idx)
    k_max = int(abs_k.max())
    diffs = np.empty(k_max + 1, dtype=np.float64)

    # k = 0 special case: expect two entries with index 0
    zero_rows = np.flatnonzero(idx == 0)
    if zero_rows.size < 2:
        raise ValueError(
            "Could not find two zero-order rows (index 0) to form a difference."
        )
    diffs[0] = cur[zero_rows[0]] - cur[zero_rows[1]]

    # k > 0: use sign convention
    for k in range(1, k_max + 1):
        pos_label = positive_sign * k
        neg_label = -positive_sign * k

        pos_rows = np.flatnonzero(idx == pos_label)
        neg_rows = np.flatnonzero(idx == neg_label)

        if pos_rows.size != 1 or neg_rows.size != 1:
            raise ValueError(
                f"Order {k}: expected one row with index {pos_label} and one with {neg_label}, "
                f"found {pos_rows.size} and {neg_rows.size}."
            )

        diffs[k] = cur[pos_rows[0]] - cur[neg_rows[0]]

    return diffs


def fwht_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """Fast Walsh-Hadamard transform along one axis (power-of-2 length).

    This implements the unnormalised transform (using +/- butterflies).
    For length n, the inverse equals (1/n) * FWHT.

    Parameters
    ----------
    a :
        Input array.
    axis :
        Axis along which to transform.

    Returns
    -------
    out :
        Transformed array (copy).
    """
    out = np.swapaxes(a, axis, -1).copy()
    n = out.shape[-1]

    if n & (n - 1) != 0:
        raise ValueError(f"FWHT requires power-of-2 length, got {n}")

    h = 1
    while h < n:
        out_reshaped = out.reshape(*out.shape[:-1], n // (2 * h), 2 * h)
        first = out_reshaped[..., :, :h]
        second = out_reshaped[..., :, h : 2 * h]

        out_reshaped[..., :, :h] = first + second
        out_reshaped[..., :, h : 2 * h] = first - second

        out = out_reshaped.reshape(*out.shape[:-1], n)
        h *= 2

    return np.swapaxes(out, axis, -1)


def fwht2d(a: np.ndarray) -> np.ndarray:
    """2D FWHT (rows then columns)."""
    out = fwht_axis(a, axis=1)  # along columns
    out = fwht_axis(out, axis=0)  # along rows
    return out


def reconstruct_map_from_differences(
    diffs: np.ndarray,
    *,
    n: int = 256,
    ordering: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct an n x n map from Hadamard coefficient differences.

    This assumes the measurement differences correspond to a flattened 2D
    Walsh-Hadamard coefficient matrix C of shape (n, n), possibly permuted.

    If ordering is provided, it is interpreted as:
        ordering[k] = (i, j) pair encoded as a single integer i*n + j
    or more simply:
        ordering is a permutation of 0..n*n-1 that maps measurement index -> coefficient slot.

    Parameters
    ----------
    diffs :
        Differences per order, length should be n*n.
    n :
        Map side length.
    ordering :
        Optional permutation mapping for the Hadamard ordering.

    Returns
    -------
    x_hat :
        Reconstructed current map, shape (n, n).
    """
    expected = n * n
    if diffs.size != expected:
        raise ValueError(
            f"Expected {expected} Hadamard orders for {n}x{n}, got {diffs.size}"
        )

    coeffs = diffs.copy()

    if ordering is not None:
        ordering = np.asarray(ordering, dtype=np.int64)
        if ordering.shape != (expected,):
            raise ValueError(
                f"ordering must have shape ({expected},), got {ordering.shape}"
            )
        if np.unique(ordering).size != expected:
            raise ValueError("ordering must be a permutation of 0..n*n-1")

        # Place measurement k into coefficient slot ordering[k]
        coeffs_reordered = np.empty_like(coeffs)
        coeffs_reordered[ordering] = coeffs
        coeffs = coeffs_reordered

    C = coeffs.reshape(n, n)

    # If C = H X H (unnormalised), then X = (1/n^2) * H C H
    x_hat = fwht2d(C) / (n * n)
    return x_hat


def read_map_file(map_path: Path, *, n: int = 256) -> np.ndarray:
    """Read a .TXT.map file into an (n, n) array.

    Handles common cases:
    - n rows x n columns numeric grid
    - single column of length n*n
    - 3 columns: x, y, value (regridded)

    Parameters
    ----------
    map_path :
        Path to the .TXT.map file.
    n :
        Expected map size.

    Returns
    -------
    m :
        Map array of shape (n, n).
    """
    arr = np.loadtxt(map_path)

    if arr.ndim == 2 and arr.shape == (n, n):
        return arr.astype(np.float64)

    if arr.ndim == 1 and arr.size == n * n:
        return arr.reshape(n, n).astype(np.float64)

    if arr.ndim == 2 and arr.shape[1] == 3:
        # columns: x, y, value
        x = arr[:, 0].astype(int)
        y = arr[:, 1].astype(int)
        v = arr[:, 2].astype(np.float64)

        m = np.empty((n, n), dtype=np.float64)
        m[:] = np.nan

        if (x.min() < 0) or (y.min() < 0) or (x.max() >= n) or (y.max() >= n):
            raise ValueError("3-column map appears to have coordinates outside 0..n-1")

        m[y, x] = v  # common convention: x is column, y is row
        if np.isnan(m).any():
            raise ValueError("3-column map did not fill a complete n x n grid")
        return m

    raise ValueError(f"Unrecognised map format for {map_path}: shape {arr.shape}")


def compare_and_plot(
    recon: np.ndarray,
    ref: np.ndarray,
    *,
    title: str,
) -> None:
    """Quick comparison plots and summary statistics."""
    recon = recon.astype(np.float64)
    ref = ref.astype(np.float64)

    diff = recon - ref
    rmse = float(np.sqrt(np.mean(diff**2)))
    corr = float(np.corrcoef(recon.ravel(), ref.ravel())[0, 1])

    print(f"{title}")
    print(
        f"  recon: min={recon.min():.6g}, max={recon.max():.6g}, mean={recon.mean():.6g}"
    )
    print(f"  ref  : min={ref.min():.6g}, max={ref.max():.6g}, mean={ref.mean():.6g}")
    print(f"  RMSE={rmse:.6g}, corr={corr:.6g}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    axes = axes[0]

    im0 = axes[0].imshow(ref, cmap="gray")
    axes[0].set_title("Reference (.map)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(recon, cmap="gray")
    axes[1].set_title("Reconstruction (from .TXT)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap="gray")
    axes[2].set_title("Recon - Ref")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def run_one_dataset(
    txt_path: Path,
    map_path: Path,
    *,
    n: int = 256,
    positive_sign: Literal[+1, -1] = +1,
    ordering: np.ndarray | None = None,
) -> None:
    """End-to-end example for one (TXT, TXT.map) pair."""
    raw = read_pcm_txt(txt_path)
    diffs = pair_pos_neg_to_differences(raw, positive_sign=positive_sign)
    recon = reconstruct_map_from_differences(diffs, n=n, ordering=ordering)
    ref = read_map_file(map_path, n=n)
    compare_and_plot(recon, ref, title=txt_path.stem)


def load_and_plot_map_image(
    example_pcm_data_dir: Path, map_image_file_name: str
) -> None:
    print(f"Loading map image from {map_image_file_name}...")
    name = map_image_file_name.split(".")[0]
    map_image_file = example_pcm_data_dir / map_image_file_name

    map_image_np = np.loadtxt(map_image_file)

    # Strip the last row until the last row is not all zeros
    while map_image_np.shape[0] > 0 and np.all(map_image_np[-1, :] == 0):
        map_image_np = map_image_np[:-1, :]
    # Strip the last column until the last column is not all zeros
    while map_image_np.shape[1] > 0 and np.all(map_image_np[:, -1] == 0):
        map_image_np = map_image_np[:, :-1]
    # Strip the first row until the first row is not all zeros
    while map_image_np.shape[0] > 0 and np.all(map_image_np[0, :] == 0):
        map_image_np = map_image_np[1:, :]
    # Strip the first column until the first column is not all zeros
    while map_image_np.shape[1] > 0 and np.all(map_image_np[:, 0] == 0):
        map_image_np = map_image_np[:, 1:]

    print(f"{name} shape: {map_image_np.shape}")
    print(f"{name} data type: {map_image_np.dtype}")
    print(f"{name} min: {map_image_np.min()}")
    print(f"{name} max: {map_image_np.max()}")
    np.save(example_pcm_data_dir / f"{name}.npy", map_image_np)

    plt.imshow(map_image_np, cmap="gray")
    plt.colorbar()
    plt.title(
        f"Loaded map image from {map_image_file_name}\n"
        f"shape: {map_image_np.shape}\n"
        f"min: {map_image_np.min():.6g}, max: {map_image_np.max():.6g}"
    )
    plt.tight_layout()
    plt.savefig(example_pcm_data_dir / f"{name}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    data_dir = Path("data")
    measurement_file_name = (
        "GaAs ref cell closer 20ms 16repeat 100mAbiasLED 800mV bias "
        "256x256 +390+330.txt"
    )
    output_measurement_file_name_only = "GaAs"

    example_pcm_data_dir = data_dir / "Example PCM data"
    # load_and_plot_map_image(example_pcm_data_dir, "output.txt")

    # measurements: list[tuple[int, float]] = []
    # indices: list[int] = []
    # measured_currents_ampere: list[float] = []

    # Map image size is 2^order_row x 2^order_col.
    # We have 2^order_row * 2^order_col * 2 measurements (positive and negative patterns).
    order_row = 8
    order_col = 8
    num_measurements_expected = (2**order_row) * (2**order_col) * 2
    print(
        f"Expecting {num_measurements_expected} measurements for {2**order_row}x{2**order_col} map."
    )
    with (example_pcm_data_dir / measurement_file_name).open(
        "r", encoding="utf-8", errors="ignore"
    ) as f:
        lines = f.readlines()

        # The content of the measurement file looks like this:
        # (1)    Output from Current mapping V2
        # (2)    (blank line)
        # (3)    Pattern Index	Measured current (A)
        # (4)    0.00000000E+0	2.13886190E-5
        # (5)    0.00000000E+0	1.30551160E-7
        # (6)    1.00000000E+0	8.66430485E-6
        # (7)    -1.00000000E+0	1.35798561E-5
        # (8)    2.00000000E+0	1.02335228E-5
        # (9)    -2.00000000E+0	1.20306721E-5

        # So we skip the first 3 header lines and parse from line 4 onwards
        lines = lines[3:]  # skip first 3 header lines
        num_lines = len(lines)
        # The file should contain a multiple of num_measurements_expected lines
        assert num_lines % num_measurements_expected == 0, (
            "Unexpected number of lines. "
            f"Should be a multiple of {num_measurements_expected}, but got {num_lines}"
        )
        num_blocks = num_lines // num_measurements_expected
        print(
            f"File contains {num_lines} lines, which is {num_blocks} blocks of {num_measurements_expected} measurements each."
        )
        for block_index in range(num_blocks):
            line_block_start = block_index * num_measurements_expected
            print(
                f"Processing block {block_index} with lines {line_block_start} to {line_block_start + num_measurements_expected - 1}..."
            )
            block = lines[
                line_block_start : line_block_start + num_measurements_expected
            ]

            # Each line in the block contains 2 numbers: pattern index and measured current
            # Let's save them as a float np.array with shape (num_measurements_expected, 2)
            out = np.fromstring("\n".join(block), sep=" ").reshape((-1, 2))
            assert out.shape == (num_measurements_expected, 2), (
                f"Unexpected shape of parsed data: {out.shape}, "
                f"expected ({num_measurements_expected}, 2)"
            )
            # The first row should contain index 0
            assert (
                out[0, 0] == 0.0
            ), f"First pattern index in block is not 0, got {out[0, 0]}"
            # Every consecutive pair of rows should have opposite indices
            # and the same absolute value, with the pair's absolute index increasing by 1 each time.
            # The order of the two rows in the pair may vary.
            for i in range(num_measurements_expected // 2):
                index_1 = out[2 * i, 0]
                index_2 = out[2 * i + 1, 0]
                print(f"Pair {i}: indices {index_1}, {index_2}")
                pos_index = max(index_1, index_2)
                neg_index = min(index_1, index_2)
                expected_abs_index = float(i)
                assert abs(pos_index) == abs(neg_index) == expected_abs_index, (
                    f"Pattern index pair at rows {2*i} and {2*i+1} do not match expected absolute index {expected_abs_index}, "
                    f"got {index_1} at row {2*i} and {index_2} at row {2*i+1}"
                )

        # line_index = 0
        # for line in f:
        #     line_index += 1  # 1-based line index
        #     if line_index < 4:
        #         continue
        #     pattern_index_str, measured_current_ampere_str = line.split()
        #     pattern_index: int = round(float(pattern_index_str))
        #     measured_current_ampere: float = float(measured_current_ampere_str)
        #     measurements.append((pattern_index, measured_current_ampere))
        #     indices.append(pattern_index)
        #     measured_currents_ampere.append(measured_current_ampere)

    # print(f"Loaded {len(measurements)} measurements from '{measurement_file_name}'")

    # indices_np = np.array(indices, dtype=np.int64)

    # print(f"indices_np shape: {indices_np.shape}")
    # print(f"indices_np dtype: {indices_np.dtype}")
    # print(f"indices_np min: {indices_np.min()}")
    # print(f"indices_np max: {indices_np.max()}")
    # print(f"indices_np unique count: {np.unique(indices_np).size}")

    # counts: dict[int, int] = {}
    # for index in indices:
    #     counts[index] = counts.get(index, 0) + 1
    # reversed_counts: dict[int, list[int]] = {}
    # for index, count in counts.items():
    #     if count not in reversed_counts:
    #         reversed_counts[count] = []
    #     reversed_counts[count].append(index)
    # print(f"counts: {reversed_counts.keys()}")
    # for count in sorted(reversed_counts.keys()):
    #     print(f"indices with count {count}: ", end="")
    #     if len(reversed_counts[count]) <= 5:
    #         print(reversed_counts[count])
    #     else:
    #         print(f"{len(reversed_counts[count])} indices")
    # print(f"Count of 65516: {counts.get(65516, 0)}")

    # measured_currents_ampere_np = np.array(measured_currents_ampere, dtype=np.float64)

    # photocurrent_data_dir = data_dir / "photocurrent_data"
    # load_and_plot_map_image(photocurrent_data_dir, "Si_256.TXT.map")
    # load_and_plot_map_image(photocurrent_data_dir, "Si_2_256.TXT.map")

    # run_one_dataset(
    #     photocurrent_data_dir / "Si_256.TXT",
    #     photocurrent_data_dir / "Si_256.TXT.map",
    #     n=256,
    #     positive_sign=+1,   # try -1 if it looks wrong
    #     ordering=None,      # fill this in once the Hadamard ordering is known
    # )
    # run_one_dataset(
    #     photocurrent_data_dir / "Si_2_256.TXT",
    #     photocurrent_data_dir / "Si_2_256.TXT.map",
    #     n=256,
    #     positive_sign=+1,
    #     ordering=None,
    # )
