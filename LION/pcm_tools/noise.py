from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    from statsmodels.tsa.arima_process import arma_generate_sample
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for the ARIMA-like noise simulation. "
        "Install it with: pip install statsmodels"
    ) from exc


@dataclass
class BeamNoiseParams:
    """
    Parameters for the beam-related relative noise model.

    Attributes
    ----------
    ar : np.ndarray
        Autoregressive coefficients used in the MATLAB ARIMA model.
    ma : np.ndarray
        Moving-average coefficients used in the MATLAB ARIMA model.
    sigma : float
        Standard deviation parameter used as the innovation scale.
    """

    ar: np.ndarray
    ma: np.ndarray
    sigma: float


@dataclass
class BiasNoiseParams:
    """
    Parameters for the bias-light absolute noise model.

    Attributes
    ----------
    sigma : float
        Standard deviation of the bias-light random fluctuation term.
    gradient_sigma : float
        Standard deviation of the linear drift gradient.
    """

    sigma: float
    gradient_sigma: float


@dataclass
class DarkNoiseParams:
    """
    Parameters for the dark-current absolute noise model.

    Attributes
    ----------
    sigma_amps : float
        Standard deviation of additive dark noise in amperes.
    """

    sigma_amps: float


def is_power_of_two(n: int) -> bool:
    """
    Check whether an integer is a power of two.

    Parameters
    ----------
    n : int
        Integer to test.

    Returns
    -------
    bool
        True if `n` is a power of two, otherwise False.
    """
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """
    Return the smallest power of two greater than or equal to `n`.

    Parameters
    ----------
    n : int
        Input integer.

    Returns
    -------
    int
        Smallest power of two >= `n`.
    """
    if n < 1:
        raise ValueError("n must be positive.")
    return 1 << (n - 1).bit_length()


def read_current_map(filename: str | Path) -> np.ndarray:
    """
    Read a ground-truth current map from a text file.

    This mirrors the MATLAB line

    `groundTruthCurrentMap = readmatrix(..., "FileType","text");`

    Parameters
    ----------
    filename : str | Path
        Path to the map file.

    Returns
    -------
    np.ndarray
        2D floating-point array.
    """
    return np.loadtxt(filename, dtype=np.float64)


def read_pcm_data(filename: str | Path) -> pd.DataFrame:
    """
    Read a PCM measurement file.

    This mirrors the MATLAB helper `readPCMData`, which skips the first
    three lines, uses tab separation, and reads two numeric columns named
    `PatternIndex` and `MeasuredCurrentA`. :contentReference[oaicite:5]{index=5}

    Parameters
    ----------
    filename : str | Path
        Path to the PCM text file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - `PatternIndex`
        - `MeasuredCurrentA`
    """
    return pd.read_csv(
        filename,
        sep=r"\t",
        skiprows=3,
        header=None,
        names=["PatternIndex", "MeasuredCurrentA"],
        usecols=[0, 1],
        engine="python",
    )


def bitparity_uint64(x: np.ndarray) -> np.ndarray:
    """
    Compute parity of the popcount of each uint64 element.

    This follows the MATLAB helper `bitparity_uint64`, which uses repeated
    XOR folding. :contentReference[oaicite:6]{index=6}

    Parameters
    ----------
    x : np.ndarray
        Array of dtype `np.uint64`.

    Returns
    -------
    np.ndarray
        Array of dtype `np.uint8` with values 0 for even parity and 1 for
        odd parity.
    """
    x = np.asarray(x, dtype=np.uint64)
    x = np.bitwise_xor(x, x >> np.uint64(32))
    x = np.bitwise_xor(x, x >> np.uint64(16))
    x = np.bitwise_xor(x, x >> np.uint64(8))
    x = np.bitwise_xor(x, x >> np.uint64(4))
    x = np.bitwise_xor(x, x >> np.uint64(2))
    x = np.bitwise_xor(x, x >> np.uint64(1))
    p = np.bitwise_and(x, np.uint64(1))
    return p.astype(np.uint8)


def hadamard_rows(n: int, rows: np.ndarray) -> np.ndarray:
    """
    Generate selected rows of the natural-order Hadamard matrix.

    This follows the MATLAB helper `hadamardRows`, which computes rows
    on-the-fly using bitwise parity instead of forming the full matrix.
    The PDF describes:

    `"H(i,j) = (-1)^(popcount((i-1) & (j)))"` :contentReference[oaicite:7]{index=7}

    Parameters
    ----------
    n : int
        Order of the Hadamard matrix. Must be a power of two.
    rows : np.ndarray
        1-based row indices to extract, matching MATLAB conventions.

    Returns
    -------
    np.ndarray
        Array of shape `(len(rows), n)` with entries in `{-1, +1}`.
    """
    if not is_power_of_two(n):
        raise ValueError("n must be a power of two.")
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if np.any(rows < 1) or np.any(rows > n):
        raise ValueError("Row indices must lie in 1..n.")

    j = np.arange(n, dtype=np.uint64)
    i0 = (rows - 1).astype(np.uint64)[:, None]
    v = np.bitwise_and(i0, j[None, :])
    parity = bitparity_uint64(v)
    return (1 - 2 * parity).astype(np.float64)


def generate_order_sorting_matrix(o_col: int, o_row: int) -> np.ndarray:
    """
    Generate the permutation from ordered indices to Hadamard indices.

    This is a direct translation of MATLAB `generateOrderSortingMatrix`.
    The PDF says:

    `"iy gives the indices of the permutations, such that H(:,iy) gives
    the ordered Hadamard matrix."` :contentReference[oaicite:8]{index=8}

    Parameters
    ----------
    o_col : int
        Base-2 order for the number of columns.
    o_row : int
        Base-2 order for the number of rows.

    Returns
    -------
    np.ndarray
        0-based permutation array for Python indexing.
    """
    bit_swap_order = np.arange(1, o_col + o_row + 1, dtype=np.int64)[::-1]

    i1 = 2 * min(o_col, o_row)
    b1 = bit_swap_order[:i1]
    b2 = bit_swap_order[i1:]

    b11 = b1[::2]
    b12 = b1[1::2]
    b11 = np.concatenate([b11, b2])
    b12 = np.concatenate([b12, b2])

    bit_swap_order = np.concatenate([b11[:o_row], b12[:o_col]])

    n = (2**o_col) * (2**o_row)
    ix = np.arange(n, dtype=np.int64)

    width = o_col + o_row
    bit_strings = np.array(
        [list(np.binary_repr(i, width=width)) for i in ix],
        dtype="U1",
    )
    permuted = bit_strings[:, bit_swap_order - 1]
    iy = np.array(
        [int("".join(row.tolist()), 2) for row in permuted],
        dtype=np.int64,
    )
    return iy


def fast_hadamard_transform(x: np.ndarray) -> np.ndarray:
    """
    Compute the fast Hadamard transform.

    This matches the MATLAB helper `fastHadamardTransform`, which performs
    in-place butterfly updates. :contentReference[oaicite:9]{index=9}

    Parameters
    ----------
    x : np.ndarray
        1D input vector of length power of two.

    Returns
    -------
    np.ndarray
        Transformed vector `H x`.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.size
    if not is_power_of_two(n):
        raise ValueError("Input length must be a power of two.")

    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            a = x[i : i + h].copy()
            b = x[i + h : i + 2 * h].copy()
            x[i : i + h] = a + b
            x[i + h : i + 2 * h] = a - b
        h *= 2
    return x


def measure_split_hadamard_transform(
    x: np.ndarray,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute split-Hadamard measurements.

    The PDF describes the split model as:

    `"The Hadamard matrix is split into a positive part
    (where 0 replaces -1) and a negative part
    (where +1 becomes 0 and -1 becomes +1)."` :contentReference[oaicite:10]{index=10}

    The MATLAB code computes `yPlus = HRows * x` and then uses
    `HRows = 1 - HRows` for `yMinus`. That is not a literal binary split
    of a `{-1,+1}` matrix, but this function reproduces the MATLAB code
    exactly.

    Parameters
    ----------
    x : np.ndarray
        1D input vector with power-of-two length.
    block_size : int, default=256
        Number of Hadamard rows to process per block.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `y_plus`, `y_minus`.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if not is_power_of_two(n):
        raise ValueError("Input length must be a power of two.")

    y_plus = np.empty(n, dtype=np.float64)
    y_minus = np.empty(n, dtype=np.float64)

    block_size = min(block_size, n)
    if n % block_size != 0:
        raise ValueError("For this direct translation, block_size must divide n.")

    n_blocks = n // block_size
    for i_block in range(n_blocks):
        i0 = i_block * block_size + 1
        i1 = i0 + block_size - 1
        rows = np.arange(i0, i1 + 1, dtype=np.int64)

        h_rows = hadamard_rows(n, rows)
        y_plus[i0 - 1 : i1] = h_rows @ x

        h_rows_minus = 1.0 - h_rows
        y_minus[i0 - 1 : i1] = h_rows_minus @ x

    return y_plus, y_minus


def generate_synthetic_noise(
    n_meas: int,
    i_bias: float,
    beam_noise_params: BeamNoiseParams,
    bias_noise_params: BiasNoiseParams,
    dark_noise_params: DarkNoiseParams,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic noise.

    This follows the MATLAB helper `generateSyntheticNoise`. The absolute
    error combines bias drift, bias random fluctuation, and dark noise.
    The relative error is generated from an ARIMA-like model. :contentReference[oaicite:11]{index=11}

    Parameters
    ----------
    n_meas : int
        Number of measurements.
    i_bias : float
        Bias photocurrent.
    beam_noise_params : BeamNoiseParams
        Parameters for the beam-related relative noise model.
    bias_noise_params : BiasNoiseParams
        Parameters for the bias-light absolute noise model.
    dark_noise_params : DarkNoiseParams
        Parameters for the dark-current absolute noise model.
    rng : np.random.Generator | None, default=None
        Random number generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `abs_error`, `rel_error`.
    """
    if rng is None:
        rng = np.random.default_rng()

    # statsmodels uses AR polynomial convention:
    #   ar = [1, -phi1, -phi2, ...]
    # and MA polynomial convention:
    #   ma = [1, theta1, theta2, ...]
    #
    # MATLAB uses an integrated ARIMA model with D=1. A simple approximation
    # is to generate the stationary ARMA part and then cumulative-sum it once.
    ar_poly = np.r_[1.0, -np.asarray(beam_noise_params.ar, dtype=np.float64)]
    ma_poly = np.r_[1.0, np.asarray(beam_noise_params.ma, dtype=np.float64)]

    arma_part = arma_generate_sample(
        ar=ar_poly,
        ma=ma_poly,
        nsample=n_meas,
        scale=beam_noise_params.sigma,
        distrvs=rng.standard_normal,
    )
    rel_error = np.cumsum(arma_part)

    gradient = rng.standard_normal() * bias_noise_params.gradient_sigma
    idx = np.arange(1, n_meas + 1, dtype=np.float64)

    abs_error = (
        i_bias * (1.0 + idx * gradient)
        + i_bias * rng.standard_normal(n_meas) * bias_noise_params.sigma
        + rng.standard_normal(n_meas) * dark_noise_params.sigma_amps
    )
    return abs_error, rel_error


def pad_map_to_powers_of_two(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Zero-pad a 2D map so that each side length is a power of two.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image.

    Returns
    -------
    tuple[np.ndarray, int, int]
        Padded image, padded number of rows, padded number of columns.
    """
    n_rows, n_cols = image.shape
    n_row = next_power_of_two(n_rows)
    n_col = next_power_of_two(n_cols)

    out = np.zeros((n_row, n_col), dtype=np.float64)
    out[:n_rows, :n_cols] = image
    return out, n_row, n_col


def reconstruct_from_sorted_measurements(
    y_sorted: np.ndarray,
    iy: np.ndarray,
    n_row: int,
    n_col: int,
    n_rows_orig: int,
    n_cols_orig: int,
    transpose_after_reshape: bool = False,
) -> np.ndarray:
    """
    Reconstruct an image from sorted Hadamard measurements.

    Parameters
    ----------
    y_sorted : np.ndarray
        Measurements already sorted in ordered-pattern order.
    iy : np.ndarray
        Permutation mapping from ordered indices to Hadamard indices.
    n_row : int
        Padded row count.
    n_col : int
        Padded column count.
    n_rows_orig : int
        Original row count before padding.
    n_cols_orig : int
        Original column count before padding.
    transpose_after_reshape : bool, default=False
        Whether to transpose after reshape.

    Returns
    -------
    np.ndarray
        Cropped reconstructed image.
    """
    y_h = np.zeros_like(y_sorted, dtype=np.float64)
    y_h[iy] = y_sorted
    x = fast_hadamard_transform(y_h) / y_h.size

    image = x.reshape(n_row, n_col)
    if transpose_after_reshape:
        image = image.T

    return image[:n_rows_orig, :n_cols_orig]


def compute_metrics(
    image: np.ndarray,
    reference: np.ndarray,
    pixel_min: float,
) -> tuple[float, float]:
    """
    Compute SSIM and PSNR.

    This follows the MATLAB example, which computes SSIM on images scaled by
    `pixelMin`, and PSNR directly on the reconstructed and reference images.
    :contentReference[oaicite:12]{index=12}

    Parameters
    ----------
    image : np.ndarray
        Reconstructed image.
    reference : np.ndarray
        Ground-truth image.
    pixel_min : float
        Minimum pixel value in the ground-truth image.

    Returns
    -------
    tuple[float, float]
        `(ssim, psnr)`.
    """
    image_ssim = image / pixel_min
    reference_ssim = reference / pixel_min

    data_range_ssim = reference_ssim.max() - reference_ssim.min()
    data_range_psnr = reference.max() - reference.min()

    ssim_val = structural_similarity(
        image_ssim,
        reference_ssim,
        data_range=data_range_ssim,
    )
    psnr_val = peak_signal_noise_ratio(
        reference,
        image,
        data_range=data_range_psnr,
    )
    return float(ssim_val), float(psnr_val)


def subsampling_curve(
    y_sorted: np.ndarray,
    iy: np.ndarray,
    ground_truth: np.ndarray,
    pixel_min: float,
    n_row: int,
    n_col: int,
    n_subsamples: int = 128,
    transpose_after_reshape: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SSIM and PSNR as a function of sampling percentage.

    This mirrors the repeated MATLAB loop

    `for i = 1:nSubSamples ...` used for the noiseless, simulated-noisy,
    and real-noisy cases. :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}

    Parameters
    ----------
    y_sorted : np.ndarray
        Sorted measurements.
    iy : np.ndarray
        Ordered-to-Hadamard permutation.
    ground_truth : np.ndarray
        Reference image.
    pixel_min : float
        Minimum value in the reference image.
    n_row : int
        Padded row count.
    n_col : int
        Padded column count.
    n_subsamples : int, default=128
        Number of sampling levels.
    transpose_after_reshape : bool, default=False
        Whether reconstruction uses an additional transpose.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Sampling percentages, SSIM values, PSNR values.
    """
    n_rows_orig, n_cols_orig = ground_truth.shape
    y = np.zeros_like(y_sorted, dtype=np.float64)

    ssim_vals = np.zeros(n_subsamples, dtype=np.float64)
    psnr_vals = np.zeros(n_subsamples, dtype=np.float64)

    for i in range(1, n_subsamples + 1):
        n_keep = (y_sorted.size * i) // n_subsamples
        y.fill(0.0)
        y[iy[:n_keep]] = y_sorted[:n_keep]

        x = fast_hadamard_transform(y) / y.size
        image = x.reshape(n_row, n_col)
        if transpose_after_reshape:
            image = image.T
        image = image[:n_rows_orig, :n_cols_orig]

        ssim_vals[i - 1], psnr_vals[i - 1] = compute_metrics(
            image=image,
            reference=ground_truth,
            pixel_min=pixel_min,
        )

    sampling_percent = 100.0 * np.arange(1, n_subsamples + 1) / n_subsamples
    return sampling_percent, ssim_vals, psnr_vals


def run_noise_simulation_example(
    ground_truth_filename: str | Path,
    noisy_measurement_filename: str | Path,
    signal_to_bias_ratio: float = 0.3,
    divide_ground_truth_by_64: bool = True,
    rng_seed: int = 0,
) -> dict[str, object]:
    """
    Run the full synthetic-noise example.

    This is the Python equivalent of the MATLAB live script shown in the PDF.
    The PDF states that the example:
    - simulates measurement using Hadamard sampling with and without noise
    - compares the result to a real noisy measurement
    - includes subsampling simulations without regularised reconstruction
    - observes SSIM saturation around 20% sampling in the example setting
      :contentReference[oaicite:15]{index=15}

    Parameters
    ----------
    ground_truth_filename : str | Path
        Path to the ground-truth current map.
    noisy_measurement_filename : str | Path
        Path to the real noisy PCM measurement file.
    signal_to_bias_ratio : float, default=0.3
        Ratio of modulated photocurrent to bias light.
    divide_ground_truth_by_64 : bool, default=True
        Whether to divide the ground-truth map by 64, matching the MATLAB
        script.
    rng_seed : int, default=0
        Random seed.

    Returns
    -------
    dict[str, object]
        Dictionary containing reconstructed images, metrics, and curves.
    """
    rng = np.random.default_rng(rng_seed)

    # Load ground-truth map
    ground_truth_current_map = read_current_map(ground_truth_filename)
    if divide_ground_truth_by_64:
        ground_truth_current_map = ground_truth_current_map / 64.0

    n_rows, n_cols = ground_truth_current_map.shape
    total_photocurrent = float(np.sum(ground_truth_current_map))
    pixel_min = float(np.min(ground_truth_current_map))

    o_row = int(np.ceil(np.log2(n_rows)))
    o_col = int(np.ceil(np.log2(n_cols)))
    true_map, n_row, n_col = pad_map_to_powers_of_two(ground_truth_current_map)
    n = n_row * n_col
    x_true = true_map.reshape(-1)

    # Prepare ordered patterns
    iy = generate_order_sorting_matrix(o_col=o_col, o_row=o_row)

    # Simulate ideal measurement
    y_plus_no_noise, y_minus_no_noise = measure_split_hadamard_transform(x_true)
    y_plus_no_noise = y_plus_no_noise[iy]
    y_minus_no_noise = y_minus_no_noise[iy]

    y_no_noise = y_plus_no_noise - y_minus_no_noise
    image_no_noise = reconstruct_from_sorted_measurements(
        y_sorted=y_no_noise,
        iy=iy,
        n_row=n_row,
        n_col=n_col,
        n_rows_orig=n_rows,
        n_cols_orig=n_cols,
        transpose_after_reshape=False,
    )
    ssim_no_noise, psnr_no_noise = compute_metrics(
        image=image_no_noise,
        reference=ground_truth_current_map,
        pixel_min=pixel_min,
    )
    sampling_no_noise, ssim_curve_no_noise, psnr_curve_no_noise = subsampling_curve(
        y_sorted=y_no_noise,
        iy=iy,
        ground_truth=ground_truth_current_map,
        pixel_min=pixel_min,
        n_row=n_row,
        n_col=n_col,
        n_subsamples=128,
        transpose_after_reshape=False,
    )

    # Simulate noisy measurement
    i0 = total_photocurrent / signal_to_bias_ratio
    beam_noise_params = BeamNoiseParams(
        ar=np.array([-0.9221, -0.1709, -0.0769, -0.0354], dtype=np.float64),
        ma=np.array([0.0377, -0.3669], dtype=np.float64),
        sigma=1.2e-4,
    )
    bias_noise_params = BiasNoiseParams(
        sigma=2.9e-5,
        gradient_sigma=1.6e-7,
    )
    dark_noise_params = DarkNoiseParams(
        sigma_amps=3.6e-7,
    )

    e_abs, e_rel = generate_synthetic_noise(
        n_meas=2 * n,
        i_bias=i0,
        beam_noise_params=beam_noise_params,
        bias_noise_params=bias_noise_params,
        dark_noise_params=dark_noise_params,
        rng=rng,
    )

    # Randomly swap paired noise sources as in MATLAB
    i_swap = np.arange(2 * n, dtype=np.int64)
    do_swap = rng.integers(0, 2, size=n, endpoint=False)
    i_swap[0::2] += do_swap
    i_swap[1::2] -= do_swap

    interleaved_signal = np.column_stack([y_plus_no_noise, y_minus_no_noise]).reshape(
        -1
    )
    i_noise = e_abs[i_swap] + interleaved_signal * e_rel[i_swap]

    i_noise_matrix = i_noise.reshape(n, 2)
    y_plus_noise = y_plus_no_noise + i_noise_matrix[:, 0]
    y_minus_noise = y_minus_no_noise + i_noise_matrix[:, 1]
    y_noise = y_plus_noise - y_minus_noise

    image_noise = reconstruct_from_sorted_measurements(
        y_sorted=y_noise,
        iy=iy,
        n_row=n_row,
        n_col=n_col,
        n_rows_orig=n_rows,
        n_cols_orig=n_cols,
        transpose_after_reshape=False,
    )
    ssim_noise, psnr_noise = compute_metrics(
        image=image_noise,
        reference=ground_truth_current_map,
        pixel_min=pixel_min,
    )
    sampling_noise, ssim_curve_noise, psnr_curve_noise = subsampling_curve(
        y_sorted=y_noise,
        iy=iy,
        ground_truth=ground_truth_current_map,
        pixel_min=pixel_min,
        n_row=n_row,
        n_col=n_col,
        n_subsamples=128,
        transpose_after_reshape=False,
    )

    # Compare to actual noisy measurement
    measurements_df = read_pcm_data(noisy_measurement_filename)
    measurements = measurements_df.to_numpy(dtype=np.float64)

    i_pos = measurements[measurements[:, 0] > 0]
    i_neg = measurements[measurements[:, 0] < 0]

    pos_idx = i_pos[:, 0].astype(np.int64)
    neg_idx = (-i_neg[:, 0]).astype(np.int64)

    max_idx = max(pos_idx.max(initial=0), neg_idx.max(initial=0))
    pos_sum = np.bincount(pos_idx, weights=i_pos[:, 1], minlength=max_idx + 1)
    pos_count = np.bincount(pos_idx, minlength=max_idx + 1)
    neg_sum = np.bincount(neg_idx, weights=i_neg[:, 1], minlength=max_idx + 1)
    neg_count = np.bincount(neg_idx, minlength=max_idx + 1)

    pos_mean = np.divide(
        pos_sum,
        pos_count,
        out=np.zeros_like(pos_sum),
        where=pos_count > 0,
    )
    neg_mean = np.divide(
        neg_sum,
        neg_count,
        out=np.zeros_like(neg_sum),
        where=neg_count > 0,
    )

    i_meas = pos_mean[1:] - neg_mean[1:]
    y_real = np.r_[measurements[0, 1] - measurements[1, 1], i_meas]

    if y_real.size < n:
        y_real = np.pad(y_real, (0, n - y_real.size))

    image_real = reconstruct_from_sorted_measurements(
        y_sorted=y_real,
        iy=iy,
        n_row=n_col,
        n_col=n_row,
        n_rows_orig=n_rows,
        n_cols_orig=n_cols,
        transpose_after_reshape=True,
    )
    ssim_real, psnr_real = compute_metrics(
        image=image_real,
        reference=ground_truth_current_map,
        pixel_min=pixel_min,
    )
    sampling_real, ssim_curve_real, psnr_curve_real = subsampling_curve(
        y_sorted=y_real,
        iy=iy,
        ground_truth=ground_truth_current_map,
        pixel_min=pixel_min,
        n_row=n_col,
        n_col=n_row,
        n_subsamples=128,
        transpose_after_reshape=True,
    )

    return {
        "ground_truth": ground_truth_current_map,
        "image_no_noise": image_no_noise,
        "image_noise": image_noise,
        "image_real": image_real,
        "metrics": {
            "no_noise": {"ssim": ssim_no_noise, "psnr": psnr_no_noise},
            "noise": {"ssim": ssim_noise, "psnr": psnr_noise},
            "real": {"ssim": ssim_real, "psnr": psnr_real},
        },
        "curves": {
            "no_noise": {
                "sampling_percent": sampling_no_noise,
                "ssim": ssim_curve_no_noise,
                "psnr": psnr_curve_no_noise,
            },
            "noise": {
                "sampling_percent": sampling_noise,
                "ssim": ssim_curve_noise,
                "psnr": psnr_curve_noise,
            },
            "real": {
                "sampling_percent": sampling_real,
                "ssim": ssim_curve_real,
                "psnr": psnr_curve_real,
            },
        },
        "measurement_traces": {
            "simulated_noisy_interleaved": interleaved_signal + i_noise,
            "real_measurements": measurements[:, 1],
        },
        "metadata": {
            "n_rows_original": n_rows,
            "n_cols_original": n_cols,
            "n_row_padded": n_row,
            "n_col_padded": n_col,
            "n_total": n,
            "order_rows": o_row,
            "order_cols": o_col,
        },
    }


def plot_results(results: dict[str, object]) -> None:
    """
    Plot the main images and SSIM subsampling curves.

    Parameters
    ----------
    results : dict[str, object]
        Output of `run_noise_simulation_example`.
    """
    ground_truth = results["ground_truth"]
    image_no_noise = results["image_no_noise"]
    image_noise = results["image_noise"]
    image_real = results["image_real"]
    metrics = results["metrics"]
    curves = results["curves"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    images = [
        (ground_truth, "Ground-truth"),
        (
            image_no_noise,
            f"Noiseless image\nSSIM={metrics['no_noise']['ssim']:.4f}, "
            f"PSNR={metrics['no_noise']['psnr']:.2f}",
        ),
        (
            image_noise,
            f"Simulated noisy image\nSSIM={metrics['noise']['ssim']:.4f}, "
            f"PSNR={metrics['noise']['psnr']:.2f}",
        ),
        (
            image_real,
            f"Real noisy image\nSSIM={metrics['real']['ssim']:.4f}, "
            f"PSNR={metrics['real']['psnr']:.2f}",
        ),
    ]

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (img, title) in zip(axes.ravel()[:4], images):
        im = ax.imshow(-img, cmap="gray")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ax.axis("on")
    ax.plot(
        curves["no_noise"]["sampling_percent"],
        curves["no_noise"]["ssim"],
        label="No noise",
    )
    ax.plot(
        curves["noise"]["sampling_percent"],
        curves["noise"]["ssim"],
        label="Simulated noise",
    )
    ax.plot(
        curves["real"]["sampling_percent"], curves["real"]["ssim"], label="Real noise"
    )
    ax.set_xlabel("Sampling (%)")
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM vs sampling")
    ax.legend()

    ax = axes[1, 2]
    ax.axis("on")
    ax.plot(
        results["measurement_traces"]["simulated_noisy_interleaved"],
        label="Simulated noisy measurement",
    )
    ax.plot(
        results["measurement_traces"]["real_measurements"],
        label="Real noisy measurement",
        alpha=0.7,
    )
    ax.set_xlabel("Measurement")
    ax.set_ylabel("Current (A)")
    ax.set_title("Measurement traces")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Change these paths if necessary.
    ground_truth_filename = "E27 full map default settings average of 4.txt.map"
    noisy_measurement_filename = "E27 full map 1 default settings.txt"

    results = run_noise_simulation_example(
        ground_truth_filename=ground_truth_filename,
        noisy_measurement_filename=noisy_measurement_filename,
        signal_to_bias_ratio=0.3,
        divide_ground_truth_by_64=True,
        rng_seed=0,
    )
    plot_results(results)

    print("Metrics:")
    for key, value in results["metrics"].items():
        print(
            f"  {key:>8s}: "
            f"SSIM={value['ssim']:.6f}, "
            f"PSNR={value['psnr']:.3f} dB"
        )
