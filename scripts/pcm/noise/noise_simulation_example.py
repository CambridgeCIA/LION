# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Synthetic Noise Example - Python translation
#
# **Source:** *Synthetic Noise Example*, James Blakesley, NPL 2026.
#
# This notebook is a Python translation of the attached MATLAB example only. It does not use the earlier Python noise-generation code.
#
# The source states that the example is **untested**, is intended for demonstration, and should not be used for published results.
#
# The main dimensions are
#
# \[
# N_{\mathrm{row}} = 2^{\lceil \log_2(n_{\mathrm{rows}})\rceil},
# \qquad
# N_{\mathrm{col}} = 2^{\lceil \log_2(n_{\mathrm{cols}})\rceil},
# \]
#
# and
#
# \[
# N = N_{\mathrm{row}}N_{\mathrm{col}}.
# \]
#
# The Python packages below provide direct analogues of the MATLAB array, plotting, image-metric, table-reading, and ARIMA/ARMA functionality used in the source.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from statsmodels.tsa.arima_process import arma_generate_sample

# %% [markdown]
# ## Hadamard rows
#
# The source constructs selected rows of the natural-order Sylvester Hadamard matrix without forming the full matrix.
#
# For one-based row index \(i\) and zero-based column index \(j\),
#
# \[
# H_{ij}
# =
# (-1)^{\operatorname{popcount}((i-1)\mathbin{\&}j)}.
# \]
#
# Equivalently, if
#
# \[
# p_{ij}
# =
# \operatorname{popcount}((i-1)\mathbin{\&}j)\bmod 2,
# \]
#
# then
#
# \[
# H_{ij}=1-2p_{ij}\in\{+1,-1\}.
# \]
#
# The source computes the parity by repeated XOR folding.


# %%
def bitparity_uint64(x):
    """Return 0 for even popcount and 1 for odd popcount."""
    x = np.asarray(x, dtype=np.uint64).copy()
    for shift in (32, 16, 8, 4, 2, 1):
        x = np.bitwise_xor(x, x >> np.uint64(shift))
    return np.bitwise_and(x, np.uint64(1)).astype(np.uint8)


def hadamard_rows(n, rows):
    """Generate selected one-based rows of the natural-order Hadamard matrix."""
    if n <= 0 or n & (n - 1):
        raise ValueError("n must be a power of two for Sylvester construction.")

    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if np.any(rows < 1) or np.any(rows > n):
        raise ValueError("Row indices must be within 1..n.")

    j = np.arange(n, dtype=np.uint64)
    i0 = (rows - 1).astype(np.uint64)[:, None]
    v = np.bitwise_and(i0, j[None, :])

    r = 1 - 2 * bitparity_uint64(v).astype(np.int8)
    return r.astype(float)


# %% [markdown]
# ## Ordered-pattern permutation
#
# The source builds a permutation \(i_y\) by reordering the binary digits of every Hadamard index.
#
# If a zero-based index has binary representation
#
# \[
# i=(b_1b_2\ldots b_m)_2,
# \qquad
# m=o_{\mathrm{row}}+o_{\mathrm{col}},
# \]
#
# then the ordered index is obtained by the source-defined bit permutation
#
# \[
# i_y
# =
# (b_{\pi(1)}b_{\pi(2)}\ldots b_{\pi(m)})_2.
# \]
#
# MATLAB returns one-based indices. The Python function below returns the equivalent zero-based indices, so it can be used directly for NumPy indexing.


# %%
def generate_order_sorting_matrix(o_col, o_row):
    """Generate the zero-based Python equivalent of the MATLAB ordering permutation."""
    bit_swap_order = np.arange(1, o_col + o_row + 1)
    bit_swap_order = np.flip(bit_swap_order)

    i1 = 2 * min(o_col, o_row)
    b1 = bit_swap_order[:i1]
    b2 = bit_swap_order[i1:]

    b11 = b1[0::2]
    b12 = b1[1::2]

    b11 = np.concatenate([b11, b2])
    b12 = np.concatenate([b12, b2])

    bit_swap_order = np.concatenate([b11[:o_row], b12[:o_col]])

    n = 2**o_col * 2**o_row
    n_bits = o_col + o_row

    ix = np.arange(n, dtype=np.uint64)
    iy = np.zeros(n, dtype=np.uint64)

    for out_position, source_column in enumerate(bit_swap_order):
        source_position = int(source_column) - 1

        source_shift = n_bits - 1 - source_position
        destination_shift = n_bits - 1 - out_position

        bit = (ix >> np.uint64(source_shift)) & np.uint64(1)
        iy |= bit << np.uint64(destination_shift)

    return iy.astype(np.int64)


# %% [markdown]
# ## Fast Hadamard transform
#
# The source computes
#
# \[
# \mathbf y = H_N\mathbf x
# \]
#
# with the standard Hadamard butterfly. At each stage, every pair of blocks is transformed as
#
# \[
# (a,b)
# \longmapsto
# (a+b,\;a-b).
# \]
#
# For a Sylvester Hadamard matrix,
#
# \[
# H_NH_N^\top = NI,
# \]
#
# so the inverse transform is
#
# \[
# H_N^{-1}=\frac{1}{N}H_N.
# \]


# %%
def fast_hadamard_transform(x):
    """Perform the fast Hadamard transform, returning H @ x."""
    x = np.asarray(x)
    n = x.size

    if n <= 0 or n & (n - 1):
        raise ValueError("Input length must be a power of 2.")

    y = x.astype(float, copy=True)
    half = 1

    while half < n:
        blocks = y.reshape(-1, 2 * half)
        a = blocks[:, :half].copy()
        b = blocks[:, half:].copy()

        blocks[:, :half] = a + b
        blocks[:, half:] = a - b

        y = blocks.reshape(-1)
        half *= 2

    return y


# %% [markdown]
# ## Split-Hadamard measurement
#
# The prose in the source describes positive and negative split-Hadamard patterns. However, the MATLAB **code itself** computes, block by block,
#
# \[
# \mathbf y^{+}=H\mathbf x,
# \]
#
# then replaces the row block by \(1-H\) and computes
#
# \[
# \mathbf y^{-}=(1-H)\mathbf x.
# \]
#
# The Python translation below follows those code statements exactly rather than silently replacing them by a different split-Hadamard convention.
#
# The source uses blocks of at most 256 rows.


# %%
def measure_split_hadamard_transform(x):
    """Translate the source's split-Hadamard measurement code directly."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size

    if n <= 0 or n & (n - 1):
        raise ValueError("Input length must be a power of 2.")

    y_plus = x.copy()
    y_minus = x.copy()

    # Retained from the source, although this array is not subsequently used.
    x_i0 = np.zeros_like(x)

    block_size = min(n, 256)
    n_blocks = n // block_size

    for i_block in range(n_blocks):
        i0 = i_block * block_size
        i1 = i0 + block_size

        rows = np.arange(i0 + 1, i1 + 1)
        h_rows = hadamard_rows(n, rows)

        y_plus[i0:i1] = h_rows @ x

        h_rows = 1 - h_rows
        y_minus[i0:i1] = h_rows @ x

    return y_plus, y_minus


# %% [markdown]
# ## Synthetic noise model
#
# The source defines the relative beam-noise component with an integrated ARMA model,
#
# \[
# \Delta e_t^{\mathrm{rel}}
# =
# \sum_{i=1}^{p}\phi_i\Delta e_{t-i}^{\mathrm{rel}}
# +
# \varepsilon_t
# +
# \sum_{j=1}^{q}\theta_j\varepsilon_{t-j},
# \]
#
# with
#
# \[
# \varepsilon_t\sim\mathcal N(0,\sigma_{\mathrm{beam}}^2),
# \qquad
# e_t^{\mathrm{rel}}
# =
# \sum_{\ell=1}^{t}\Delta e_\ell^{\mathrm{rel}}.
# \]
#
# This corresponds to the source's ARIMA model with \(D=1\).
#
# The source draws one random bias-gradient value
#
# \[
# g
# =
# \sigma_{\mathrm{gradient}}\eta,
# \qquad
# \eta\sim\mathcal N(0,1),
# \]
#
# and defines the absolute component for measurement \(k\) as
#
# \[
# e_k^{\mathrm{abs}}
# =
# I_{\mathrm{bias}}(1+kg)
# +
# I_{\mathrm{bias}}\sigma_{\mathrm{bias}}\xi_k
# +
# \sigma_{\mathrm{dark}}\zeta_k,
# \]
#
# where
#
# \[
# \xi_k,\zeta_k\sim\mathcal N(0,1).
# \]
#
# The Python code implements the specified \(D=1\) ARIMA model as an ARMA process followed by a cumulative sum. The random realization need not match MATLAB's realization.


# %%
def generate_synthetic_noise(
    n_meas,
    i_bias,
    beam_noise_params,
    bias_noise_params,
    dark_noise_params,
    rng,
):
    """Generate the source's relative and absolute synthetic-noise components."""
    beam_ar = np.asarray(beam_noise_params["AR"], dtype=float)
    beam_ma = np.asarray(beam_noise_params["MA"], dtype=float)

    ar_polynomial = np.r_[1.0, -beam_ar]
    ma_polynomial = np.r_[1.0, beam_ma]

    arma_increment = arma_generate_sample(
        ar=ar_polynomial,
        ma=ma_polynomial,
        nsample=n_meas,
        scale=beam_noise_params["Sigma"],
        distrvs=rng.standard_normal,
    )
    rel_error = np.cumsum(arma_increment)

    gradient = rng.standard_normal() * bias_noise_params["GradientSigma"]
    measurement_index = np.arange(1, n_meas + 1, dtype=float)

    abs_error = (
        i_bias * (1 + measurement_index * gradient)
        + i_bias * rng.standard_normal(n_meas) * bias_noise_params["Sigma"]
        + rng.standard_normal(n_meas) * dark_noise_params["SigmaAmps"]
    )

    return abs_error, rel_error


# %% [markdown]
# ## PCM text-file reader
#
# Each real measurement is represented by a pair
#
# \[
# (p_j,I_j),
# \]
#
# where \(p_j\) is the signed pattern index and \(I_j\) is the measured current in amperes.
#
# The source skips the first three lines of the file and reads the two columns
#
# \[
# \texttt{PatternIndex},
# \qquad
# \texttt{MeasuredCurrentA}.
# \]


# %%
def read_pcm_data(filename):
    """Read the two-column PCM measurement file used by the source."""
    return pd.read_csv(
        filename,
        sep="\t",
        skiprows=3,
        names=["PatternIndex", "MeasuredCurrentA"],
        usecols=[0, 1],
        dtype=float,
        skip_blank_lines=False,
    )


# %% [markdown]
# # Simulation parameters and ground-truth map
#
# The source uses
#
# \[
# r_{\mathrm{signal/bias}}=0.3.
# \]
#
# After loading the current map \(X\), it applies the source-specific scale
#
# \[
# X\leftarrow \frac{X}{64}.
# \]
#
# It then computes
#
# \[
# I_{\mathrm{photo}}
# =
# \sum_{i,j}X_{ij},
# \qquad
# x_{\min}
# =
# \min_{i,j}X_{ij},
# \]
#
# pads the map to dimensions \(N_{\mathrm{row}}\times N_{\mathrm{col}}\), and vectorizes it using MATLAB column-major order:
#
# \[
# \mathbf x_{\mathrm{true}}
# =
# \operatorname{vec}(X_{\mathrm{padded}}).
# \]
#
# The file paths below are retained from the source document.

# %%
ground_truth_filename = (
    r"T:\ELECMAGN\Projects 2 new folder\EPM\A3SMARTML\Technical\Noise model"
    r"\Example implementation\Example measurements"
    r"\E27 full map default settings average of 4.txt.map"
)

noisy_measurement_filename = (
    r"T:\ELECMAGN\Projects 2 new folder\EPM\A3SMARTML\Technical\Noise model"
    r"\Example implementation\Example measurements"
    r"\E27 full map 1 default settings.txt"
)

signal_to_bias_ratio = 0.3

ground_truth_current_map = np.loadtxt(ground_truth_filename)

# Source's optional quicker-testing crop:
# ground_truth_current_map = ground_truth_current_map[199:455, 199:455]

ground_truth_current_map = ground_truth_current_map / 64

n_rows, n_cols = ground_truth_current_map.shape
n_pix = n_rows * n_cols
total_photo_current = np.sum(ground_truth_current_map)
pixel_min = np.min(ground_truth_current_map)

o_row = int(np.ceil(np.log2(n_rows)))
o_col = int(np.ceil(np.log2(n_cols)))

n_row = 2**o_row
n_col = 2**o_col
n = n_row * n_col

true_map = np.zeros((n_row, n_col), dtype=ground_truth_current_map.dtype)
true_map[:n_rows, :n_cols] = ground_truth_current_map

x_true = np.reshape(true_map, n, order="F")

# %% [markdown]
# ## Plot the ground truth
#
# The displayed image is
#
# \[
# -X,
# \]
#
# matching the source's call to `imagesc(-groundTruthCurrentMap)`.

# %%
plt.figure()
image_handle = plt.imshow(-ground_truth_current_map, cmap="gray", aspect="auto")
plt.colorbar(image_handle)
plt.title("Ground-truth image")
plt.show()

# %% [markdown]
# # Simulate ideal measurement
#
# The source first computes the two measurement vectors and then applies the ordered-pattern permutation:
#
# \[
# \mathbf y_{\mathrm{ordered}}^{+}
# =
# \mathbf y^{+}[i_y],
# \qquad
# \mathbf y_{\mathrm{ordered}}^{-}
# =
# \mathbf y^{-}[i_y].
# \]
#
# It then forms the differential measurement
#
# \[
# \mathbf y_{\mathrm{no\ noise}}
# =
# \mathbf y_{\mathrm{ordered}}^{+}
# -
# \mathbf y_{\mathrm{ordered}}^{-}.
# \]

# %%
iy = generate_order_sorting_matrix(o_col, o_row)

y_plus_no_noise, y_minus_no_noise = measure_split_hadamard_transform(x_true)

y_plus_no_noise = y_plus_no_noise[iy]
y_minus_no_noise = y_minus_no_noise[iy]

y_no_noise = y_plus_no_noise - y_minus_no_noise

# %% [markdown]
# ## Ideal reconstruction
#
# The ordered measurements are placed back at their natural Hadamard indices:
#
# \[
# y_H[i_y]=y_{\mathrm{no\ noise}}.
# \]
#
# Using
#
# \[
# H_N^{-1}
# =
# \frac{1}{N}H_N,
# \]
#
# the reconstruction is
#
# \[
# \mathbf x_{\mathrm{no\ noise}}
# =
# \frac{1}{N}H_N\mathbf y_H.
# \]
#
# The vector is reshaped using MATLAB column-major order and then cropped back to the original image size.

# %%
y_no_noise_h = np.empty_like(y_no_noise)
y_no_noise_h[iy] = y_no_noise

x_no_noise = fast_hadamard_transform(y_no_noise_h) / n

image_no_noise = np.reshape(x_no_noise, (n_row, n_col), order="F")
image_no_noise = image_no_noise[:n_rows, :n_cols]

# %% [markdown]
# ## Noiseless image metrics
#
# The source evaluates SSIM after scaling both images by the minimum ground-truth pixel value:
#
# \[
# \operatorname{SSIM}
# \left(
# \frac{X_{\mathrm{recon}}}{x_{\min}},
# \frac{X_{\mathrm{true}}}{x_{\min}}
# \right).
# \]
#
# PSNR is
#
# \[
# \operatorname{PSNR}
# =
# 10\log_{10}
# \left(
# \frac{L^2}{\operatorname{MSE}}
# \right),
# \]
#
# where MATLAB's default peak value for floating-point images is \(L=1\).
#
# `skimage` is used as the Python analogue of MATLAB's `ssim` and `psnr`; numerical values can differ slightly between implementations.

# %%
ssim_no_noise = structural_similarity(
    image_no_noise / pixel_min,
    ground_truth_current_map / pixel_min,
    data_range=1.0,
)
psnr_no_noise = peak_signal_noise_ratio(
    ground_truth_current_map,
    image_no_noise,
    data_range=1.0,
)

plt.figure()
image_handle = plt.imshow(-image_no_noise, cmap="gray", aspect="auto")
plt.colorbar(image_handle)
plt.title(f"Noiseless image\nSSIM = {ssim_no_noise}\nPSNR = {psnr_no_noise}")
plt.show()

# %% [markdown]
# ## Noiseless subsampling
#
# For subsampling step \(i=1,\ldots,128\), the source keeps
#
# \[
# m_i
# =
# \frac{N}{128}i
# \]
#
# ordered Hadamard measurements.
#
# The corresponding sampling percentage is
#
# \[
# s_i
# =
# 100\frac{i}{128}.
# \]
#
# All unmeasured Hadamard coefficients remain zero, and each intermediate reconstruction is
#
# \[
# \mathbf x_i
# =
# \frac{1}{N}H_N\mathbf y_i.
# \]

# %%
n_subsamples = 128
y = np.zeros_like(y_no_noise_h)

ssim_no_noise_curve = np.empty(n_subsamples)
psnr_no_noise_curve = np.empty(n_subsamples)

for i in range(1, n_subsamples + 1):
    n_keep = n // n_subsamples * i

    y[iy[:n_keep]] = y_no_noise[:n_keep]

    x = fast_hadamard_transform(y) / n
    im_temp = np.reshape(x, (n_row, n_col), order="F")
    im_temp = im_temp[:n_rows, :n_cols]

    ssim_no_noise_curve[i - 1] = structural_similarity(
        im_temp / pixel_min,
        ground_truth_current_map / pixel_min,
        data_range=1.0,
    )
    psnr_no_noise_curve[i - 1] = peak_signal_noise_ratio(
        ground_truth_current_map,
        im_temp,
        data_range=1.0,
    )

sampling_percent = 100 * np.arange(1, n_subsamples + 1) / n_subsamples

plt.figure()
plt.plot(sampling_percent, ssim_no_noise_curve)
plt.xlabel("Sampling (%)")
plt.ylabel("SSIM")
plt.title("No noise")
plt.show()

# %% [markdown]
# # Simulate noisy measurement
#
# The source defines the bias photocurrent as
#
# \[
# I_0
# =
# \frac{I_{\mathrm{photo}}}{r_{\mathrm{signal/bias}}}.
# \]
#
# Its numerical noise parameters are
#
# \[
# \boldsymbol\phi
# =
# (-0.9221,-0.1709,-0.0769,-0.0354),
# \]
#
# \[
# \boldsymbol\theta
# =
# (0.0377,-0.3669),
# \]
#
# \[
# \sigma_{\mathrm{beam}}=1.2\times10^{-4},
# \qquad
# \sigma_{\mathrm{bias}}=2.9\times10^{-5},
# \]
#
# \[
# \sigma_{\mathrm{gradient}}=1.6\times10^{-7},
# \qquad
# \sigma_{\mathrm{dark}}=3.6\times10^{-7}\ \mathrm A.
# \]

# %%
i0_bias = total_photo_current / signal_to_bias_ratio

beam_noise_params = {
    "AR": [-0.9221, -0.1709, -0.0769, -0.0354],
    "MA": [0.0377, -0.3669],
    "Sigma": 1.2e-4,
}

bias_noise_params = {
    "Sigma": 2.9e-5,
    "GradientSigma": 1.6e-7,
}

dark_noise_params = {
    "SigmaAmps": 3.6e-7,
}

rng = np.random.default_rng()

# %% [markdown]
# ## Add synthetic noise to the split measurements
#
# Let the ordered clean split measurements be interleaved as
#
# \[
# \mathbf s
# =
# (y_1^+,y_1^-,y_2^+,y_2^-,\ldots,y_N^+,y_N^-)^\top.
# \]
#
# For each adjacent pair, the source randomly either keeps or swaps the two noise indices. Denote this permutation by \(\pi\).
#
# The source then computes
#
# \[
# I_{\mathrm{noise},j}
# =
# e_{\pi(j)}^{\mathrm{abs}}
# +
# s_j e_{\pi(j)}^{\mathrm{rel}},
# \]
#
# and therefore the plotted noisy trace is
#
# \[
# \widetilde s_j
# =
# s_j + I_{\mathrm{noise},j}
# =
# s_j\left(1+e_{\pi(j)}^{\mathrm{rel}}\right)
# +
# e_{\pi(j)}^{\mathrm{abs}}.
# \]
#
# The interleaved noise is finally split back into its \(+\) and \(-\) components.

# %%
e_abs, e_rel = generate_synthetic_noise(
    2 * n,
    i0_bias,
    beam_noise_params,
    bias_noise_params,
    dark_noise_params,
    rng,
)

i_swap = np.arange(2 * n, dtype=np.int64)
do_swap = rng.integers(0, 2, size=n)

i_swap[0::2] += do_swap
i_swap[1::2] -= do_swap

interleaved_signal = np.column_stack([y_plus_no_noise, y_minus_no_noise]).reshape(-1)

i_noise = e_abs[i_swap] + interleaved_signal * e_rel[i_swap]
noisy_interleaved_trace = i_noise + interleaved_signal

plt.figure()
plt.plot(noisy_interleaved_trace)
plt.xlabel("Measurement")
plt.ylabel("Measurement (A)")
plt.title("Measurment with simulated noise")
plt.show()

i_noise = i_noise.reshape(n, 2)

y_plus_noise = y_plus_no_noise + i_noise[:, 0]
y_minus_noise = y_minus_no_noise + i_noise[:, 1]

# %% [markdown]
# ## Simulated noisy reconstruction
#
# The split measurements are differenced,
#
# \[
# \mathbf y_{\mathrm{noise}}
# =
# \mathbf y_{\mathrm{noise}}^+
# -
# \mathbf y_{\mathrm{noise}}^-,
# \]
#
# unsorted back to natural Hadamard order,
#
# \[
# y_{\mathrm{noise},H}[i_y]
# =
# y_{\mathrm{noise}},
# \]
#
# and reconstructed as
#
# \[
# \mathbf x_{\mathrm{noise}}
# =
# \frac{1}{N}H_N\mathbf y_{\mathrm{noise},H}.
# \]

# %%
y_noise = y_plus_noise - y_minus_noise

y_noise_h = np.empty_like(y_noise)
y_noise_h[iy] = y_noise

x_noise = fast_hadamard_transform(y_noise_h) / n

image_noise = np.reshape(x_noise, (n_row, n_col), order="F")
image_noise = image_noise[:n_rows, :n_cols]

# %% [markdown]
# ## Simulated noisy image metrics
#
# As in the noiseless case, the source computes
#
# \[
# \operatorname{SSIM}
# \left(
# \frac{X_{\mathrm{noise}}}{x_{\min}},
# \frac{X_{\mathrm{true}}}{x_{\min}}
# \right)
# \]
#
# and
#
# \[
# \operatorname{PSNR}
# =
# 10\log_{10}
# \left(
# \frac{1}{\operatorname{MSE}}
# \right)
# \]
#
# for floating-point images with the default peak value \(1\).

# %%
ssim_noise = structural_similarity(
    image_noise / pixel_min,
    ground_truth_current_map / pixel_min,
    data_range=1.0,
)
psnr_noise = peak_signal_noise_ratio(
    ground_truth_current_map,
    image_noise,
    data_range=1.0,
)

plt.figure()
image_handle = plt.imshow(-image_noise, cmap="gray", aspect="auto")
plt.colorbar(image_handle)
plt.title(f"Simulated noisy image\nSSIM = {ssim_noise}\nPSNR = {psnr_noise}")
plt.show()

# %% [markdown]
# ## Noisy subsampling
#
# The same cumulative ordered subsampling rule is used:
#
# \[
# m_i
# =
# \frac{N}{128}i,
# \qquad
# s_i
# =
# 100\frac{i}{128}.
# \]
#
# At each step,
#
# \[
# \mathbf x_i
# =
# \frac{1}{N}H_N\mathbf y_i,
# \]
#
# where only the first \(m_i\) ordered noisy measurements have been inserted into \(\mathbf y_i\).

# %%
n_subsamples = 128
y = np.zeros_like(y_noise_h)

ssim_noise_curve = np.empty(n_subsamples)
psnr_noise_curve = np.empty(n_subsamples)

for i in range(1, n_subsamples + 1):
    n_keep = n // n_subsamples * i

    y[iy[:n_keep]] = y_noise[:n_keep]

    x = fast_hadamard_transform(y) / n
    im_temp = np.reshape(x, (n_row, n_col), order="F")
    im_temp = im_temp[:n_rows, :n_cols]

    ssim_noise_curve[i - 1] = structural_similarity(
        im_temp / pixel_min,
        ground_truth_current_map / pixel_min,
        data_range=1.0,
    )
    psnr_noise_curve[i - 1] = peak_signal_noise_ratio(
        ground_truth_current_map,
        im_temp,
        data_range=1.0,
    )

sampling_percent = 100 * np.arange(1, n_subsamples + 1) / n_subsamples

plt.figure()
plt.plot(sampling_percent, ssim_noise_curve)
plt.xlabel("Sampling (%)")
plt.ylabel("SSIM")
plt.title("With noise")
plt.show()

# %% [markdown]
# # Compare with an actual noisy measurement
#
# The real measurement file contains signed pattern indices \(p_j\) and currents \(I_j\).
#
# The source separates the rows into
#
# \[
# \mathcal P
# =
# \{(p_j,I_j):p_j>0\},
# \qquad
# \mathcal N
# =
# \{(p_j,I_j):p_j<0\}.
# \]
#
# The raw measured-current sequence is plotted before reconstruction.

# %%
measurements_table = read_pcm_data(noisy_measurement_filename)
measurements = measurements_table.to_numpy()

i_pos = measurements[measurements[:, 0] > 0]
i_neg = measurements[measurements[:, 0] < 0]

plt.figure()
plt.plot(measurements[:, 1])
plt.xlabel("Measurement")
plt.ylabel("Current (A)")
plt.title("Real measurment with noise")
plt.show()

# %% [markdown]
# ## Average repeated real measurements
#
# For each positive pattern index \(k\), the source averages repeated measurements:
#
# \[
# \bar I_k^+
# =
# \frac{1}{|\mathcal P_k|}
# \sum_{j\in\mathcal P_k} I_j.
# \]
#
# For the corresponding negative pattern,
#
# \[
# \bar I_k^-
# =
# \frac{1}{|\mathcal N_k|}
# \sum_{j\in\mathcal N_k} I_j.
# \]
#
# The differential measurement is
#
# \[
# y_k^{\mathrm{real}}
# =
# \bar I_k^+
# -
# \bar I_k^-.
# \]
#
# The source handles the zero-index Hadamard measurement separately using the first two rows:
#
# \[
# y_0^{\mathrm{real}}
# =
# I_1-I_2.
# \]
#
# If fewer than \(N\) entries are available, the remainder is filled with zeros.

# %%
positive_indices = i_pos[:, 0].astype(int)
negative_indices = (-i_neg[:, 0]).astype(int)

max_positive_index = positive_indices.max() if positive_indices.size else 0
max_negative_index = negative_indices.max() if negative_indices.size else 0
max_pattern_index = int(max(max_positive_index, max_negative_index))

positive_sum = np.bincount(
    positive_indices,
    weights=i_pos[:, 1],
    minlength=max_pattern_index + 1,
)
positive_count = np.bincount(
    positive_indices,
    minlength=max_pattern_index + 1,
)

negative_sum = np.bincount(
    negative_indices,
    weights=i_neg[:, 1],
    minlength=max_pattern_index + 1,
)
negative_count = np.bincount(
    negative_indices,
    minlength=max_pattern_index + 1,
)

positive_mean = np.zeros(max_pattern_index + 1)
negative_mean = np.zeros(max_pattern_index + 1)

np.divide(
    positive_sum,
    positive_count,
    out=positive_mean,
    where=positive_count > 0,
)
np.divide(
    negative_sum,
    negative_count,
    out=negative_mean,
    where=negative_count > 0,
)

i_meas = positive_mean[1:] - negative_mean[1:]

y_real = np.concatenate(([measurements[0, 1] - measurements[1, 1]], i_meas))

if y_real.size < n:
    y_real = np.pad(y_real, (0, n - y_real.size))

# %% [markdown]
# ## Real-data reconstruction
#
# As before,
#
# \[
# y_H^{\mathrm{real}}[i_y]
# =
# y^{\mathrm{real}},
# \]
#
# and
#
# \[
# \mathbf x_{\mathrm{real}}
# =
# \frac{1}{N}H_N\mathbf y_H^{\mathrm{real}}.
# \]
#
# The source reshapes the real-data reconstruction differently from the simulated reconstructions:
#
# \[
# X_{\mathrm{real}}
# =
# \operatorname{reshape}
# (\mathbf x_{\mathrm{real}},N_{\mathrm{col}},N_{\mathrm{row}})^\top.
# \]
#
# The Python code preserves this exact MATLAB reshape/transposition convention.

# %%
y_real_h = np.empty(n, dtype=float)
y_real_h[iy] = y_real

x_real = fast_hadamard_transform(y_real_h) / n

image_real = np.reshape(x_real, (n_col, n_row), order="F").T
image_real = image_real[:n_rows, :n_cols]

# %% [markdown]
# ## Real noisy image metrics
#
# The source again computes
#
# \[
# \operatorname{SSIM}
# \left(
# \frac{X_{\mathrm{real}}}{x_{\min}},
# \frac{X_{\mathrm{true}}}{x_{\min}}
# \right)
# \]
#
# and
#
# \[
# \operatorname{PSNR}
# =
# 10\log_{10}
# \left(
# \frac{1}{\operatorname{MSE}}
# \right).
# \]

# %%
ssim_real = structural_similarity(
    image_real / pixel_min,
    ground_truth_current_map / pixel_min,
    data_range=1.0,
)
psnr_real = peak_signal_noise_ratio(
    ground_truth_current_map,
    image_real,
    data_range=1.0,
)

plt.figure()
image_handle = plt.imshow(-image_real, cmap="gray", aspect="auto")
plt.colorbar(image_handle)
plt.title(f"Real noisy image\nSSIM = {ssim_real}\nPSNR = {psnr_real}")
plt.show()

# %% [markdown]
# ## Real-data subsampling
#
# The source applies the same ordered cumulative sampling schedule to the real data:
#
# \[
# m_i
# =
# \frac{N}{128}i,
# \qquad
# s_i
# =
# 100\frac{i}{128}.
# \]
#
# The source again uses the real-data reshape convention
#
# \[
# X_i
# =
# \operatorname{reshape}
# (\mathbf x_i,N_{\mathrm{col}},N_{\mathrm{row}})^\top.
# \]

# %%
n_subsamples = 128
y = np.zeros_like(y_real_h)

ssim_real_curve = np.empty(n_subsamples)
psnr_real_curve = np.empty(n_subsamples)

for i in range(1, n_subsamples + 1):
    n_keep = n // n_subsamples * i

    y[iy[:n_keep]] = y_real[:n_keep]

    x = fast_hadamard_transform(y) / n
    im_temp = np.reshape(x, (n_col, n_row), order="F").T
    im_temp = im_temp[:n_rows, :n_cols]

    ssim_real_curve[i - 1] = structural_similarity(
        im_temp / pixel_min,
        ground_truth_current_map / pixel_min,
        data_range=1.0,
    )
    psnr_real_curve[i - 1] = peak_signal_noise_ratio(
        ground_truth_current_map,
        im_temp,
        data_range=1.0,
    )

sampling_percent = 100 * np.arange(1, n_subsamples + 1) / n_subsamples

plt.figure()
plt.plot(sampling_percent, ssim_real_curve)
plt.xlabel("Sampling (%)")
plt.ylabel("SSIM")
plt.title("With noise")
plt.show()
