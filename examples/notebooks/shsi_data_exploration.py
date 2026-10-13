# %% [markdown]
# # Explore the sHSI measurements and the masked forward model
#
# This example only loads data and draws figures. It does not train a model,
# run a reconstruction algorithm, or download any data.
#
# The source team's data has shape (height, width, spectral samples).
# Its measurement model is y = M * x + noise at observed entries.

# %% Imports and settings
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from LION.data_loaders.shsi.data import load_shsi_npz
from LION.operators.MaskedObservationOp import MaskedObservationOp

# Edit these settings in a notebook, or use the environment variables.
DATA_DIR = Path(
    os.environ.get(
        "LION_SHSI_DATA_DIR",
        "/mnt/store_ext4/store/LION/ttv22/shsi/data",
    )
)
EXAMPLE_NAME = os.environ.get("LION_SHSI_EXAMPLE", "test_input_0.npz")
SAVE_FIGURES = os.environ.get("SHSI_SAVE_FIGURES", "0") == "1"
SHOW_FIGURES = os.environ.get("SHSI_SHOW_FIGURES", "1") == "1"

# Leave these as None to select a strong signal channel and spatial pixel.
SPECTRAL_INDEX: int | None = None
PIXEL: tuple[int, int] | None = None

FIGURES_DIR = DATA_DIR / "visualizations" / Path(EXAMPLE_NAME).stem


def finish_figure(fig: plt.Figure, filename: str) -> None:
    """Optionally save and display a figure."""
    if SAVE_FIGURES:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved: {path}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# %% Load the team's reference cube
data_path = DATA_DIR / "examples" / EXAMPLE_NAME
if not data_path.exists():
    raise FileNotFoundError(
        f"Example data not found: {data_path}\n"
        "Set LION_SHSI_DATA_DIR to the sHSI data workspace."
    )

sample = load_shsi_npz(data_path)
if sample.ground_truth is None:
    raise ValueError("This visual exploration requires an NPZ file containing 'gt'.")
if sample.measurement.ndim != 3:
    raise ValueError("The example requires a 3D hyperspectral cube.")

measurement = sample.measurement
mask = sample.mask
ground_truth = sample.ground_truth
height, width, num_spectral_samples = ground_truth.shape

print(f"Dataset: {data_path.name}")
print(f"Cube shape: {ground_truth.shape} (height, width, spectral samples)")
print(f"Observed: {mask.sum():,} / {mask.size:,} entries ({100 * mask.mean():.2f}%)")
print(f"Provided measurement noise sigma: {sample.sigma:.6g}")
print(f"Data type: {ground_truth.dtype}")

# %% Apply LION's sHSI forward operator
# With the same sampling mask, A(x) = M * x.
# The difference between the observed data and A(ground truth) is measurement error.
operator = MaskedObservationOp(mask)

with torch.no_grad():
    noiseless_measurement = operator(torch.from_numpy(ground_truth)).numpy()
    adjoint_measurement = operator.adjoint(torch.from_numpy(measurement)).numpy()

observed_residual = (measurement - noiseless_measurement)[mask]
print("\nForward model:")
print(f"Operator domain: {operator.domain_shape}")
print(f"Operator range:  {operator.range_shape}")
print(
    f"A(ground truth) is zero at missing entries: {np.all(noiseless_measurement[~mask] == 0)}"
)
print(f"Measurements are zero at missing entries: {np.all(measurement[~mask] == 0)}")
print(
    f"A*(measured data) equals measured data: {np.array_equal(adjoint_measurement, measurement)}"
)
print(f"Mean observed residual: {observed_residual.mean():.6g}")
print(f"Std observed residual:  {observed_residual.std():.6g}")
if not np.any(observed_residual):
    print(
        "Note: all observed values equal the masked ground truth, despite the nonzero sigma metadata."
    )

# %% Choose a representative spectral channel and spatial pixel
if SPECTRAL_INDEX is None:
    spectral_strength = np.mean(np.abs(ground_truth), axis=(0, 1))
    spectral_index = int(np.argmax(spectral_strength))
else:
    spectral_index = int(SPECTRAL_INDEX)
if not 0 <= spectral_index < num_spectral_samples:
    raise ValueError("SPECTRAL_INDEX must be within the spectral dimension.")

if PIXEL is None:
    strength_map = np.mean(np.abs(ground_truth), axis=2)
    pixel = tuple(
        int(i) for i in np.unravel_index(np.argmax(strength_map), strength_map.shape)
    )
else:
    pixel = PIXEL
row, col = pixel
if not (0 <= row < height and 0 <= col < width):
    raise ValueError("PIXEL lies outside the spatial image.")

print(f"Displayed spectral channel: {spectral_index}")
print(f"Displayed pixel: (row={row}, col={col})")

# %% Spatial view: what is observed at one spectral channel?
slice_mask = mask[:, :, spectral_index]
slice_gt = ground_truth[:, :, spectral_index]
slice_forward = noiseless_measurement[:, :, spectral_index]
slice_observed = measurement[:, :, spectral_index]

# The masked array hides missing positions. A missing measurement is not
# the same thing as a measured value of zero.
forward_visible = np.ma.masked_where(~slice_mask, slice_forward)
observed_visible = np.ma.masked_where(~slice_mask, slice_observed)

scale = max(float(np.quantile(np.abs(slice_gt), 0.99)), 1e-12)
cmap = plt.get_cmap("coolwarm").with_extremes(bad="lightgray")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.7), constrained_layout=True)
titles = [
    "Ground truth",
    "Sampling mask",
    "Forward model A(x)",
    "Observed measurements",
]
images = [slice_gt, slice_mask, forward_visible, observed_visible]

for index, (ax, title, image) in enumerate(zip(axes, titles, images)):
    if index == 1:
        im = ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        im = ax.imshow(
            image, cmap=cmap, vmin=-scale, vmax=scale, interpolation="nearest"
        )
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    if index != 1:
        fig.colorbar(im, ax=ax, fraction=0.047, pad=0.04)

fig.suptitle(f"{EXAMPLE_NAME} | spectral channel {spectral_index}")
finish_figure(fig, "spatial_forward_model.png")

# %% Spectra, sampling coverage, and measurement errors
spectral_positions = np.arange(num_spectral_samples)
observed_at_pixel = mask[row, col, :]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

axes[0, 0].plot(
    spectral_positions, ground_truth[row, col, :], label="Ground truth", linewidth=1.5
)
axes[0, 0].scatter(
    spectral_positions[observed_at_pixel],
    measurement[row, col, observed_at_pixel],
    label="Measured entries",
    s=15,
    zorder=3,
)
axes[0, 0].set_title(f"Signal at pixel (row={row}, col={col})")
axes[0, 0].set_xlabel("Spectral sample index")
axes[0, 0].set_ylabel("Signal")
axes[0, 0].legend()

coverage_by_channel = mask.mean(axis=(0, 1))
axes[0, 1].plot(spectral_positions, 100 * coverage_by_channel)
axes[0, 1].axhline(100 * mask.mean(), linestyle="--", label="Overall sampling rate")
axes[0, 1].set_title("Observed fraction per spectral channel")
axes[0, 1].set_xlabel("Spectral sample index")
axes[0, 1].set_ylabel("Observed (%)")
axes[0, 1].legend()

spatial_coverage = mask.mean(axis=2)
heatmap = axes[1, 0].imshow(spatial_coverage, cmap="viridis", vmin=0, vmax=1)
axes[1, 0].set_title("Fraction of observed channels per pixel")
axes[1, 0].set_xlabel("Column")
axes[1, 0].set_ylabel("Row")
fig.colorbar(heatmap, ax=axes[1, 0], fraction=0.047, pad=0.04)

if np.any(observed_residual):
    axes[1, 1].hist(observed_residual, bins=60, density=True)
else:
    axes[1, 1].text(
        0.5,
        0.75,
        "All observed residuals are zero",
        ha="center",
        va="center",
        transform=axes[1, 1].transAxes,
    )
axes[1, 1].axvline(0, linestyle="--", label="Zero")
axes[1, 1].axvline(sample.sigma, linestyle=":", label="Provided +sigma")
axes[1, 1].axvline(-sample.sigma, linestyle=":", label="Provided -sigma")
axes[1, 1].set_title("Measured data minus forward model, observed entries")
axes[1, 1].set_xlabel("Residual")
axes[1, 1].set_ylabel("Density")
axes[1, 1].legend()

finish_figure(fig, "spectra_sampling_and_noise.png")

# %% Three spatial slices, to show that a cube is not one image
slice_indices = sorted(
    {
        num_spectral_samples // 4,
        num_spectral_samples // 2,
        3 * num_spectral_samples // 4,
    }
)
fig, axes = plt.subplots(
    1, len(slice_indices), figsize=(10, 3.5), constrained_layout=True
)
for ax, index in zip(np.atleast_1d(axes), slice_indices):
    im = ax.imshow(ground_truth[:, :, index], cmap="coolwarm", interpolation="nearest")
    ax.set_title(f"Ground truth, channel {index}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, fraction=0.047, pad=0.04)

finish_figure(fig, "ground_truth_spectral_slices.png")
