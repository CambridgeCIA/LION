"""Plotting utilities for noise simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

from LION.pcm_tools.noise_metrics import compute_metrics_hybrid
from LION.pcm_tools.noise_subsampling_curve import SubsamplingCurve


def plot_results(
    ground_truth: torch.Tensor,
    image_no_noise: torch.Tensor,
    image_noise: torch.Tensor,
    image_real: torch.Tensor,
    curves: dict[str, SubsamplingCurve],
    measurement_traces: dict[str, np.ndarray],
    output_filepath: str | Path,
    suptitle: str,
    dpi: int = 200,
) -> None:
    """
    Plot images, SSIM curves, and measurement traces.

    Args:
        ground_truth: Ground-truth image.
        image_no_noise: Noiseless image.
        image_noise: Simulated noisy image.
        image_real: Real noisy image.
        curves: Dictionary containing 'sampling_percent' and 'ssim' arrays for each noise type
        measurement_traces: Dictionary containing measurement traces for simulated and real noise.
        output_filepath: Output figure filepath.
        suptitle: Suptitle for the figure.
    """
    pixel_min = ground_truth.min()

    no_noise_ssim, no_noise_psnr = compute_metrics_hybrid(
        image_no_noise, reference=ground_truth, pixel_min=pixel_min
    )
    noise_ssim, noise_psnr = compute_metrics_hybrid(
        image_noise, reference=ground_truth, pixel_min=pixel_min
    )
    real_ssim, real_psnr = compute_metrics_hybrid(
        image_real, reference=ground_truth, pixel_min=pixel_min
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    plt.axis("off")  # Turn on axes later when necessary

    vmin = torch.quantile(-ground_truth.cpu(), 0.005)
    vmax = torch.quantile(-ground_truth.cpu(), 0.995)
    plot_image(axes[0, 0], ground_truth.cpu(), vmin, vmax, "Ground-truth image", fig)
    plot_image(
        axes[0, 1],
        image_no_noise.cpu(),
        vmin,
        vmax,
        f"Noiseless image\nSSIM={no_noise_ssim:.4f}, PSNR={no_noise_psnr:.2f}",
        fig,
    )
    plot_image(
        axes[0, 2],
        image_noise.cpu(),
        vmin,
        vmax,
        f"Simulated noisy image\nSSIM={noise_ssim:.4f}, PSNR={noise_psnr:.2f}",
        fig,
    )
    plot_image(
        axes[1, 0],
        image_real.cpu(),
        vmin,
        vmax,
        f"Real noisy image\nSSIM={real_ssim:.4f}, PSNR={real_psnr:.2f}",
        fig,
    )

    plot_ssim_vs_sampling(axes[1, 1], curves)
    plot_measurement_traces(axes[1, 2], measurement_traces)

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to '{output_filepath}'")


def plot_image(
    ax: Axes, img: torch.Tensor, vmin: float, vmax: float, title: str, fig: plt.Figure
) -> None:
    """Plot an image with a colorbar.

    Args:
        ax: Matplotlib Axes to plot on.
        img: Image to plot.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        title: Title for the image.
        fig: Matplotlib Figure to add the colorbar to.
    """
    im = ax.imshow(-img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_ssim_vs_sampling(ax: Axes, curves: dict[str, SubsamplingCurve]) -> None:
    """Plot SSIM vs sampling curves.

    Args:
        ax: Matplotlib Axes to plot on.
        curves: Dictionary containing 'sampling_percent' and 'ssim' arrays for each noise type.
    """
    ax.axis("on")

    for key, label in [
        ("no_noise", "No noise"),
        ("sim_noise", "Simulated noise"),
        ("real_noise", "Real noise"),
    ]:
        sampling_percent = curves[key].sampling_percent
        ssim_values = curves[key].ssim_vals

        if sampling_percent.size > 0 and ssim_values.size > 0:
            ax.plot(sampling_percent, ssim_values, label=label)

    ax.set_xlabel("Sampling (%)")
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM vs sampling")
    ax.legend()


def plot_measurement_traces(
    ax: Axes, measurement_traces: dict[str, np.ndarray]
) -> None:
    """Plot measurement traces for simulated and real noise.

    Args:
        ax: Matplotlib Axes to plot on.
        measurement_traces: Dictionary containing measurement traces for simulated and real noise.
    """
    ax.axis("on")
    ax.plot(
        measurement_traces["simulated_noisy_interleaved"],
        label="Simulated noisy measurement",
    )
    ax.plot(
        measurement_traces["real_measurements"],
        label="Real noisy measurement",
        alpha=0.7,
    )
    ax.set_xlabel("Measurement")
    ax.set_ylabel("Current (A)")
    ax.set_title("Measurement traces")
    ax.legend()


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to a NumPy array.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    np.ndarray
        CPU NumPy array.
    """
    return x.detach().cpu().numpy()
