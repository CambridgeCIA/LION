"""Diagnose the contributions of the PCM beam, bias, and dark noise components.

This script compares the provided real residual

    E27-full-map-1-default-settings.txt.map
    - E27-full-map-default-settings-average-of-4.txt.map

against simulated full-acquisition residuals generated from the provided
average-of-4 map using:

- all noise components;
- beam noise only;
- bias noise only;
- dark noise only.

Each simulated residual is compared directly against the provided average-of-4
map in the original measured field of view.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from LION.data_loaders.pcm.data_primitives import PCMScalingParameter
from LION.pcm.noise import PCMNoiseSimulation
from LION.pcm.simulation import simulate_pcm_acquisition
from LION.utils.paths import PCM_REAL_DATA_EXAMPLE_PATH
from LION.utils.shape import pad_to_powers_of_two_torch

J_ORDER = 10
DIVIDE_BY = 64.0
# R_LOW = -4e-8
# R_LOW = -3.5e-8
# R_HIGH = 0.0
R_LOW = 0.0
R_HIGH = 3.5e-8
DATA_SIGN = -1
RNG_SEED = 0

DATA_DIR = PCM_REAL_DATA_EXAMPLE_PATH / "E27" / "pos1"
SINGLE_MAP_PATH = DATA_DIR / "E27-full-map-1-default-settings.txt.map"
AVERAGE_MAP_PATH = DATA_DIR / "E27-full-map-default-settings-average-of-4.txt.map"

OUTPUT_DIR = Path("results/pcm/noise_diagnostics/e27_pos1/components")


def load_map(path: Path) -> torch.Tensor:
    """Load one NPL current map and apply the configured map correction."""
    return DATA_SIGN * torch.from_numpy(np.loadtxt(path, dtype=np.float64) / DIVIDE_BY)


def noise_stats(x: np.ndarray) -> dict[str, float]:
    """Return scalar statistics for one residual image."""
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "mean_abs": float(np.mean(np.abs(flat))),
        "min": float(np.min(flat)),
        "q01": float(np.quantile(flat, 0.01)),
        "q05": float(np.quantile(flat, 0.05)),
        "median": float(np.median(flat)),
        "q95": float(np.quantile(flat, 0.95)),
        "q99": float(np.quantile(flat, 0.99)),
        "max": float(np.max(flat)),
    }


def save_noise_image(noise: np.ndarray, path: Path, *, vmax: float) -> None:
    """Save one zero-centred residual image."""
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(noise, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_axis_off()
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def radial_power_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a radial average of the 2D power spectrum."""
    x = np.asarray(image, dtype=np.float64)
    x = x - np.mean(x)

    spectrum = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(spectrum) ** 2 / x.size

    height, width = x.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2).astype(int)

    radial_sum = np.bincount(radius.ravel(), weights=power.ravel())
    radial_count = np.bincount(radius.ravel())
    radial_power = radial_sum / np.maximum(radial_count, 1)

    frequency = np.arange(len(radial_power), dtype=np.float64) / min(height, width)
    return frequency, radial_power


def make_component_models(
    base_noise: PCMNoiseSimulation,
) -> dict[str, PCMNoiseSimulation]:
    """Return all-noise and one-component-only versions of the NPL model."""
    p = base_noise.noise_params

    beam_only = replace(
        base_noise,
        noise_params=replace(
            p,
            bias_sigma=None,
            bias_gradient_sigma=None,
            dark_sigma=None,
        ),
    )
    bias_only = replace(
        base_noise,
        noise_params=replace(
            p,
            beam_ar=None,
            beam_ma=None,
            beam_sigma=None,
            dark_sigma=None,
        ),
    )
    dark_only = replace(
        base_noise,
        noise_params=replace(
            p,
            beam_ar=None,
            beam_ma=None,
            beam_sigma=None,
            bias_sigma=None,
            bias_gradient_sigma=None,
        ),
    )

    return {
        "all": base_noise,
        "beam_only": beam_only,
        "bias_only": bias_only,
        "dark_only": dark_only,
    }


def simulate_residual(
    *,
    average_map_model: torch.Tensor,
    average_map: torch.Tensor,
    original_region_mask: torch.Tensor,
    scaling: PCMScalingParameter,
    full_indices: np.ndarray,
    noise_model: PCMNoiseSimulation,
    seed: int,
) -> np.ndarray:
    """Simulate one full acquisition and subtract the provided average map."""
    acquisition = simulate_pcm_acquisition(
        average_map_model,
        j_order=J_ORDER,
        sampled_indices=full_indices,
        noise=noise_model,
        rng=np.random.default_rng(seed),
        device="cpu",
        seed=seed,
    )
    recon_model = acquisition.op.pseudo_inv(acquisition.measurement)
    simulated_map = scaling.rescale(recon_model).squeeze()

    simulated_roi = simulated_map[original_region_mask].reshape(average_map.shape)
    return (simulated_roi - average_map).numpy()


def main() -> None:
    """Run the component-wise full-acquisition noise diagnostic."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    single_map = load_map(SINGLE_MAP_PATH)
    average_map = load_map(AVERAGE_MAP_PATH)
    assert single_map.shape == average_map.shape

    real_residual = (single_map - average_map).numpy()

    scaling = PCMScalingParameter(
        r_low=R_LOW,
        r_high=R_HIGH,
        divide_by=DIVIDE_BY,
        fill_value=0.0,
    )

    average_map_padded = pad_to_powers_of_two_torch(
        average_map,
        dim=(-2, -1),
        fill_value=scaling.fill_value,
    )
    original_region_mask = pad_to_powers_of_two_torch(
        torch.ones_like(average_map),
        dim=(-2, -1),
        fill_value=0.0,
    ).bool()

    average_map_model = scaling.scale(average_map_padded).unsqueeze(0).unsqueeze(0)

    n = 1 << J_ORDER
    full_indices = np.arange(n * n)

    base_noise = PCMNoiseSimulation.npl_default(
        current_to_model_scale=1.0 / (R_HIGH - R_LOW),
        j_order=J_ORDER,
        n_repeat_acquisitions=1,
    )
    component_models = make_component_models(base_noise)

    residuals: dict[str, np.ndarray] = {
        "real_single_minus_provided_average4": real_residual,
    }

    for name, noise_model in component_models.items():
        residuals[f"sim_{name}_minus_provided_average4"] = simulate_residual(
            average_map_model=average_map_model,
            average_map=average_map,
            original_region_mask=original_region_mask,
            scaling=scaling,
            full_indices=full_indices,
            noise_model=noise_model,
            seed=RNG_SEED,
        )

    rows = [
        {"noise": name, **noise_stats(residual)} for name, residual in residuals.items()
    ]

    stats_path = OUTPUT_DIR / "noise_component_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stats = {row["noise"]: row for row in rows}
    real_std = stats["real_single_minus_provided_average4"]["std"]
    all_std = stats["sim_all_minus_provided_average4"]["std"]

    print()
    print("PCM noise-component diagnostic")
    print("==============================")
    for row in rows:
        ratio = row["std"] / real_std
        print(
            f"{row['noise']:>42}: "
            f"mean={row['mean']:.6e}, "
            f"std={row['std']:.6e}, "
            f"rms={row['rms']:.6e}, "
            f"std/real={ratio:.6f}"
        )

    component_names = ("beam_only", "bias_only", "dark_only")
    component_variance_sum = sum(
        stats[f"sim_{name}_minus_provided_average4"]["std"] ** 2
        for name in component_names
    )
    quadrature_std = float(np.sqrt(component_variance_sum))

    print()
    print(f"all-noise std                         = {all_std:.6e}")
    print(f"quadrature(component stds)            = {quadrature_std:.6e}")
    print(f"quadrature / all-noise std            = {quadrature_std / all_std:.6f}")

    for name in component_names:
        component_std = stats[f"sim_{name}_minus_provided_average4"]["std"]
        print(
            f"{name:>12} variance / all variance      = "
            f"{(component_std / all_std) ** 2:.6f}"
        )

    common_vmax = max(float(np.max(np.abs(x))) for x in residuals.values())
    for name, residual in residuals.items():
        np.save(OUTPUT_DIR / f"{name}.npy", residual)
        save_noise_image(residual, OUTPUT_DIR / f"{name}.png", vmax=common_vmax)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, residual in residuals.items():
        ax.hist(
            residual.reshape(-1),
            bins=200,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=name,
        )
    ax.set_xlabel("Noise value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "noise_component_histograms.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, residual in residuals.items():
        frequency, power = radial_power_spectrum(residual)
        ax.loglog(frequency[1:], power[1:], label=name)
    ax.set_xlabel("Radial spatial frequency")
    ax.set_ylabel("Power")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "noise_component_radial_power_spectra.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print()
    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
