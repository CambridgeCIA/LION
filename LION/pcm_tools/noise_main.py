"""Main code for PCM noise simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from LION.experiments.pcm_experiments import PCMNoiseSimulation
from LION.pcm_tools.noise_plot_results import plot_results
from LION.pcm_tools.noise_run_simulation import run_noise_simulation


def get_torch_device() -> torch.device:
    """Get a torch device (GPU if available, otherwise CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main() -> None:
    """Run the script entry point."""
    parser = argparse.ArgumentParser(description="Run PCM noise simulation.")
    parser.add_argument(
        "--test-case",
        type=str,
        choices=["default", "extrapolate"],
        default="default",
        help=(
            "Which test case to run. 'default' uses noise parameters from a real measurement, "
            "while 'extrapolate' uses more extreme noise parameters to better demonstrate the effects of noise."
        ),
    )
    args = parser.parse_args()

    if args.test_case == "default":
        example_dir = Path("example_data/E27/pos1")
        noise_simulation_params = PCMNoiseSimulation.simulation_default()
        divide_by = 64.0  # For some reason the maps are sometimes multiplied by a factor of 64 - check measurment code!
    elif args.test_case == "extrapolate":
        example_dir = Path("example_data/E27/pos1b")
        noise_simulation_params = PCMNoiseSimulation.simulation_extrapolate()
        divide_by = 64.0  # For some reason the maps are sometimes multiplied by a factor of 64 - check measurment code!
    else:
        raise ValueError(f"Unknown test case: {args.test_case}")

    assert example_dir.is_dir(), f"Example directory not found: {example_dir}"
    assert noise_simulation_params.ground_truth_filepath(
        example_dir
    ).is_file(), f"Ground truth file not found: {noise_simulation_params.ground_truth_filepath(example_dir)}"
    assert noise_simulation_params.noisy_measurement_filepath(
        example_dir
    ).is_file(), f"Noisy measurement file not found: {noise_simulation_params.noisy_measurement_filepath(example_dir)}"

    # Run simulation and get results
    results = run_noise_simulation(
        noise_simulation_params=noise_simulation_params,
        example_dir=example_dir,
        device=get_torch_device(),
        run_subsampling_curves=True,  # Set to False for fast debugging
        n_subsamples=128,
        subsampling_batch_size=8,  # May help speed up subsampling curve computation, especially on GPU
        compute_ssim_curves=True,
        ssim_stride=1,
        divide_by=divide_by,
    )

    # Plot and save results
    ground_truth_filename = noise_simulation_params.ground_truth_filename
    noisy_measurement_filename = noise_simulation_params.noisy_measurement_filename
    noise_params = noise_simulation_params.noise_params
    signal_to_bias_ratio = noise_simulation_params.signal_to_bias_ratio
    rng_seed = noise_simulation_params.rng_seed
    output_dir = Path("/store/LION/ttv22/pcm/noise_simulation_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(
        ground_truth=results["ground_truth"],  # type: ignore
        image_no_noise=results["image_no_noise"],  # type: ignore
        image_noise=results["image_sim_noise"],  # type: ignore
        image_real=results["image_real_noise"],  # type: ignore
        curves=results["curves"],  # type: ignore
        measurement_traces=results["measurement_traces"],  # type: ignore
        output_filepath=output_dir / (f"test_{args.test_case}-seed_{rng_seed}.png"),
        suptitle=(
            "PCM Noise Simulation"
            + "\n"
            + f"Ground truth: `{ground_truth_filename}.txt.map`"
            + ", "
            + f"Noisy measurement: `{noisy_measurement_filename}.txt`"
            + "\n"
            + f"Beam noise AR: {noise_params.beam_ar}, MA: {noise_params.beam_ma}, sigma: {noise_params.beam_sigma:.1e}"
            + "\n"
            + f"Bias noise sigma: {noise_params.bias_sigma:.1e}, gradient sigma: {noise_params.bias_gradient_sigma:.1e}"
            + "\n"
            + f"Dark noise sigma_amps: {noise_params.dark_sigma_amps:.1e}"
            + "\n"
            + f"Signal-to-bias ratio: {signal_to_bias_ratio}, RNG seed: {rng_seed}"
        ),
    )


if __name__ == "__main__":
    main()
