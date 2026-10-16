"""Run simulation"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from LION.data_loaders.pcm.data_utils import (
    read_pcm_current_map_from_txt,
    read_pcm_measurement_data_from_txt,
)
from LION.experiments.pcm_experiments import PCMNoiseSimulation
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm_tools.build_measurement import build_real_measurement_vector
from LION.pcm_tools.measure import measure_split_hadamard_transform
from LION.pcm_tools.noise_generate import add_noise, generate_noise
from LION.pcm_tools.noise_reconstruct import PCMReconstructor
from LION.pcm_tools.noise_subsampling_curve import SubsamplingCurveBuilder
from LION.pcm_tools.noise_utils import generate_order_sorting_matrix
from LION.utils.shape import pad_to_powers_of_two_torch, torch_flatten_fortran_2d


def run_noise_simulation(
    noise_simulation_params: PCMNoiseSimulation,
    example_dir: Path,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    run_subsampling_curves: bool = True,
    n_subsamples: int = 128,
    subsampling_batch_size: int = 8,
    compute_ssim_curves: bool = True,
    ssim_stride: int = 1,
    *,
    divide_by: float,
) -> dict[str, object]:
    """Run the synthetic-noise example.

    Args:
        noise_simulation_params: Parameters for the noise simulation.
        device: Torch device to use (e.g., "cuda", "mps", or "cpu"). If None, automatically select an available device.
        dtype: Torch floating-point dtype to use for computations.
        run_subsampling_curves: Whether to compute subsampling curves.
        n_subsamples: Number of subsampling levels to compute.
        subsampling_batch_size: Batch size for subsampling reconstructions.
        compute_ssim_curves: Whether to compute SSIM values for the subsampling curves.
        ssim_stride: Compute SSIM every `ssim_stride` sampling levels.

    Returns:
    -------
        dict[str, object]: Dictionary containing images, metrics, curves, traces, and metadata.
    """
    rng = np.random.default_rng(noise_simulation_params.rng_seed)

    ground_truth = read_pcm_current_map_from_txt(
        noise_simulation_params.ground_truth_filepath(example_dir),
        device=device,
        dtype=dtype,
    )
    pixel_min = ground_truth.min().item()
    pixel_max = ground_truth.max().item()
    print(f"Ground truth pixel range: [{pixel_min}, {pixel_max}]")

    # For some reason the maps are sometimes multiplied by a factor of 64 -
    # need to check the measurment code to find out why!
    ground_truth = ground_truth / divide_by

    n_rows, n_cols = ground_truth.shape

    print(f"n_rows={n_rows}, n_cols={n_cols}", flush=True)

    true_map = pad_to_powers_of_two_torch(ground_truth)
    n_rows_padded, n_cols_padded = true_map.shape
    n = n_rows_padded * n_cols_padded

    print(
        f"n_row_padded={n_rows_padded}, n_col_padded={n_cols_padded}, n_total={n}",
        flush=True,
    )

    x_true = torch_flatten_fortran_2d(true_map)

    print("Generating order sorting matrix...", flush=True)
    iy_np = generate_order_sorting_matrix(
        o_col=int(np.ceil(np.log2(n_cols))), o_row=int(np.ceil(np.log2(n_rows)))
    )
    iy = torch.as_tensor(iy_np, dtype=torch.long, device=device)

    reconstructor = PCMReconstructor(
        iy=iy,
        n_row=n_rows_padded,
        n_col=n_cols_padded,
        n_rows_orig=n_rows,
        n_cols_orig=n_cols,
    )
    subsampling_curve_builder = SubsamplingCurveBuilder(
        iy=iy,
        ground_truth=ground_truth,
        pixel_min=pixel_min,
        n_row=n_rows_padded,
        n_col=n_cols_padded,
        n_subsamples=n_subsamples,
        batch_size=subsampling_batch_size,
        compute_ssim=compute_ssim_curves,
        ssim_stride=ssim_stride,
    )
    skip_subsampling_curves = not run_subsampling_curves

    with torch.no_grad():
        print("Simulating ideal split-Hadamard measurement...", flush=True)
        y_plus_no_noise, y_minus_no_noise = measure_split_hadamard_transform(x_true)

        # Sort the measurements into ordered-pattern order
        y_plus_no_noise = y_plus_no_noise[iy]
        y_minus_no_noise = y_minus_no_noise[iy]
        y_no_noise = y_plus_no_noise - y_minus_no_noise

        print("Reconstructing noiseless image...", flush=True)
        # image_no_noise = reconstructor.reconstruct_from_sorted_measurements(y_no_noise, transpose_after_reshape=False)
        pcm_op_full = PhotocurrentMapOp(
            j_order=int(np.ceil(np.log2(n_cols))), device=device
        )
        image_no_noise = pcm_op_full.pseudo_inv(y_no_noise)
        image_no_noise = image_no_noise.T[:n_rows, :n_cols]
        print(
            f"Noiseless image pixel range: [{image_no_noise.min().item()}, {image_no_noise.max().item()}]",
            flush=True,
        )

        print("Computing noiseless subsampling curve...", flush=True)
        subsampling_curve_no_noise = subsampling_curve_builder.build_curve(
            y_no_noise, transpose_after_reshape=False, skip=skip_subsampling_curves
        )

        print("Generating synthetic noise...", flush=True)

        total_true_photocurrent = ground_truth.sum()

        i0 = float(
            total_true_photocurrent / noise_simulation_params.signal_to_bias_ratio
        )

        e_abs_dark, e_abs_bias_np, e_rel_beam_np = generate_noise(
            num_measurements=2 * n,
            i_bias=i0,
            noise_params=noise_simulation_params.noise_params,
            rng=rng,
        )
        e_abs_np = e_abs_bias_np + e_abs_dark
        y_plus_sim_noise, y_minus_sim_noise, sim_noisy_trace = add_noise(
            y_clean_plus=y_plus_no_noise,
            y_clean_minus=y_minus_no_noise,
            e_abs_trace=torch.as_tensor(e_abs_np, dtype=dtype, device=device),
            e_rel=torch.as_tensor(e_rel_beam_np, dtype=dtype, device=device),
            rng=rng,
        )

        y_sim_noise = y_plus_sim_noise - y_minus_sim_noise

        print("Reconstructing simulated noisy image...", flush=True)
        image_simulated_noise = reconstructor.reconstruct_from_sorted_measurements(
            y_sim_noise, transpose_after_reshape=False
        )
        print(
            f"Simulated noisy image pixel range: [{image_simulated_noise.min().item()}, {image_simulated_noise.max().item()}]",
            flush=True,
        )

        print("Computing simulated-noise subsampling curve...", flush=True)
        subsampling_curve_sim_noise = subsampling_curve_builder.build_curve(
            y_sim_noise, transpose_after_reshape=False, skip=skip_subsampling_curves
        )

        measurements_df = read_pcm_measurement_data_from_txt(
            noise_simulation_params.noisy_measurement_filepath(example_dir)
        )
        measurements_np = measurements_df.to_numpy(dtype=np.float64)

        y_real_noise_np = build_real_measurement_vector(measurements_np, n=n)
        y_real_noise = torch.as_tensor(y_real_noise_np, dtype=dtype, device=device)

        print("Reconstructing real noisy image...", flush=True)
        image_real_noise = reconstructor.reconstruct_from_sorted_measurements(
            y_real_noise,
            transpose_after_reshape=True,  # TODO: Why transpose?
        )
        print(
            f"Real noisy image pixel range: [{image_real_noise.min().item()}, {image_real_noise.max().item()}]",
            flush=True,
        )

        print("Computing real-noise subsampling curve...", flush=True)
        subsampling_curve_real_noise = subsampling_curve_builder.build_curve(
            y_real_noise, transpose_after_reshape=True, skip=skip_subsampling_curves
        )

    return {
        "ground_truth": ground_truth.cpu(),
        "image_no_noise": image_no_noise.cpu(),
        "image_sim_noise": image_simulated_noise.cpu(),
        "image_real_noise": image_real_noise.cpu(),
        "curves": {
            "no_noise": subsampling_curve_no_noise,
            "sim_noise": subsampling_curve_sim_noise,
            "real_noise": subsampling_curve_real_noise,
        },
        "measurement_traces": {
            "simulated_noisy_interleaved": sim_noisy_trace.cpu(),
            "real_measurements": measurements_np[:, 1],
        },
    }
