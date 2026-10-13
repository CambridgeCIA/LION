"""Run a single-sample PCM experiment."""

from __future__ import annotations

import csv
import dataclasses
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from LION.classical_algorithms.pdhg_direct import gradient as tv_gradient
from LION.classical_algorithms.spgl1_torch import SPGL1Parameter
from LION.experiments.pcm_experiments import PCMExperimentParameter, make_test_cases
from LION.pcm.data import prepare_data
from LION.pcm.denoiser import build_grayscale_denoiser
from LION.pcm.experiment import resolve_device
from LION.pcm.output_paths import current_datetime_prefix
from LION.pcm.pcm_config import (
    SelfSupervisedPCMMethodParameter,
    SupervisedPCMMethodParameter,
    ZeroShotPCMMethodParameter,
)
from LION.pcm.pcm_sapm import TrainedPCMReconFn
from LION.pcm.reconstruct import (
    GSPnPReconFn,
    PnPADMMReconFn,
    ReconInput,
    ScalarTVParameter,
    ScalarTVReconFn,
    SPGL1ReconFn,
)
from LION.pcm.results import ZERO_FILLED_METHOD_ID
from LION.pcm.run_pcm_single_sample_trial import (
    PCMSingleSampleTrialRun,
    ReconResult,
    Trial,
)
from LION.pcm.simulation import (
    acquisition_from_prepared_measurement,
    sample_and_simulate_pcm_acquisition,
)
from LION.pcm.ZeroShotPCMTrainer import ZeroShotPCMReconFn
from LION.pcm_tools.pcm_sampling import sample_indices
from LION.reconstructors.GSPnP import GSDenoiser, GSPnPParameter
from LION.reconstructors.PnPADMM import PnPADMMParameter
from LION.utils.log import CompactListDumper, to_yaml_friendly
from LION.utils.metrics import compute_metrics
from LION.utils.shape import pad_to_powers_of_two_torch
from scripts.snippets.random_utils import seed_from_parts

IterationCallback = Callable[[int, torch.Tensor], None]
IterationMetricRow = dict[str, float | int]
ExternalIterationCallback = Callable[[int, torch.Tensor, IterationMetricRow], None]
MethodIterationCallbackFactory = Callable[
    [PCMSingleSampleTrialRun, str], ExternalIterationCallback | None
]


def prepare_single_sample_experiment(
    config: PCMExperimentParameter,
    sample_name: str | None = None,
    *,
    save_config: bool = True,
) -> tuple[Path, torch.device]:
    device = resolve_device(config.runtime.device)
    # if config.tv is not None:
    #     config.tv.set_device(device)
    print(f"Using device: {device}")

    if sample_name is None:
        sample_name = f"{current_datetime_prefix()}_{config.id}"
    experiment_log_dir = config.runtime.root_output_dir / f"{sample_name}"
    saves_artifacts = (
        config.log.save_plots
        or config.log.save_recon
        or config.log.save_diff
        or config.log.save_masks
        or config.log.save_zero_filled
    )
    saves_config = save_config and config.log.save_config
    if saves_config or config.log.save_metrics or saves_artifacts:
        experiment_log_dir.mkdir(parents=True, exist_ok=True)

    if saves_config:
        config_path = experiment_log_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                to_yaml_friendly(dataclasses.asdict(config)),
                f,
                Dumper=CompactListDumper,
                sort_keys=False,
            )
        print(f"Saved experiment config to {config_path}")

    return experiment_log_dir, device


def run_single_sample_experiment(
    config: PCMExperimentParameter,
    sample_name: str | None = None,
    *,
    category: str | None = None,
    sample_id: int | None = None,
    return_last_run: bool = False,
    sampling_ready_callback: Callable[[PCMSingleSampleTrialRun], None] | None = None,
    method_iteration_callback_factory: MethodIterationCallbackFactory | None = None,
    noise_seed: int | None = None,
    show_progress_bars: bool = True,
) -> Path | PCMSingleSampleTrialRun:
    """Run the full PCM experiment defined by ``config``.

    Args:
        config: Single-sample experiment configuration.
        sample_name: Optional name for the sample. If not provided, a timestamped name will be generated.
        category: Optional dataset category.
        sample_id: Optional dataset sample ID.
        return_last_run: Whether to return the last in-memory trial run instead of the output directory.
        sampling_ready_callback: Optional callback called after acquisition/sampling preparation
            and before reconstruction.
        method_iteration_callback_factory: Optional factory for method-specific iteration callbacks.
        noise_seed: Optional independent noise-realization seed. When omitted, sampling and noise
            keep the existing shared-RNG behaviour.
        show_progress_bars: Whether to show the generic trial/sampling tqdm bars.

    Returns:
        Output directory for the experiment, or the last trial run when ``return_last_run`` is True.
    """

    trial = Trial()
    trial.config = config
    is_dataset_sample = category is not None
    experiment_log_dir, trial.device = prepare_single_sample_experiment(
        config, sample_name, save_config=not is_dataset_sample
    )
    trial.results_root = (
        config.runtime.root_output_dir if is_dataset_sample else experiment_log_dir
    )
    trial.category = category
    trial.sample_id = sample_id
    trial.target, trial.measurement = prepare_data(config, device=trial.device)
    test_cases = make_test_cases(config)
    plot_helper = config.plot
    artifact_root = (
        experiment_log_dir if is_dataset_sample else experiment_log_dir / "artifacts"
    )
    if config.log.save_plots:
        artifact_root.mkdir(parents=True, exist_ok=True)
        plot_helper.save_image(trial.target, artifact_root / "ground_truth.png")
    if config.data.j_order is not None:
        trial.j_order = config.data.j_order
    else:  # Infer j_order from the target shape if not provided in the config
        trial.j_order = (max(trial.target.shape) - 1).bit_length()

    if config.log.save_plots and plot_helper.zoom is not None:
        trial.target_roi = plot_helper.zoom.extract_roi(trial.target)
        plot_helper.save_image(
            trial.target_roi, artifact_root / "ground_truth_roi.png", roi="preset"
        )

    trial.target = trial.target.unsqueeze(0).unsqueeze(0)

    saves_artifacts = (
        config.log.save_plots
        or config.log.save_recon
        or config.log.save_diff
        or config.log.save_masks
        or config.log.save_zero_filled
    )

    last_run: PCMSingleSampleTrialRun | None = None
    trial_indices = range(
        config.runtime.num_trials_skip + 1, config.runtime.num_trials + 1
    )
    trial_iterator = (
        tqdm(trial_indices, desc="Trials", dynamic_ncols=True)
        if show_progress_bars
        else trial_indices
    )
    for trial_index in trial_iterator:
        print(f"\n=== Trial {trial_index} ===")
        trial_log_dir = artifact_root / f"trial_{trial_index:03d}"
        if saves_artifacts:
            trial_log_dir.mkdir(parents=True, exist_ok=True)
        sampling_iterator = (
            tqdm(test_cases, desc="Running samplings", dynamic_ncols=True)
            if show_progress_bars
            else test_cases
        )
        for sampling_ratio, coarse_j in sampling_iterator:
            run = PCMSingleSampleTrialRun()
            run.trial, run.trial_index = trial, trial_index
            run.results = {}
            last_run = run
            rng_seed = seed_from_parts(
                str(trial_index),
                str(category),
                str(sample_id),
                str(sample_name),
                str(sampling_ratio),
                str(coarse_j),
                str(config.base_seed),
            )
            run.sampling_ratio, run.coarse_j, run.rng_seed = (
                sampling_ratio,
                coarse_j,
                rng_seed,
            )
            run.sampling_case_dir = (
                trial_log_dir / f"s_{run.sampling_ratio:.2f}_cJ_{run.coarse_j}"
            )
            if saves_artifacts:
                run.sampling_case_dir.mkdir(parents=True, exist_ok=True)
            prepare_sampling_run(run, noise_seed=noise_seed)
            if sampling_ready_callback is not None:
                sampling_ready_callback(run)
            run.log_result(ZERO_FILLED_METHOD_ID, run.zf_result)
            log_if_needed(run)
            for method_name in config.methods:
                method_iteration_callback = (
                    method_iteration_callback_factory(run, method_name)
                    if method_iteration_callback_factory is not None
                    else None
                )
                run_method(
                    run,
                    method_name,
                    external_iteration_callback=method_iteration_callback,
                )

    if return_last_run:
        assert last_run is not None
        return last_run
    return experiment_log_dir


def prepare_sampling_run(
    run: PCMSingleSampleTrialRun,
    dtype: torch.dtype = torch.float32,
    *,
    noise_seed: int | None = None,
) -> None:
    config, target, measurement, device = (
        run.trial.config,
        run.trial.target,
        run.trial.measurement,
        run.trial.device,
    )
    assert config.uses_normalized_op

    target_2d = target.squeeze(0).squeeze(0)
    j_order = run.trial.j_order
    rng = np.random.default_rng(seed=run.rng_seed)
    noise_rng = (
        None
        if noise_seed is None
        else np.random.default_rng(
            seed=seed_from_parts(str(run.rng_seed), "noise", str(noise_seed))
        )
    )

    map_true = pad_to_powers_of_two_torch(
        target_2d, dim=(-2, -1), fill_value=config.scaling.fill_value
    )
    target = map_true.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
    run.trial.target = target

    if measurement is None:
        acquisition = sample_and_simulate_pcm_acquisition(
            config.scaling.scale(target),
            j_order=j_order,
            sampling=config.sampling,
            sampling_ratio=run.sampling_ratio,
            coarse_j=run.coarse_j,
            noise=config.noise,
            rng=rng,
            device=device,
            seed=run.rng_seed,
            noise_rng=noise_rng,
        )
    else:
        sampled_indices = sample_indices(
            j_order=j_order,
            sampling_ratio=run.sampling_ratio,
            coarse_j=run.coarse_j,
            randomising_scheme=config.sampling.randomising_scheme,
            rng=rng,
        )
        sampled_index_tensor = torch.as_tensor(
            sampled_indices, dtype=torch.long, device=measurement.device
        )
        prepared_measurement = (
            measurement.index_select(-1, sampled_index_tensor).unsqueeze(0).unsqueeze(0)
        )
        acquisition = acquisition_from_prepared_measurement(
            prepared_measurement,
            j_order=j_order,
            sampled_indices=sampled_indices,
            device=device,
            seed=run.rng_seed,
        )

    start_zero_filled_time = time.time()
    zero_filled_recon_tensor = acquisition.op.pseudo_inv(acquisition.measurement)
    end_zero_filled_time = time.time()

    zero_filled_recon = config.scaling.rescale(zero_filled_recon_tensor)
    zf_metrics = compute_metrics(
        recon=zero_filled_recon, target=target, data_range=run.trial.data_range
    )
    run.zf_result = ReconResult(
        recon=zero_filled_recon,
        metrics=zf_metrics,
        duration=end_zero_filled_time - start_zero_filled_time,
    )

    i0 = None
    if acquisition.i0 is not None:
        i0 = float(acquisition.i0.reshape(-1)[0].item())

    run.recon_input = ReconInput(
        op=acquisition.op, y=acquisition.measurement, x0=zero_filled_recon_tensor, i0=i0
    )
    run.sampled_indices = acquisition.sampled_indices


def log_if_needed(run: PCMSingleSampleTrialRun) -> None:
    trial, config, plotter = run.trial, run.trial.config, run.trial.config.plot

    sampling_percentage = run.sampling_ratio * 100.0

    if config.log.save_masks:
        masks_dir = run.sampling_case_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        n = config.data.n
        mask_of_masks_np = np.zeros(n * n, dtype=bool)
        mask_of_masks_np[run.sampled_indices] = True
        mask_filename = (
            f"sampling_{sampling_percentage}_percent_coarse_J={run.coarse_j}_"
            f"{config.sampling.randomising_scheme}_random"
        )
        np.save(masks_dir / f"{mask_filename}.npy", mask_of_masks_np.reshape(n, n))

    if config.log.save_zero_filled:
        zf_dir = run.sampling_case_dir / "zero_filled"
        zf_dir.mkdir(parents=True, exist_ok=True)
        zf_filename = (
            f"sampling_{sampling_percentage}_percent_" f"coarse_J={run.coarse_j}"
        )
        np.save(
            zf_dir / f"{zf_filename}.npy", run.zf_result.recon.squeeze().cpu().numpy()
        )
    if config.log.save_plots:
        plotter.save_image(
            run.zf_result.recon, run.sampling_case_dir / "zero_filled.png"
        )
    if config.log.save_plots and plotter.zoom is not None:
        assert trial.target_roi is not None
        zf_roi = plotter.zoom.extract_roi(run.zf_result.recon.squeeze())
        zf_roi_metrics = compute_metrics(
            recon=zf_roi, target=trial.target_roi, data_range=trial.data_range
        )
        run.zf_roi_result = ReconResult(
            recon=zf_roi, metrics=zf_roi_metrics, duration=run.zf_result.duration
        )
        plotter.save_image(
            zf_roi, run.sampling_case_dir / "zero_filled_roi.png", roi="preset"
        )


def _synchronize_for_iteration_timing(device: torch.device | None) -> None:
    """Synchronize asynchronous device work before reading the wall clock."""
    if device is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def save_pnp_iteration_metrics(
    run: PCMSingleSampleTrialRun,
    method_id: str,
    iteration_metrics: list[IterationMetricRow],
) -> None:
    """Save reconstruction metrics and cumulative reconstruction time over iterations."""
    method_dir = run.method_dir(method_id)
    method_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["iteration", "time", "psnr", "ssim", "haarpsi", "mse"]
    is_scalar_tv = "objective" in iteration_metrics[0]
    if is_scalar_tv:
        fieldnames.extend(["data_fidelity", "tv", "objective"])
    if run.config.log.save_metrics:
        csv_path = method_dir / "iteration_metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(iteration_metrics)

    if run.config.log.save_plots:
        metric_specs = [
            ("psnr", "PSNR (dB)"),
            ("ssim", "SSIM"),
            ("haarpsi", "HaarPSI"),
            ("mse", "MSE"),
        ]
        for x_key, x_label in (
            ("iteration", "PnP iteration"),
            ("time", "Reconstruction time (s)"),
        ):
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            x_values = [row[x_key] for row in iteration_metrics]
            for ax, (metric_key, metric_label) in zip(
                axes.flat, metric_specs, strict=True
            ):
                ax.plot(x_values, [row[metric_key] for row in iteration_metrics])
                ax.set_xlabel(x_label)
                ax.set_ylabel(metric_label)
                ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(
                method_dir / f"iteration_metrics_vs_{x_key}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        if is_scalar_tv:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            for ax, (metric_key, title) in zip(
                axes,
                (
                    ("data_fidelity", "Data fidelity"),
                    ("tv", "Anisotropic TV"),
                    ("objective", "Objective"),
                ),
                strict=True,
            ):
                ax.plot(
                    [row["iteration"] for row in iteration_metrics],
                    [row[metric_key] for row in iteration_metrics],
                )
                ax.set_xlabel("PDHG iteration")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                method_dir / "tv_objective_vs_iteration.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)


def run_method(
    run: PCMSingleSampleTrialRun,
    method_id: str,
    *,
    external_iteration_callback: ExternalIterationCallback | None = None,
) -> None:
    config, device = run.trial.config, run.trial.device
    method_config, scaling_config = config.methods[method_id], config.scaling
    iteration_metrics: list[IterationMetricRow] | None = None
    iteration_elapsed = 0.0
    iteration_timer_start: float | None = None

    supports_iteration_callback = isinstance(
        method_config, (PnPADMMParameter, ScalarTVParameter)
    ) or (isinstance(method_config, GSPnPParameter) and method_config.uses_deepinv)
    records_iteration_metrics = (
        isinstance(method_config, (PnPADMMParameter, ScalarTVParameter))
        and (config.log.save_metrics or config.log.save_plots)
    ) or external_iteration_callback is not None
    if supports_iteration_callback and records_iteration_metrics:
        iteration_metrics = []

    if supports_iteration_callback and (
        iteration_metrics is not None or external_iteration_callback is not None
    ):

        def iteration_callback(iteration: int, x: torch.Tensor) -> None:
            nonlocal iteration_elapsed, iteration_timer_start

            _synchronize_for_iteration_timing(device)
            now = time.perf_counter()
            if iteration_timer_start is not None:
                iteration_elapsed += now - iteration_timer_start

            recon = scaling_config.rescale(x.detach())
            metric_values = compute_metrics(
                recon=recon, target=run.trial.target, data_range=run.data_range
            )
            row: IterationMetricRow = {
                "iteration": iteration,
                "time": iteration_elapsed,
                "psnr": metric_values["psnr"],
                "ssim": metric_values["ssim"],
                "haarpsi": metric_values["haarpsi"],
                "mse": metric_values["mse"],
            }
            if isinstance(method_config, ScalarTVParameter):
                with torch.no_grad():
                    data_fidelity = (
                        0.5
                        * (run.recon_input.op(x) - run.recon_input.y)
                        .square()
                        .sum()
                        .item()
                    )
                    tv_value = tv_gradient(x).abs().sum().item()
                row.update(
                    data_fidelity=data_fidelity,
                    tv=tv_value,
                    objective=data_fidelity + method_config.lambda_reg * tv_value,
                )
            if iteration_metrics is not None:
                iteration_metrics.append(row)
            if external_iteration_callback is not None:
                external_iteration_callback(iteration, recon, row)

            _synchronize_for_iteration_timing(device)
            iteration_timer_start = time.perf_counter()

    else:
        iteration_callback = None

    external_early_stop_callback = (
        getattr(external_iteration_callback, "early_stop", None)
        if external_iteration_callback is not None
        else None
    )

    def early_stop_callback(iteration: int, x: torch.Tensor) -> None:
        if external_early_stop_callback is None:
            return
        recon = scaling_config.rescale(x.detach())
        external_early_stop_callback(iteration, recon)

    external_backtracking_stop_callback = (
        getattr(external_iteration_callback, "backtracking_stop", None)
        if external_iteration_callback is not None
        else None
    )

    def backtracking_stop_callback(
        last_accepted_iteration: int,
        x: torch.Tensor,
        attempted_iteration: int,
        failed_attempts: int,
        stepsize: float,
    ) -> None:
        if external_backtracking_stop_callback is None:
            return
        recon = scaling_config.rescale(x.detach())
        external_backtracking_stop_callback(
            last_accepted_iteration,
            recon,
            attempted_iteration,
            failed_attempts,
            stepsize,
        )

    if external_iteration_callback is not None and not supports_iteration_callback:
        raise TypeError(
            f"Iteration callbacks are not supported for method {method_id!r}."
        )

    if isinstance(method_config, GSPnPParameter):
        trial_description = (
            f"GS-PnP {method_config.denoiser.name}\n"
            f"{method_config.iters} iters, sigma={method_config.sigma}\n"
            f"inverse_sign={config.plot.color.inverse_sign}"
        )
        denoiser = build_grayscale_denoiser(method_config.denoiser, device)
        assert isinstance(
            denoiser, GSDenoiser
        ), f"Expected GSDenoiser, got {type(denoiser)}"
        recon_fn = GSPnPReconFn(
            method_config,
            denoiser,
            device=device,
            iteration_callback=iteration_callback,
            early_stop_callback=early_stop_callback
            if external_early_stop_callback is not None
            else None,
            backtracking_stop_callback=(
                backtracking_stop_callback
                if external_backtracking_stop_callback is not None
                else None
            ),
        )

    elif isinstance(method_config, PnPADMMParameter):
        trial_description = (
            f"PnP-ADMM {method_config.denoiser.name}\n"
            f"{method_config.iters} iters, sigma={method_config.sigma}\n"
            f"inverse_sign={config.plot.color.inverse_sign}"
        )
        denoiser = build_grayscale_denoiser(method_config.denoiser, device)

        recon_fn = PnPADMMReconFn(
            method_config,
            denoiser,
            device=device,
            iteration_callback=iteration_callback,
        )

    # if config.tv is not None:
    #     tv_recon_fn = make_tv_reconstructor(config.tv)
    #     method_name = f"tv_lambda={float(config.tv.lambda_v):.3f}"
    #     make_csv(method_name=method_name, log_dir=trial_log_dir)
    #     try:
    #         run_pcm_single_sample(
    #             config=config,
    #             recon_description=f"TV (lambda={float(config.tv.lambda_v)})",
    #             recon_fn=tv_recon_fn,
    #             ground_truth_2d=ground_truth_image,
    #             method_name=method_name,
    #             sampling_ratio=sampling_ratio,
    #             coarse_j=coarse_j,
    #             measurement_vector=measurement_vector,
    #             log_dir=trial_log_dir,
    #             device=device,
    #             seed=i_seed,
    #             rng=np.random.default_rng(seed=i_seed),
    #             uses_normalized_op=config.tv.uses_normalized_op,
    #         )
    #     except Exception as e:
    #         print(f"Error in trial {i_seed}, sampling_ratio={sampling_ratio}, coarse_j={coarse_j}: {e}")
    #         if config.runtime.skip_exceptions:
    #             continue
    #         else:
    #             raise e

    elif isinstance(method_config, ScalarTVParameter):
        trial_description = f"Scalar anisotropic TV (lambda={method_config.lambda_reg:g}, iters={method_config.iters})"
        recon_fn = ScalarTVReconFn(method_config, iteration_callback=iteration_callback)

    elif isinstance(method_config, SPGL1Parameter):
        trial_description = f"SPGL1\n" f"inverse_sign={config.plot.color.inverse_sign})"
        recon_fn = SPGL1ReconFn(method_config, scaling_config, device=device)

    elif isinstance(
        method_config, (SupervisedPCMMethodParameter, SelfSupervisedPCMMethodParameter)
    ):
        training_type = (
            "Self-supervised"
            if isinstance(method_config, SelfSupervisedPCMMethodParameter)
            else "Supervised"
        )
        trial_description = f"{training_type} PCM ({method_id})"
        recon_fn = TrainedPCMReconFn(
            parameter=method_config,
            sampling_ratio=run.sampling_ratio,
            coarse_j=run.coarse_j,
            device=device,
        )

    elif isinstance(method_config, ZeroShotPCMMethodParameter):
        assert run.sampled_indices is not None
        assert run.rng_seed is not None
        trial_description = f"Zero-shot PCM ({method_id})"
        recon_fn = ZeroShotPCMReconFn(
            parameter=method_config,
            sampled_indices=run.sampled_indices,
            sampling_ratio=run.sampling_ratio,
            coarse_j=run.coarse_j,
            rng_seed=run.rng_seed,
            j_order=run.trial.j_order,
            device=device,
            uses_normalized_op=config.uses_normalized_op,
            target_image=scaling_config.scale(run.trial.target).squeeze(0).squeeze(0),
            category=run.trial.category or "",
            sample_id=run.trial.sample_id if run.trial.sample_id is not None else -1,
            output_dir=run.method_dir(method_id),
        )

    else:
        raise ValueError(
            f"Unsupported method configuration type: {type(method_config)}"
        )

    # print()
    # print(f"Running method {method_id}")
    if iteration_callback is not None:
        _synchronize_for_iteration_timing(device)
        iteration_timer_start = time.perf_counter()
    run.run_pcm_single_sample_trial(
        recon_fn, trial_description=trial_description, method_name=method_id
    )

    if iteration_metrics:
        save_pnp_iteration_metrics(run, method_id, iteration_metrics)
    # print()
