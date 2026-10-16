"""Run one PCM reconstruction case and save its outputs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from LION.data_loaders.pcm.data_loader import GrayscaleImage2D, Measurement1D
from LION.experiments.pcm_experiments import PCMExperimentParameter
from LION.pcm.pcm_visualize import absolute_error
from LION.pcm.reconstruct import AdaptiveMapReconOutput, ReconFn, ReconInput
from LION.pcm.results import PCMResultRow, append_result
from LION.utils.metrics import compute_metrics


@dataclass(kw_only=True)
class ReconResult:
    recon: torch.Tensor
    metrics: dict[str, float] | None = None
    duration: float | None = None


@dataclass(kw_only=True)
class AdaptiveMapReconResult(ReconResult):
    lambda_maps: torch.Tensor


@dataclass(init=False)
class Trial:
    config: PCMExperimentParameter
    results_root: Path
    category: str | None = None
    sample_id: int | None = None
    target: GrayscaleImage2D
    target_roi: GrayscaleImage2D | None = None
    measurement: Measurement1D | None = None
    j_order: int
    device: torch.device

    @property
    def data_range(self) -> float:
        return (self.target.max() - self.target.min()).item()


@dataclass(init=False)
class PCMSingleSampleTrialRun:
    trial: Trial
    trial_index: int
    sampling_case_dir: Path
    sampling_ratio: float
    coarse_j: int
    sampled_indices: np.ndarray | None = None
    zf_result: ReconResult
    zf_roi_result: ReconResult | None = None
    recon_input: ReconInput
    rng_seed: int | None = None
    results: dict[str, ReconResult]

    @property
    def config(self) -> PCMExperimentParameter:
        return self.trial.config

    @property
    def n(self) -> int:
        return 1 << self.trial.j_order

    @property
    def data_range(self) -> float:
        return (self.trial.target.max() - self.trial.target.min()).item()

    @property
    def sampling_percentage(self) -> float:
        return self.sampling_ratio * 100.0

    @property
    def in_order_measurements_percentage(self) -> float:
        return (1 << (2 * self.coarse_j)) / (self.trial.target.numel()) * 100.0

    def run_pcm_single_sample_trial(
        self, recon_fn: ReconFn, *, trial_description: str, method_name: str
    ) -> None:
        """Run one PCM reconstruction case and save its outputs.

        Args:
            trial_description: Short description of the trial to put in the figure title.
            recon_fn: Reconstruction function.
            method_name: Method-specific output directory name.
        """
        start_recon_time = time.time()
        output = recon_fn(self.recon_input)
        recon = self.config.scaling.rescale(output.reconstruction.detach())
        end_recon_time = time.time()
        recon_metrics = compute_metrics(
            recon=recon, target=self.trial.target, data_range=self.data_range
        )
        if isinstance(output, AdaptiveMapReconOutput):
            recon_result = AdaptiveMapReconResult(
                recon=recon,
                duration=end_recon_time - start_recon_time,
                metrics=recon_metrics,
                lambda_maps=output.lambda_maps,
            )
        else:
            recon_result = ReconResult(
                recon=recon,
                duration=end_recon_time - start_recon_time,
                metrics=recon_metrics,
            )
        # print(
        #     f"Reconstruction time: {recon_result.duration:.2f} seconds. Shape: {recon_result.recon.shape}. "
        #     f"Metrics: {recon_result.metrics}"
        # )

        self.log_result(method_name, recon_result)
        self.log_and_save(
            trial_description=trial_description,
            method_name=method_name,
            recon_result=recon_result,
        )

    def method_dir(self, method_name: str) -> Path:
        return Path(self.sampling_case_dir) / method_name

    def log_result(self, method_name: str, result: ReconResult) -> None:
        """Store one result and optionally append it to the method CSV."""

        self.results[method_name] = result

        if not self.config.log.save_metrics:
            return

        assert result.metrics is not None
        assert result.duration is not None
        assert self.rng_seed is not None

        append_result(
            root=self.trial.results_root,
            method_id=method_name,
            category=self.trial.category,
            result=PCMResultRow(
                sample_id=self.trial.sample_id,
                trial_num=self.trial_index,
                sampling_rate=self.sampling_ratio,
                coarse_j=self.coarse_j,
                rng_seed=self.rng_seed,
                mse=result.metrics["mse"],
                psnr=result.metrics["psnr"],
                ssim=result.metrics["ssim"],
                haarpsi=result.metrics["haarpsi"],
                pearson_corr=result.metrics["pearson_corr"],
                duration=result.duration,
            ),
        )

    def log_and_save(
        self, *, trial_description: str, method_name: str, recon_result: ReconResult
    ) -> None:
        """Save optional outputs for one PCM reconstruction case."""
        method_dir = self.method_dir(method_name)

        if self.config.log.save_recon:
            method_dir.mkdir(parents=True, exist_ok=True)
            recons_dir = method_dir / "recons"
            recons_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                recons_dir / "recon.npy", recon_result.recon.squeeze().cpu().numpy()
            )
            if isinstance(recon_result, AdaptiveMapReconResult):
                np.save(
                    recons_dir / "lambda_maps.npy",
                    recon_result.lambda_maps.detach().cpu().numpy(),
                )

        if not (self.config.log.save_plots or self.config.log.save_diff):
            return

        method_dir.mkdir(parents=True, exist_ok=True)
        # images_dir = method_dir / "images"
        # images_dir.mkdir(parents=True, exist_ok=True)
        images_dir = method_dir
        plot_helper = self.config.plot

        if self.config.log.save_diff:
            diff = absolute_error(recon_result.recon, self.trial.target)
            plot_helper.save_image(
                diff,
                images_dir / "absolute_error.png",
                cmap_name="viridis",
                clim=(0.0, float(diff.max().item())),
                inverse_sign=False,
            )

        if not self.config.log.save_plots:
            return

        if isinstance(recon_result, AdaptiveMapReconResult):
            lambda_maps = recon_result.lambda_maps.detach().cpu()
            if lambda_maps.ndim == 4:
                lambda_maps = lambda_maps.squeeze(0)
            for map_index, lambda_map in enumerate(lambda_maps):
                map_filename = (
                    "lambda_map.png"
                    if lambda_maps.shape[0] == 1
                    else f"lambda_map_{map_index}.png"
                )
                plot_helper.save_image(
                    lambda_map,
                    images_dir / map_filename,
                    cmap_name="magma",
                    clim=(
                        float(lambda_map.min().item()),
                        float(lambda_map.max().item()),
                    ),
                    inverse_sign=False,
                )

        plot_helper.save_image(recon_result.recon, images_dir / "recon.png")
        plot_helper.show_images_with_inset(
            [self.trial.target, self.zf_result.recon, recon_result.recon],
            fig_filepath=images_dir / "comparison.png",
            titles=[
                "Target",
                self.subplot_title("Inverse WHT (Zero-filled)", self.zf_result),
                self.subplot_title(trial_description, recon_result),
            ],
            suptitle=self.suptitle(
                f"PCM Reconstructions, J={self.trial.j_order} ({self.n}x{self.n} image)"
            ),
            saves_fig=True,
        )

        if plot_helper.zoom is not None:
            recon_roi = plot_helper.zoom.extract_roi(recon_result.recon.squeeze())
            assert self.trial.target_roi is not None
            recon_roi_metrics = compute_metrics(
                recon=recon_roi,
                target=self.trial.target_roi,
                data_range=self.data_range,
            )
            recon_roi_result = ReconResult(
                recon=recon_roi,
                metrics=recon_roi_metrics,
                duration=recon_result.duration,
            )

            plot_helper.save_image(
                recon_roi, images_dir / "recon_roi.png", roi="preset"
            )
            plot_helper.zoom.adds_insets = False
            x, y, width, height = plot_helper.zoom.roi
            assert self.zf_roi_result is not None
            plot_helper.show_images_with_inset(
                [self.trial.target_roi, self.zf_roi_result.recon, recon_roi],
                fig_filepath=images_dir / "comparison_roi.png",
                titles=[
                    "Target (ROI)",
                    self.subplot_title("Inverse WHT (Zero-filled)", self.zf_roi_result),
                    self.subplot_title(trial_description, recon_roi_result),
                ],
                suptitle=self.suptitle(
                    f"ROI (x={x}, y={y}, width={width}, height={height})"
                ),
                saves_fig=True,
            )

    def subplot_title(self, description: str, result: ReconResult) -> str:
        assert result.metrics is not None
        return (
            f"{description}\n"
            f"PSNR: {result.metrics['psnr']:.2f} dB, SSIM: {result.metrics['ssim']:.4f}, "
            f"HAARPsi: {result.metrics['haarpsi']:.4f}\n"
            f"MSE: {result.metrics['mse']:.3e}\n"
            f"Time: {result.duration:.2f} seconds"
        )

    def suptitle(self, description: str) -> str:
        return (
            f"{description}\n"
            f"PCM Noise: {self.noise_description}\n"
            f"Sampling {self.sampling_percentage:.2f}%, keep {self.coarse_j} coarse levels "
            f"({self.in_order_measurements_percentage}%), "
            f"the rest: {self.config.sampling.randomising_scheme} random\n"
            f"Base seed: {self.config.base_seed}, trial {self.trial_index}"
        )

    @property
    def noise_description(self) -> str:
        """Make short description of the noise configuration for figure title."""
        if self.config.noise is None:
            return "none\n"
        noise_params = self.config.noise.noise_params
        return (
            f"AR={noise_params.beam_ar}, MA={noise_params.beam_ma}, "
            f"beam_sigma={noise_params.beam_sigma},\n"
            f"bias_sigma={noise_params.bias_sigma}, bias_gradient_sigma={noise_params.bias_gradient_sigma}, "
            f"dark_sigma={noise_params.dark_sigma}"
        )
