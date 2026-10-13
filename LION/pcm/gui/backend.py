"""Backend adapter used by the PCM GUI."""

from __future__ import annotations

import copy
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from LION.experiments.pcm_presets.presets import PRESETS
from LION.pcm.results import ZERO_FILLED_METHOD_ID

if TYPE_CHECKING:
    from LION.pcm.run_pcm_single_sample_trial import PCMSingleSampleTrialRun


@dataclass(frozen=True, kw_only=True)
class GuiRunRequest:
    """Values selected in the GUI for one PCM run."""

    preset: str
    algorithm: str
    device: str
    sampling_ratio: float
    coarse_j: int
    uses_noise: bool
    output_dir: Path
    save_config: bool
    save_metrics: bool
    save_plots: bool
    save_recon: bool
    save_diff: bool
    save_masks: bool
    save_zero_filled: bool


@dataclass(frozen=True, kw_only=True)
class GuiAcquisitionPreview:
    """In-memory data needed to animate one PCM acquisition."""

    target: np.ndarray
    sampled_indices: np.ndarray
    j_order: int
    measurements: np.ndarray
    sampling_ratio: float
    inverse_sign: bool
    plot_helper: object


@dataclass(frozen=True, kw_only=True)
class GuiRunResult:
    """In-memory result returned to the GUI."""

    experiment_dir: Path
    algorithm: str
    run: PCMSingleSampleTrialRun
    saved_outputs: bool


def available_devices() -> list[str]:
    """Return compute devices without initializing a PyTorch accelerator backend."""
    devices = ["auto", "cpu"]

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        try:
            result = subprocess.run(
                [nvidia_smi, "-L"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                devices.append("cuda")
        except (OSError, subprocess.SubprocessError):
            pass

    if sys.platform == "darwin" and platform.machine() == "arm64":
        devices.append("mps")

    return devices


def make_acquisition_preview(run: PCMSingleSampleTrialRun) -> GuiAcquisitionPreview:
    """Copy the data needed by the GUI acquisition animation to CPU memory."""
    return GuiAcquisitionPreview(
        target=run.trial.target.detach().squeeze().cpu().numpy().copy(),
        sampled_indices=np.asarray(run.sampled_indices, dtype=np.int64).copy(),
        j_order=int(run.trial.j_order),
        measurements=run.recon_input.y.detach().squeeze().cpu().numpy().copy(),
        sampling_ratio=float(run.sampling_ratio),
        inverse_sign=run.config.plot.color.inverse_sign,
        plot_helper=copy.deepcopy(run.config.plot),
    )


def run_gui_experiment(
    request: GuiRunRequest,
    *,
    acquisition_ready_callback: Callable[[GuiAcquisitionPreview], None] | None = None,
) -> GuiRunResult:
    """Run one existing PCM experiment from GUI-selected values."""
    from LION.pcm.run_pcm_single_sample_experiment import run_single_sample_experiment
    from LION.pcm.run_pcm_single_sample_trial import PCMSingleSampleTrialRun

    config = copy.deepcopy(PRESETS[request.preset])

    if (
        request.algorithm != ZERO_FILLED_METHOD_ID
        and request.algorithm not in config.methods
    ):
        raise ValueError(
            f"Algorithm {request.algorithm!r} is not available in preset {request.preset!r}."
        )

    config.runtime.device = request.device
    config.runtime.root_output_dir = request.output_dir
    config.runtime.num_trials = 1
    config.runtime.num_trials_skip = 0

    config.sampling.sampling_ratios = [request.sampling_ratio]
    config.sampling.coarse_j_values = [request.coarse_j]

    if request.algorithm == ZERO_FILLED_METHOD_ID:
        config.methods = {}
    else:
        config.methods = {request.algorithm: config.methods[request.algorithm]}

    if not request.uses_noise:
        config.noise = None

    config.log.save_config = request.save_config
    config.log.save_metrics = request.save_metrics
    config.log.save_plots = request.save_plots
    config.log.save_recon = request.save_recon
    config.log.save_diff = request.save_diff
    config.log.save_masks = request.save_masks
    config.log.save_zero_filled = request.save_zero_filled

    def sampling_ready(run: PCMSingleSampleTrialRun) -> None:
        if acquisition_ready_callback is not None:
            acquisition_ready_callback(make_acquisition_preview(run))

    run = run_single_sample_experiment(
        config,
        return_last_run=True,
        sampling_ready_callback=sampling_ready,
    )
    assert isinstance(run, PCMSingleSampleTrialRun)

    saved_outputs = any(
        (
            request.save_config,
            request.save_metrics,
            request.save_plots,
            request.save_recon,
            request.save_diff,
            request.save_masks,
            request.save_zero_filled,
        )
    )

    return GuiRunResult(
        experiment_dir=run.trial.results_root,
        algorithm=request.algorithm,
        run=run,
        saved_outputs=saved_outputs,
    )
