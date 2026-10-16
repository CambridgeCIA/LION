"""Run and plot PnP iteration comparisons across methods and noise scenarios."""

from __future__ import annotations

import copy
import csv
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from tqdm import tqdm

from LION.experiments.pcm_experiments import PCMExperimentParameter, make_test_cases
from LION.pcm.config_cli import parse_pcm_experiment_config_cli
from LION.pcm.output_paths import name_with_datetime_prefix
from LION.pcm.pnp_analysis_io import save_analysis_definition
from LION.pcm.run_pcm_single_sample_experiment import (
    MethodIterationCallbackFactory,
    run_single_sample_experiment,
)
from LION.pcm.run_pcm_single_sample_trial import PCMSingleSampleTrialRun
from LION.reconstructors.GSPnP import GSPnPParameter
from LION.reconstructors.PnPADMM import PnPADMMParameter
from LION.utils.plot_helper import PlotHelper

PlotMode = Literal["individual", "methods-per-scenario", "scenarios-per-method"]
SnapshotPosition = Literal["top", "bottom", "left", "right"]
MetricName = Literal["psnr", "ssim", "haarpsi", "mse"]
XAxis = Literal["iteration", "time"]
TimeOrigin = Literal["reconstruction-start", "first-point"]
LegendPosition = Literal["right", "bottom", "top", "none"]

METRIC_LABELS: dict[MetricName, str] = {
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "haarpsi": "HaarPSI",
    "mse": "MSE",
}


@dataclass(kw_only=True)
class PnPIterationPlotParameter:
    """Plot options for PnP iteration comparisons."""

    snapshot_iterations: list[int] = field(default_factory=list)
    snapshot_every: int = 0
    save_slide_export: bool = False
    show_best: bool = False
    best_metric: MetricName = "psnr"
    show_last: bool = False
    show_early_stop: bool = True
    show_backtracking_stop: bool = True
    metrics: list[MetricName] = field(
        default_factory=lambda: ["psnr", "ssim", "haarpsi", "mse"]
    )
    x_axes: list[XAxis] = field(default_factory=lambda: ["iteration"])
    modes: list[PlotMode] = field(
        default_factory=lambda: [
            "individual",
            "methods-per-scenario",
            "scenarios-per-method",
        ]
    )
    snapshot_position: SnapshotPosition = "top"
    dpi: int = 300
    save_automatic_plots: bool = True


@dataclass(kw_only=True)
class PnPIterationMetricsParameter:
    """CLI options specific to PnP iteration comparisons."""

    method_ids: list[str]
    sample_name: str
    output_name: str
    category: str | None = None
    sample_id: int | None = None
    match_dataset_default_noise: bool = False
    prefix_datetime: bool = False
    include_no_noise: bool = True
    noise_realizations: list[int] = field(default_factory=lambda: [0])
    method_labels: dict[str, str] = field(default_factory=dict)
    plot: PnPIterationPlotParameter = field(default_factory=PnPIterationPlotParameter)


@dataclass(frozen=True)
class IterationMetrics:
    """Reconstruction metrics and cumulative reconstruction time over iterations."""

    iterations: list[int]
    times: list[float]
    psnr: list[float]
    ssim: list[float]
    haarpsi: list[float]
    mse: list[float]

    def metric_values(self, metric: MetricName) -> list[float]:
        """Return one metric series."""
        return getattr(self, metric)

    def x_values(self, x_axis: XAxis) -> list[int] | list[float]:
        """Return x values for iteration- or time-based plots."""
        return self.iterations if x_axis == "iteration" else self.times


@dataclass
class IterationSnapshots:
    """Selected reconstructed images over iterations."""

    explicit_images: dict[int, np.ndarray] = field(default_factory=dict)
    periodic_images: dict[int, np.ndarray] = field(default_factory=dict)
    best_iteration: int | None = None
    best_image: np.ndarray | None = None
    best_value: float | None = None
    last_iteration: int | None = None
    last_image: np.ndarray | None = None
    early_stop_iteration: int | None = None
    early_stop_image: np.ndarray | None = None
    backtracking_stop_iteration: int | None = None
    backtracking_stop_image: np.ndarray | None = None
    backtracking_attempted_iteration: int | None = None
    backtracking_failed_attempts: int | None = None
    backtracking_stepsize: float | None = None
    requested_iterations: int | None = None
    termination_reason: str | None = None


@dataclass(frozen=True)
class PlotSeries:
    """One labelled metric/snapshot series."""

    label: str
    metrics: IterationMetrics
    snapshots: IterationSnapshots


@dataclass(frozen=True)
class NoiseScenario:
    """One acquisition-noise scenario."""

    id: str
    label: str
    noise_realization: int | None


@dataclass(frozen=True)
class PnPIterationMetricsResult:
    """In-memory result used by interactive frontends such as the PCM GUI."""

    output_root: Path
    scenarios: list[NoiseScenario]
    metrics: dict[tuple[str, str], IterationMetrics]
    snapshots: dict[tuple[str, str], IterationSnapshots]
    comparison_config: PnPIterationMetricsParameter
    clim: tuple[float, float] | None
    plot_helper: PlotHelper
    phase_timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PnPProgressUpdate:
    """Progress information shared by terminal and GUI frontends."""

    phase: Literal["running", "saving", "finished"]
    current_run: int
    total_runs: int
    scenario_label: str
    method_label: str
    iteration: int
    total_iterations: int
    completed_iterations: int
    total_iterations_all: int
    elapsed_seconds: float
    estimated_remaining_seconds: float | None


@dataclass(frozen=True)
class SnapshotDisplay:
    """One resolved snapshot shown in a figure."""

    iteration: int
    title: str
    image: np.ndarray


def parse_pnp_iteration_metrics_cli(
    argv: list[str] | None = None,
) -> tuple[PnPIterationMetricsParameter, PCMExperimentParameter]:
    """Parse comparison-specific options, then the existing PCM experiment options."""
    if argv is None:
        argv = sys.argv[1:]

    comparison_config, pcm_args = tyro.cli(
        PnPIterationMetricsParameter,
        args=argv,
        add_help=False,
        return_unknown_args=True,
    )
    experiment_config = parse_pcm_experiment_config_cli(pcm_args)
    return comparison_config, experiment_config


def supports_iteration_metrics(method_config: object) -> bool:
    """Return whether a reconstruction method exposes per-iteration reconstructions."""
    return isinstance(method_config, PnPADMMParameter) or (
        isinstance(method_config, GSPnPParameter) and method_config.uses_deepinv
    )


def available_iteration_method_ids(config: PCMExperimentParameter) -> list[str]:
    """Return method IDs that can be used in the PnP iteration analysis."""
    return [
        method_id
        for method_id, method_config in config.methods.items()
        if supports_iteration_metrics(method_config)
    ]


def method_display_label(config: PnPIterationMetricsParameter, method_id: str) -> str:
    """Return a user-facing label for one method or parameter-sweep variant."""
    return config.method_labels.get(method_id, method_id)


def _make_scenarios(config: PnPIterationMetricsParameter) -> list[NoiseScenario]:
    scenarios: list[NoiseScenario] = []
    if config.include_no_noise:
        scenarios.append(
            NoiseScenario(id="no_noise", label="No noise", noise_realization=None)
        )
    scenarios.extend(
        NoiseScenario(
            id=f"noise_realization_{realization}",
            label=f"Noise realization {realization}",
            noise_realization=realization,
        )
        for realization in config.noise_realizations
    )
    if not scenarios:
        raise ValueError("At least one no-noise or noisy scenario is required.")
    return scenarios


def _validate_config(
    comparison_config: PnPIterationMetricsParameter,
    experiment_config: PCMExperimentParameter,
) -> None:
    if (comparison_config.category is None) != (comparison_config.sample_id is None):
        raise ValueError("Dataset category and sample ID must be provided together.")
    if len(make_test_cases(experiment_config)) != 1:
        raise ValueError(
            "PnP iteration plots currently require exactly one sampling/coarse-J test case."
        )
    if (
        experiment_config.runtime.num_trials != 1
        or experiment_config.runtime.num_trials_skip != 0
    ):
        raise ValueError(
            "PnP iteration plots currently require exactly one trial with no skipped trials."
        )

    missing = [
        method_id
        for method_id in comparison_config.method_ids
        if method_id not in experiment_config.methods
    ]
    if missing:
        raise KeyError(f"Methods not found in preset/config: {missing}")

    unsupported = [
        method_id
        for method_id in comparison_config.method_ids
        if not supports_iteration_metrics(experiment_config.methods[method_id])
    ]
    if unsupported:
        raise TypeError(
            "Iteration plotting currently supports PnP-ADMM and DeepInv GS-PnP methods. "
            f"Unsupported methods: {unsupported}"
        )

    if comparison_config.noise_realizations and experiment_config.noise is None:
        raise ValueError(
            "Noise realizations were requested, but the selected PCM configuration has no noise model."
        )

    if not comparison_config.plot.metrics:
        raise ValueError("Select at least one metric to plot.")
    if not comparison_config.plot.x_axes:
        raise ValueError("Select at least one x-axis.")
    if any(iteration < 1 for iteration in comparison_config.plot.snapshot_iterations):
        raise ValueError("Snapshot iterations must be positive.")
    if comparison_config.plot.snapshot_every < 0:
        raise ValueError(
            "Snapshot every N must be zero (disabled) or a positive integer."
        )
    if (
        comparison_config.plot.save_slide_export
        and "spgl1" not in experiment_config.methods
    ):
        raise ValueError(
            "Slide export requires an SPGL1 method in the selected experiment preset."
        )


def find_iteration_metrics(run_dir: Path, method_id: str) -> Path:
    """Find the iteration-metric CSV produced by one run."""
    matches = list(run_dir.rglob(f"{method_id}/iteration_metrics.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one iteration_metrics.csv under {run_dir}, found {len(matches)}"
        )
    return matches[0]


def load_iteration_metrics(path: Path) -> IterationMetrics:
    """Load iteration metrics and cumulative reconstruction time from one CSV."""
    iterations: list[int] = []
    times: list[float] = []
    psnr: list[float] = []
    ssim: list[float] = []
    haarpsi: list[float] = []
    mse: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            iterations.append(int(row["iteration"]))
            times.append(float(row["time"]))
            psnr.append(float(row["psnr"]))
            ssim.append(float(row["ssim"]))
            haarpsi.append(float(row["haarpsi"]))
            mse.append(float(row["mse"]))

    return IterationMetrics(
        iterations=iterations,
        times=times,
        psnr=psnr,
        ssim=ssim,
        haarpsi=haarpsi,
        mse=mse,
    )


def _is_better(metric: MetricName, value: float, best_value: float | None) -> bool:
    if best_value is None:
        return True
    if metric == "mse":
        return value < best_value
    return value > best_value


class _ProgressReporter:
    """Report one coherent analysis-level progress stream."""

    def __init__(
        self,
        *,
        comparison_config: PnPIterationMetricsParameter,
        experiment_config: PCMExperimentParameter,
        scenarios: list[NoiseScenario],
        callback: Callable[[PnPProgressUpdate], None] | None,
    ) -> None:
        self.comparison_config = comparison_config
        self.scenarios = scenarios
        self.callback = callback
        self.started_at = time.perf_counter()
        self.method_iterations = {
            method_id: int(experiment_config.methods[method_id].iters)
            for method_id in comparison_config.method_ids
        }
        self.iterations_per_scenario = sum(self.method_iterations.values())
        self.total_iterations = self.iterations_per_scenario * len(scenarios)
        self.total_runs = len(comparison_config.method_ids) * len(scenarios)
        self.method_offsets: dict[str, int] = {}
        offset = 0
        for method_id in comparison_config.method_ids:
            self.method_offsets[method_id] = offset
            offset += self.method_iterations[method_id]

        self._terminal_completed = 0
        self._terminal_bar = None
        if callback is None:
            self._terminal_bar = tqdm(
                total=self.total_iterations,
                desc="PnP iteration analysis",
                unit="iter",
                dynamic_ncols=True,
            )

    def emit(
        self,
        *,
        scenario_index: int,
        scenario: NoiseScenario,
        method_id: str,
        iteration: int,
    ) -> None:
        method_index = self.comparison_config.method_ids.index(method_id)
        method_total = self.method_iterations[method_id]
        completed = (
            scenario_index * self.iterations_per_scenario
            + self.method_offsets[method_id]
            + min(iteration, method_total)
        )
        elapsed = time.perf_counter() - self.started_at
        eta = None
        if completed > 0 and completed < self.total_iterations:
            eta = elapsed * (self.total_iterations - completed) / completed

        update = PnPProgressUpdate(
            phase="running",
            current_run=scenario_index * len(self.comparison_config.method_ids)
            + method_index
            + 1,
            total_runs=self.total_runs,
            scenario_label=scenario.label,
            method_label=method_display_label(self.comparison_config, method_id),
            iteration=iteration,
            total_iterations=method_total,
            completed_iterations=completed,
            total_iterations_all=self.total_iterations,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=eta,
        )
        self._dispatch(update)

    def saving(self) -> None:
        elapsed = time.perf_counter() - self.started_at
        update = PnPProgressUpdate(
            phase="saving",
            current_run=self.total_runs,
            total_runs=self.total_runs,
            scenario_label="",
            method_label="Saving plots",
            iteration=0,
            total_iterations=0,
            completed_iterations=self.total_iterations,
            total_iterations_all=self.total_iterations,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=None,
        )
        self._dispatch(update)

    def finish(self) -> None:
        elapsed = time.perf_counter() - self.started_at
        update = PnPProgressUpdate(
            phase="finished",
            current_run=self.total_runs,
            total_runs=self.total_runs,
            scenario_label="",
            method_label="Finished",
            iteration=0,
            total_iterations=0,
            completed_iterations=self.total_iterations,
            total_iterations_all=self.total_iterations,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=0.0,
        )
        self._dispatch(update)
        if self._terminal_bar is not None:
            self._terminal_bar.close()

    def _dispatch(self, update: PnPProgressUpdate) -> None:
        if self.callback is not None:
            self.callback(update)
            return
        assert self._terminal_bar is not None
        delta = max(0, update.completed_iterations - self._terminal_completed)
        if delta:
            self._terminal_bar.update(delta)
            self._terminal_completed = update.completed_iterations
        if update.phase == "running":
            self._terminal_bar.set_description(
                f"{update.method_label} | {update.scenario_label}"
            )
        elif update.phase == "saving":
            self._terminal_bar.set_description("Saving plots")


def _make_snapshot_callback_factory(
    *,
    scenario_id: str,
    scenario_index: int,
    scenario: NoiseScenario,
    plot_config: PnPIterationPlotParameter,
    snapshots: dict[tuple[str, str], IterationSnapshots],
    progress: _ProgressReporter,
) -> MethodIterationCallbackFactory:
    snapshot_iteration_set = set(plot_config.snapshot_iterations)

    class SnapshotCallback:
        """Collect selected snapshots while remaining callable as an iteration callback."""

        def __init__(self, method_id: str) -> None:
            self.method_id = method_id
            self.method_snapshots = IterationSnapshots()
            snapshots[(scenario_id, method_id)] = self.method_snapshots

        def __call__(
            self, iteration: int, recon: torch.Tensor, row: dict[str, float | int]
        ) -> None:
            progress.emit(
                scenario_index=scenario_index,
                scenario=scenario,
                method_id=self.method_id,
                iteration=iteration,
            )

            is_periodic = (
                plot_config.snapshot_every > 0
                and iteration % plot_config.snapshot_every == 0
            )
            best_value = float(row[plot_config.best_metric])
            is_best = plot_config.show_best and _is_better(
                plot_config.best_metric,
                best_value,
                self.method_snapshots.best_value,
            )
            needs_image = (
                iteration in snapshot_iteration_set
                or is_periodic
                or plot_config.show_last
                or is_best
            )
            image = recon.squeeze().detach().cpu().numpy() if needs_image else None

            if iteration in snapshot_iteration_set and image is not None:
                self.method_snapshots.explicit_images[iteration] = image

            if is_periodic and image is not None:
                self.method_snapshots.periodic_images[iteration] = image

            if plot_config.show_last and image is not None:
                self.method_snapshots.last_iteration = iteration
                self.method_snapshots.last_image = image

            if is_best and image is not None:
                self.method_snapshots.best_iteration = iteration
                self.method_snapshots.best_image = image
                self.method_snapshots.best_value = best_value

        def early_stop(self, iteration: int, recon: torch.Tensor) -> None:
            """Record the first iterate that would have triggered GS-PnP early stopping."""
            if not plot_config.show_early_stop:
                return
            self.method_snapshots.early_stop_iteration = iteration
            self.method_snapshots.early_stop_image = (
                recon.squeeze().detach().cpu().numpy()
            )

        def backtracking_stop(
            self,
            last_accepted_iteration: int,
            recon: torch.Tensor,
            attempted_iteration: int,
            failed_attempts: int,
            stepsize: float,
        ) -> None:
            """Record the last valid iterate when GS-PnP backtracking is exhausted."""
            self.method_snapshots.backtracking_stop_iteration = last_accepted_iteration
            self.method_snapshots.backtracking_attempted_iteration = attempted_iteration
            self.method_snapshots.backtracking_failed_attempts = failed_attempts
            self.method_snapshots.backtracking_stepsize = stepsize
            self.method_snapshots.termination_reason = "backtracking stop"
            if plot_config.show_backtracking_stop:
                self.method_snapshots.backtracking_stop_image = (
                    recon.squeeze().detach().cpu().numpy()
                )

    def factory(_run, method_id):
        if method_id not in progress.method_iterations:
            return None
        progress.emit(
            scenario_index=scenario_index,
            scenario=scenario,
            method_id=method_id,
            iteration=0,
        )
        return SnapshotCallback(method_id)

    return factory


def _resolved_snapshots(
    item: PlotSeries,
    plot_config: PnPIterationPlotParameter,
) -> list[SnapshotDisplay]:
    ordered_iterations: list[int] = []
    entries: dict[int, dict[str, object]] = {}

    def add(iteration: int | None, image: np.ndarray | None, tag: str | None) -> None:
        if iteration is None or image is None:
            return
        if iteration not in entries:
            entries[iteration] = {
                "image": image,
                "explicit": False,
                "tags": [],
            }
            ordered_iterations.append(iteration)
        if tag is None:
            entries[iteration]["explicit"] = True
        else:
            tags = entries[iteration]["tags"]
            assert isinstance(tags, list)
            if tag not in tags:
                tags.append(tag)

    for iteration in plot_config.snapshot_iterations:
        add(iteration, item.snapshots.explicit_images.get(iteration), None)

    if plot_config.snapshot_every > 0:
        for iteration in sorted(item.snapshots.periodic_images):
            add(iteration, item.snapshots.periodic_images.get(iteration), None)

    if plot_config.show_best:
        add(item.snapshots.best_iteration, item.snapshots.best_image, "best")
    if plot_config.show_last:
        add(item.snapshots.last_iteration, item.snapshots.last_image, "last")
    if plot_config.show_early_stop:
        add(
            item.snapshots.early_stop_iteration,
            item.snapshots.early_stop_image,
            "early stop",
        )
    if plot_config.show_backtracking_stop:
        add(
            item.snapshots.backtracking_stop_iteration,
            item.snapshots.backtracking_stop_image,
            "backtracking stop",
        )

    displays: list[SnapshotDisplay] = []
    for iteration in ordered_iterations:
        entry = entries[iteration]
        image = entry["image"]
        tags = entry["tags"]
        explicit = bool(entry["explicit"])
        assert isinstance(image, np.ndarray)
        assert isinstance(tags, list)

        if explicit:
            title = f"iter {iteration}"
            if tags:
                title += "\n" + ", ".join(tags)
        else:
            title = ", ".join(tags) + f"\niter {iteration}"

        displays.append(SnapshotDisplay(iteration=iteration, title=title, image=image))

    return displays


def resolve_snapshots(
    item: PlotSeries, plot_config: PnPIterationPlotParameter
) -> list[SnapshotDisplay]:
    """Resolve explicit, best, and last snapshots without duplicates."""
    return _resolved_snapshots(item, plot_config)


def _metric_value_at_iteration(
    metrics: IterationMetrics,
    metric: MetricName,
    iteration: int,
) -> float | None:
    try:
        index = metrics.iterations.index(iteration)
    except ValueError:
        return None
    return metrics.metric_values(metric)[index]


def _x_value_at_iteration(
    metrics: IterationMetrics,
    x_axis: XAxis,
    iteration: int,
) -> int | float | None:
    try:
        index = metrics.iterations.index(iteration)
    except ValueError:
        return None
    return metrics.x_values(x_axis)[index]


def _plot_metrics(
    axes: list,
    series: list[PlotSeries],
    plot_config: PnPIterationPlotParameter,
    x_axis: XAxis,
) -> None:
    for ax, metric in zip(axes, plot_config.metrics, strict=True):
        for item in series:
            x_values = item.metrics.x_values(x_axis)
            y_values = item.metrics.metric_values(metric)
            line = ax.plot(x_values, y_values, label=item.label)[0]

            selected_iterations = [
                snapshot.iteration
                for snapshot in _resolved_snapshots(item, plot_config)
            ]
            points = []
            for iteration in selected_iterations:
                x_value = _x_value_at_iteration(item.metrics, x_axis, iteration)
                y_value = _metric_value_at_iteration(item.metrics, metric, iteration)
                if x_value is not None and y_value is not None:
                    points.append((x_value, y_value))

            if points:
                ax.scatter(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=line.get_color(),
                    zorder=3,
                )

        ax.set_xlabel(
            "PnP iteration" if x_axis == "iteration" else "Reconstruction time (s)"
        )
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, alpha=0.3)


def _add_common_legend(
    fig,
    axes: list,
    *,
    position: LegendPosition,
    columns: int = 1,
) -> None:
    """Add one figure-level legend using layout-aware outside placement."""
    if position == "none" or not axes:
        return

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        return

    columns = max(1, min(columns, len(labels)))
    locations = {
        "right": "outside right center",
        "bottom": "outside lower center",
        "top": "outside upper center",
    }
    location = locations[position]

    try:
        fig.legend(
            handles,
            labels,
            loc=location,
            ncol=columns,
            frameon=False,
        )
    except ValueError:
        # Compatibility fallback for older Matplotlib versions without
        # constrained-layout "outside" legend locations.
        fallback = {
            "right": ("center left", (0.80, 0.5)),
            "bottom": ("lower center", (0.5, 0.005)),
            "top": ("upper center", (0.5, 0.995)),
        }
        loc, anchor = fallback[position]
        fig.legend(
            handles,
            labels,
            loc=loc,
            bbox_to_anchor=anchor,
            ncol=columns,
            frameon=False,
        )


def _comparison_x_values(
    metrics: IterationMetrics,
    x_axis: XAxis,
    time_origin: TimeOrigin,
) -> list[int] | list[float]:
    """Return plot x values, optionally aligning time curves at their first recorded point."""
    values = metrics.x_values(x_axis)
    if x_axis != "time" or time_origin == "reconstruction-start" or not values:
        return values

    first = float(values[0])
    return [float(value) - first for value in values]


def _comparison_x_label(x_axis: XAxis, time_origin: TimeOrigin) -> str:
    """Return a label describing the selected x-axis convention."""
    if x_axis == "iteration":
        return "PnP iteration"
    if time_origin == "first-point":
        return "Time since first recorded iteration (s)"
    return "Reconstruction time (s)"


def make_metric_comparison_figure(
    series: list[PlotSeries],
    *,
    metrics: list[MetricName],
    x_axis: XAxis,
    title: str = "",
    figsize: tuple[float, float] = (10.0, 7.0),
    dpi: int = 120,
    line_width: float = 1.8,
    show_markers: bool = False,
    grid: bool = True,
    legend_position: LegendPosition = "right",
    legend_columns: int = 1,
    time_origin: TimeOrigin = "reconstruction-start",
):
    """Build a metric-only comparison figure for interactive frontends."""
    if not series:
        raise ValueError("Select at least one reconstruction series.")
    if not metrics:
        raise ValueError("Select at least one metric.")

    cols = 1 if len(metrics) == 1 else 2
    rows = math.ceil(len(metrics) / cols)

    fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
    grid_spec = fig.add_gridspec(rows, cols)
    axes = []

    for index, metric in enumerate(metrics):
        ax = fig.add_subplot(grid_spec[index // cols, index % cols])
        axes.append(ax)

        for item in series:
            kwargs = {"linewidth": line_width, "label": item.label}
            if show_markers:
                kwargs.update(marker="o", markersize=3.5)
            ax.plot(
                _comparison_x_values(item.metrics, x_axis, time_origin),
                item.metrics.metric_values(metric),
                **kwargs,
            )

        ax.set_xlabel(_comparison_x_label(x_axis, time_origin))
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(grid, alpha=0.3)

    for index in range(len(metrics), rows * cols):
        ax = fig.add_subplot(grid_spec[index // cols, index % cols])
        ax.axis("off")

    if title:
        fig.suptitle(title)

    _add_common_legend(
        fig,
        axes,
        position=legend_position,
        columns=legend_columns,
    )
    return fig


def _make_metric_axes(fig, panel_spec, num_metrics: int) -> list:
    cols = 1 if num_metrics == 1 else 2
    rows = math.ceil(num_metrics / cols)
    grid = panel_spec.subgridspec(rows, cols)
    return [
        fig.add_subplot(grid[index // cols, index % cols])
        for index in range(num_metrics)
    ]


def _plot_snapshots(
    fig,
    panel_spec,
    series: list[PlotSeries],
    plot_config: PnPIterationPlotParameter,
    *,
    plot_helper: PlotHelper,
) -> None:
    series_grid = panel_spec.subgridspec(max(1, len(series)), 1)

    for row, item in enumerate(series):
        displays = _resolved_snapshots(item, plot_config)
        row_grid = series_grid[row, 0].subgridspec(1, max(1, len(displays)))

        if not displays:
            ax = fig.add_subplot(row_grid[0, 0])
            ax.axis("off")
            ax.set_title(item.label, fontsize=9)
            continue

        for col, display in enumerate(displays):
            ax = fig.add_subplot(row_grid[0, col])
            image = np.asarray(display.image)
            if plot_helper.color.inverse_sign:
                image = -image
            cmap = plot_helper.get_cmap()
            clim = plot_helper.color.clim
            if clim is None:
                ax.imshow(image, cmap=cmap)
            else:
                ax.imshow(image, cmap=cmap, vmin=clim[0], vmax=clim[1])
            ax.set_xticks([])
            ax.set_yticks([])
            title = display.title
            if col == 0:
                title = f"{item.label}\n{title}"
            ax.set_title(title, fontsize=9)


def _save_series_figure(
    *,
    series: list[PlotSeries],
    plot_config: PnPIterationPlotParameter,
    x_axis: XAxis,
    output_path: Path,
    title: str,
    plot_helper: PlotHelper,
) -> None:
    has_snapshots = (
        bool(plot_config.snapshot_iterations)
        or plot_config.snapshot_every > 0
        or plot_config.show_best
        or plot_config.show_last
        or plot_config.show_early_stop
        or plot_config.show_backtracking_stop
    )
    num_series = max(1, len(series))
    max_snapshots = max(
        [len(_resolved_snapshots(item, plot_config)) for item in series] or [0]
    )
    metric_rows = (
        math.ceil(len(plot_config.metrics) / 2) if len(plot_config.metrics) > 1 else 1
    )

    if not has_snapshots:
        fig = plt.figure(figsize=(10, 3.6 * metric_rows), layout="constrained")
        outer = fig.add_gridspec(1, 1)
        axes = _make_metric_axes(fig, outer[0, 0], len(plot_config.metrics))
        _plot_metrics(axes, series, plot_config, x_axis)
    elif plot_config.snapshot_position in {"top", "bottom"}:
        fig = plt.figure(
            figsize=(
                max(10.0, 2.0 * max(1, max_snapshots)),
                3.6 * metric_rows + 2.0 * num_series,
            ),
            layout="constrained",
        )
        snapshot_index = 0 if plot_config.snapshot_position == "top" else 1
        metrics_index = 1 - snapshot_index
        height_ratios = (
            [2.0 * num_series, 3.6 * metric_rows]
            if plot_config.snapshot_position == "top"
            else [3.6 * metric_rows, 2.0 * num_series]
        )
        outer = fig.add_gridspec(2, 1, height_ratios=height_ratios)
        _plot_snapshots(
            fig, outer[snapshot_index], series, plot_config, plot_helper=plot_helper
        )
        axes = _make_metric_axes(fig, outer[metrics_index], len(plot_config.metrics))
        _plot_metrics(axes, series, plot_config, x_axis)
    else:
        fig = plt.figure(
            figsize=(
                10.0 + 1.6 * max(1, max_snapshots),
                max(7.0, 3.2 * metric_rows, 2.0 * num_series),
            ),
            layout="constrained",
        )
        snapshot_index = 0 if plot_config.snapshot_position == "left" else 1
        metrics_index = 1 - snapshot_index
        width_ratios = (
            [2.0 * max(1, max_snapshots), 6.0]
            if plot_config.snapshot_position == "left"
            else [6.0, 2.0 * max(1, max_snapshots)]
        )
        outer = fig.add_gridspec(1, 2, width_ratios=width_ratios)
        _plot_snapshots(
            fig, outer[snapshot_index], series, plot_config, plot_helper=plot_helper
        )
        axes = _make_metric_axes(fig, outer[metrics_index], len(plot_config.metrics))
        _plot_metrics(axes, series, plot_config, x_axis)

    axis_title = "iteration" if x_axis == "iteration" else "reconstruction time"
    fig.suptitle(f"{title} | vs {axis_title}")
    legend_position: LegendPosition = "right" if len(series) > 1 else "none"
    _add_common_legend(fig, axes, position=legend_position)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path.with_suffix(".png"), dpi=plot_config.dpi, bbox_inches="tight"
    )
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _save_series_figures(
    *,
    series: list[PlotSeries],
    plot_config: PnPIterationPlotParameter,
    output_path: Path,
    title: str,
    plot_helper: PlotHelper,
) -> None:
    for x_axis in plot_config.x_axes:
        _save_series_figure(
            series=series,
            plot_config=plot_config,
            x_axis=x_axis,
            output_path=output_path.parent / f"{output_path.name}_vs_{x_axis}",
            title=title,
            plot_helper=plot_helper,
        )


def _save_requested_plots(
    *,
    comparison_config: PnPIterationMetricsParameter,
    experiment_config: PCMExperimentParameter,
    scenarios: list[NoiseScenario],
    metrics: dict[tuple[str, str], IterationMetrics],
    snapshots: dict[tuple[str, str], IterationSnapshots],
    output_root: Path,
) -> None:
    if "individual" in comparison_config.plot.modes:
        for scenario in scenarios:
            for method_id in comparison_config.method_ids:
                item = PlotSeries(
                    label=method_display_label(comparison_config, method_id),
                    metrics=metrics[(scenario.id, method_id)],
                    snapshots=snapshots[(scenario.id, method_id)],
                )
                _save_series_figures(
                    series=[item],
                    plot_config=comparison_config.plot,
                    output_path=output_root
                    / "plots"
                    / "individual"
                    / scenario.id
                    / method_id,
                    title=f"{method_id} | {scenario.label}",
                    plot_helper=experiment_config.plot,
                )

    if "methods-per-scenario" in comparison_config.plot.modes:
        for scenario in scenarios:
            series = [
                PlotSeries(
                    label=method_display_label(comparison_config, method_id),
                    metrics=metrics[(scenario.id, method_id)],
                    snapshots=snapshots[(scenario.id, method_id)],
                )
                for method_id in comparison_config.method_ids
            ]
            _save_series_figures(
                series=series,
                plot_config=comparison_config.plot,
                output_path=output_root
                / "plots"
                / "methods_per_scenario"
                / scenario.id,
                title=scenario.label,
                plot_helper=experiment_config.plot,
            )

    if "scenarios-per-method" in comparison_config.plot.modes:
        for method_id in comparison_config.method_ids:
            series = [
                PlotSeries(
                    label=scenario.label,
                    metrics=metrics[(scenario.id, method_id)],
                    snapshots=snapshots[(scenario.id, method_id)],
                )
                for scenario in scenarios
            ]
            _save_series_figures(
                series=series,
                plot_config=comparison_config.plot,
                output_path=output_root / "plots" / "scenarios_per_method" / method_id,
                title=method_display_label(comparison_config, method_id),
                plot_helper=experiment_config.plot,
            )


def run_pnp_iteration_metrics_analysis(
    comparison_config: PnPIterationMetricsParameter,
    experiment_config: PCMExperimentParameter,
    *,
    progress_callback: Callable[[PnPProgressUpdate], None] | None = None,
    analysis_definition: dict | None = None,
) -> PnPIterationMetricsResult:
    """Run selected methods and return saved outputs plus in-memory snapshots."""
    _validate_config(comparison_config, experiment_config)
    scenarios = _make_scenarios(comparison_config)
    output_name = name_with_datetime_prefix(
        comparison_config.output_name,
        comparison_config.prefix_datetime,
    )
    output_root = experiment_config.runtime.root_output_dir / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    if analysis_definition is not None:
        definition_to_save = copy.deepcopy(analysis_definition)
        definition_to_save["last_run"] = {"output_root": str(output_root)}
        analysis_path = save_analysis_definition(
            output_root / "analysis.yaml", definition_to_save
        )
        print(f"Saved PnP iteration analysis definition to {analysis_path}")

    progress = _ProgressReporter(
        comparison_config=comparison_config,
        experiment_config=experiment_config,
        scenarios=scenarios,
        callback=progress_callback,
    )

    metrics: dict[tuple[str, str], IterationMetrics] = {}
    snapshots: dict[tuple[str, str], IterationSnapshots] = {}
    analysis_started_at = time.perf_counter()

    for scenario_index, scenario in enumerate(scenarios):
        scenario_config = copy.deepcopy(experiment_config)
        method_ids = list(comparison_config.method_ids)
        if comparison_config.plot.save_slide_export:
            method_ids.append("spgl1")
            if "scalar_tv" in scenario_config.methods:
                method_ids.append("scalar_tv")
        scenario_config.methods = {
            method_id: scenario_config.methods[method_id] for method_id in method_ids
        }
        scenario_config.log.save_metrics = True
        if not comparison_config.plot.save_automatic_plots:
            # The iteration-analysis GUI already keeps snapshots/metrics in memory.
            # Avoid the generic PCM PNG/comparison figures unless explicitly requested.
            scenario_config.log.save_plots = False
        scenario_config.runtime.root_output_dir = output_root / scenario.id

        if scenario.noise_realization is None:
            scenario_config.noise = None

        callback_factory = _make_snapshot_callback_factory(
            scenario_id=scenario.id,
            scenario_index=scenario_index,
            scenario=scenario,
            plot_config=comparison_config.plot,
            snapshots=snapshots,
            progress=progress,
        )
        # Dataset runs normally draw noise from the sampling RNG itself.
        # Preserve that convention for realization 0 when explicitly requested.
        noise_seed = scenario.noise_realization
        if comparison_config.match_dataset_default_noise and noise_seed == 0:
            noise_seed = None
        run_result = run_single_sample_experiment(
            scenario_config,
            sample_name=comparison_config.sample_name,
            category=comparison_config.category,
            sample_id=comparison_config.sample_id,
            return_last_run=comparison_config.plot.save_slide_export,
            method_iteration_callback_factory=callback_factory,
            noise_seed=noise_seed,
            show_progress_bars=False,
        )

        slide_image_dir = None
        if comparison_config.plot.save_slide_export:
            assert isinstance(run_result, PCMSingleSampleTrialRun)
            run_dir = run_result.trial.results_root
            slide_image_dir = (
                output_root
                / "slide_export"
                / "figures"
                / "pnp_convergence"
                / scenario.id
            )
            slide_image_dir.mkdir(parents=True, exist_ok=True)
            scenario_config.plot.save_image(
                run_result.trial.target, slide_image_dir / "ground_truth.png"
            )
            slide_data_dir = output_root / "slide_export" / "real_data" / scenario.id
            slide_data_dir.mkdir(parents=True, exist_ok=True)
            reference_results = {
                "zero_filled": run_result.zf_result,
                "spgl1": run_result.results["spgl1"],
            }
            if "scalar_tv" in run_result.results:
                reference_results["scalar_tv"] = run_result.results["scalar_tv"]
            for reference_id, reference_result in reference_results.items():
                scenario_config.plot.save_image(
                    reference_result.recon, slide_image_dir / f"{reference_id}.png"
                )
                if (
                    reference_result.metrics is None
                    or reference_result.duration is None
                ):
                    raise RuntimeError(
                        f"{reference_id} result has no HaarPSI metric or reconstruction duration."
                    )
                with (slide_data_dir / f"{reference_id}_metrics.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    writer = csv.DictWriter(stream, fieldnames=("haarpsi", "time"))
                    writer.writeheader()
                    writer.writerow(
                        {
                            "haarpsi": reference_result.metrics["haarpsi"],
                            "time": reference_result.duration,
                        }
                    )
            if "scalar_tv" in run_result.results:
                shutil.copyfile(
                    find_iteration_metrics(run_dir, "scalar_tv"),
                    slide_data_dir / "scalar_tv_iteration_metrics.csv",
                )
        else:
            assert isinstance(run_result, Path)
            run_dir = run_result

        for method_id in comparison_config.method_ids:
            iteration_metrics_path = find_iteration_metrics(run_dir, method_id)
            method_metrics = load_iteration_metrics(iteration_metrics_path)
            metrics[(scenario.id, method_id)] = method_metrics

            method_snapshots = snapshots[(scenario.id, method_id)]
            method_config = scenario_config.methods[method_id]
            requested_iterations = getattr(method_config, "iters", None)
            method_snapshots.requested_iterations = requested_iterations

            if slide_image_dir is not None:
                slide_data_dir = (
                    output_root / "slide_export" / "real_data" / scenario.id
                )
                slide_data_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(
                    iteration_metrics_path,
                    slide_data_dir / f"{method_id}_iteration_metrics.csv",
                )
                method_image_dir = slide_image_dir / method_id
                method_image_dir.mkdir(parents=True, exist_ok=True)
                series = PlotSeries(
                    label=method_id, metrics=method_metrics, snapshots=method_snapshots
                )
                for snapshot in resolve_snapshots(series, comparison_config.plot):
                    scenario_config.plot.save_image(
                        torch.as_tensor(snapshot.image),
                        method_image_dir / f"iter_{snapshot.iteration:02d}.png",
                    )

            accepted_iterations = (
                method_metrics.iterations[-1] if method_metrics.iterations else 0
            )
            if method_snapshots.termination_reason is None:
                if (
                    isinstance(method_config, GSPnPParameter)
                    and method_config.early_stop
                    and not method_config.continue_after_early_stop
                    and method_snapshots.early_stop_iteration is not None
                ):
                    method_snapshots.termination_reason = "early stop"
                elif (
                    requested_iterations is not None
                    and accepted_iterations >= requested_iterations
                ):
                    method_snapshots.termination_reason = "requested iterations reached"
                else:
                    method_snapshots.termination_reason = (
                        "stopped before requested iterations"
                    )

    runs_finished_at = time.perf_counter()
    automatic_plot_seconds = 0.0
    if comparison_config.plot.save_automatic_plots:
        progress.saving()
        plot_started_at = time.perf_counter()
        _save_requested_plots(
            comparison_config=comparison_config,
            experiment_config=experiment_config,
            scenarios=scenarios,
            metrics=metrics,
            snapshots=snapshots,
            output_root=output_root,
        )
        automatic_plot_seconds = time.perf_counter() - plot_started_at

    total_seconds = time.perf_counter() - analysis_started_at
    run_seconds = runs_finished_at - analysis_started_at
    progress.finish()
    print(
        "PnP iteration analysis timing: "
        f"runs/data/metric logging={run_seconds:.2f}s, "
        f"automatic plots={automatic_plot_seconds:.2f}s, "
        f"total={total_seconds:.2f}s"
    )
    print(f"PnP iteration comparison results: {output_root}")
    return PnPIterationMetricsResult(
        output_root=output_root,
        scenarios=scenarios,
        metrics=metrics,
        snapshots=snapshots,
        comparison_config=comparison_config,
        clim=experiment_config.plot.color.clim,
        plot_helper=experiment_config.plot,
        phase_timings={
            "runs": run_seconds,
            "automatic_plots": automatic_plot_seconds,
            "total": total_seconds,
        },
    )
