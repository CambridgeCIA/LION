"""Run matching PnP, SPGL1, and ground-truth exports for the CIA slides."""

from __future__ import annotations

import argparse
from pathlib import Path

from LION.experiments.pcm_presets.presets import PRESETS
from LION.pcm.dataset_single_sample import (
    is_synthetic_dataset,
    select_synthetic_sample,
    with_iteration_methods,
)
from LION.pcm.pnp_iteration_metrics import (
    PnPIterationMetricsParameter,
    PnPIterationPlotParameter,
    run_pnp_iteration_metrics_analysis,
)
from LION.pcm.reconstruct import ScalarTVParameter

METHOD_IDS = ("pnp_admm_drunet", "pnp_gs_deepinv_drunet", "spgl1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default="synthetic_png_dataset_box_plots"
    )
    parser.add_argument("--category", default="CIGS", help="Synthetic dataset category")
    parser.add_argument(
        "--sample-id",
        type=int,
        default=288,
        help="Image ID, e.g. 288 for image0288.png",
    )
    parser.add_argument(
        "--data-root", type=Path, help="Override the input directory from the preset"
    )
    parser.add_argument("--output-root", type=Path, default=Path("pcm_slide_results"))
    parser.add_argument("--output-name", help="Experiment output directory name")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--stop-gs-early", action="store_true")
    parser.add_argument(
        "--independent-noise",
        action="store_true",
        help="Use an independent noise seed 0 instead of the dataset runner's original noise RNG",
    )
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--scalar-tv-lambda", type=float, default=0.001)
    # parser.add_argument("--scalar-tv-lambda", type=float, default=0.02)
    parser.add_argument("--tv-iters", type=int, default=1024)
    parser.add_argument("--tv-metric-every", type=int, default=4)
    parser.add_argument(
        "--no-scalar-tv", action="store_true", help="Skip scalar TV slide baseline"
    )
    parser.add_argument(
        "--snapshot-iterations",
        type=int,
        nargs="+",
        # Previous wider snapshot selection:
        # default=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 90, 100],
        default=[1, 5, 10, 20, 30, 50],
    )
    args = parser.parse_args()
    if args.max_iters < 1:
        parser.error("--max-iters must be positive")
    if any(i < 1 or i > args.max_iters for i in args.snapshot_iterations):
        parser.error("Snapshot iterations must lie between 1 and --max-iters")
    if args.scalar_tv_lambda < 0 or args.tv_iters <= 0 or args.tv_metric_every <= 0:
        parser.error(
            "Scalar TV lambda must be nonnegative and its iterations/metric interval must be positive"
        )

    experiment = with_iteration_methods(PRESETS[args.preset])
    if args.data_root is not None:
        experiment.data.root = args.data_root.expanduser()

    if is_synthetic_dataset(experiment):
        try:
            sample_name = select_synthetic_sample(
                experiment, args.category, args.sample_id
            )
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        category, sample_id = args.category, args.sample_id
    else:
        assert experiment.this_sample is not None
        required_image = experiment.data.root / experiment.this_sample.data_filename
        if not required_image.is_file():
            parser.error(f"Input image not found: {required_image}")
        sample_name = args.preset
        category = sample_id = None

    missing_methods = [
        method for method in METHOD_IDS if method not in experiment.methods
    ]
    if missing_methods:
        parser.error(f"Preset lacks required methods: {missing_methods}")

    experiment.runtime.root_output_dir = args.output_root.expanduser()
    experiment.runtime.device = args.device
    experiment.runtime.num_trials = 1
    experiment.runtime.num_trials_skip = 0
    experiment.sampling.sampling_ratios = [0.2]
    # experiment.sampling.sampling_ratios = [0.3]
    # experiment.sampling.sampling_ratios = [0.25]
    # experiment.sampling.sampling_ratios = [0.5]
    experiment.sampling.coarse_j_values = [6]
    experiment.methods = {method: experiment.methods[method] for method in METHOD_IDS}
    if not args.no_scalar_tv:
        experiment.methods["scalar_tv"] = ScalarTVParameter(
            lambda_reg=args.scalar_tv_lambda,
            iters=args.tv_iters,
            metric_every=args.tv_metric_every,
        )
    for method in METHOD_IDS[:2]:
        experiment.methods[method].iters = args.max_iters
    experiment.methods[
        "pnp_gs_deepinv_drunet"
    ].continue_after_early_stop = not args.stop_gs_early
    experiment.log.save_plots = False
    experiment.log.save_recon = False
    experiment.log.save_diff = False
    experiment.log.save_zero_filled = False

    output_name = args.output_name or (
        f"cia_pnp_slide_{category}_{sample_id:04d}"
        if category is not None
        else "cia_pnp_slide_data"
    )
    comparison = PnPIterationMetricsParameter(
        method_ids=list(METHOD_IDS[:2]),
        sample_name=sample_name,
        output_name=output_name,
        category=category,
        sample_id=sample_id,
        match_dataset_default_noise=category is not None and not args.independent_noise,
        include_no_noise=True,
        noise_realizations=[0],
        plot=PnPIterationPlotParameter(
            snapshot_iterations=args.snapshot_iterations,
            show_last=True,
            metrics=["haarpsi"],
            x_axes=["iteration", "time"],
            modes=["methods-per-scenario"],
            save_automatic_plots=False,
            save_slide_export=True,
        ),
    )
    result = run_pnp_iteration_metrics_analysis(comparison, experiment)
    print(f"Slide-ready files: {result.output_root / 'slide_export'}")


if __name__ == "__main__":
    main()
