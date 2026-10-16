"""Tune scalar-TV lambda for one PCM image with shared measurements."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from LION.experiments.pcm_presets.presets import PRESETS
from LION.pcm.dataset_single_sample import is_synthetic_dataset, select_synthetic_sample
from LION.pcm.reconstruct import ScalarTVParameter
from LION.pcm.run_pcm_single_sample_experiment import run_single_sample_experiment
from LION.pcm.run_pcm_single_sample_trial import PCMSingleSampleTrialRun


def _method_id(lam: float) -> str:
    return f"scalar_tv_lam_{lam:g}".replace(".", "p").replace("-", "m")


def _save_curves(
    rows_by_lambda: dict[float, list[dict[str, str]]], output_dir: Path
) -> None:
    for metric, title in (("haarpsi", "HaarPSI"), ("objective", "TV objective")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for lam, rows in rows_by_lambda.items():
            ax.plot(
                [int(row["iteration"]) for row in rows],
                [float(row[metric]) for row in rows],
                label=f"lambda={lam:g}",
            )
        ax.set_xlabel("PDHG iteration")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"lambda_sweep_{metric}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default="synthetic_png_dataset_box_plots"
    )
    parser.add_argument("--category", default="CIGS")
    parser.add_argument("--sample-id", type=int, default=288)
    parser.add_argument(
        "--data-root", type=Path, help="Override the dataset root from the preset"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("pcm_tv_sweep_results")
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.5],
    )
    parser.add_argument("--iters", type=int, default=1024)
    parser.add_argument("--metric-every", type=int, default=4)
    parser.add_argument(
        "--scenario", choices=("both", "no-noise", "noise"), default="both"
    )
    args = parser.parse_args()

    if args.iters <= 0 or args.metric_every <= 0:
        parser.error("--iters and --metric-every must be positive")
    if any(lam < 0 for lam in args.lambdas) or len(set(args.lambdas)) != len(
        args.lambdas
    ):
        parser.error("--lambdas must be nonnegative and distinct")

    experiment = copy.deepcopy(PRESETS[args.preset])
    if args.data_root is not None:
        experiment.data.root = args.data_root.expanduser()
    # Previous restriction (synthetic datasets only):
    # if not is_synthetic_dataset(experiment):
    #     parser.error("The lambda sweep currently requires a synthetic PCM dataset preset")
    synthetic_dataset = is_synthetic_dataset(experiment)
    if synthetic_dataset:
        try:
            sample_name = select_synthetic_sample(
                experiment, args.category, args.sample_id
            )
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        category, sample_id = args.category, args.sample_id
    else:
        if experiment.data.data_type != "image" or experiment.this_sample is None:
            parser.error(
                "The lambda sweep requires a synthetic dataset or a single-image PCM preset"
            )
        required_image = experiment.data.root / experiment.this_sample.data_filename
        if not required_image.is_file():
            parser.error(f"Input image not found: {required_image}")
        sample_name = args.preset
        category = sample_id = None

    experiment.runtime.device = args.device
    experiment.runtime.num_trials = 1
    experiment.runtime.num_trials_skip = 0
    experiment.sampling.sampling_ratios = [0.2]
    experiment.sampling.coarse_j_values = [6]
    experiment.methods = {
        _method_id(lam): ScalarTVParameter(
            lambda_reg=lam, iters=args.iters, metric_every=args.metric_every
        )
        for lam in args.lambdas
    }
    experiment.log.save_metrics = True
    experiment.log.save_plots = True
    experiment.log.save_recon = True
    experiment.log.save_diff = False
    experiment.log.save_zero_filled = True

    output_root = args.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sweep_settings.json").write_text(
        json.dumps(
            {
                "preset": args.preset,
                "category": category,
                "sample_id": sample_id,
                "input_image": str(
                    experiment.data.root / experiment.this_sample.data_filename
                ),
                "sampling_ratio": 0.2,
                "coarse_j": 6,
                "base_seed": experiment.base_seed,
                "lambda_values": args.lambdas,
                "iters": args.iters,
                "metric_every": args.metric_every,
                "scenario": args.scenario,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    scenarios = (
        ["no_noise", "noise_realization_0"]
        if args.scenario == "both"
        else ["no_noise" if args.scenario == "no-noise" else "noise_realization_0"]
    )
    summary: list[dict[str, float | str | int]] = []
    for scenario in scenarios:
        scenario_config = copy.deepcopy(experiment)
        scenario_config.runtime.root_output_dir = output_root / scenario
        if scenario == "no_noise":
            scenario_config.noise = None
        run = run_single_sample_experiment(
            scenario_config,
            sample_name=sample_name,
            category=category,
            sample_id=sample_id,
            return_last_run=True,
            # Match the slide runner: synthetic samples use the dataset RNG;
            # single-image presets use independent noise realization 0.
            noise_seed=None if synthetic_dataset or scenario == "no_noise" else 0,
            show_progress_bars=True,
        )
        assert isinstance(run, PCMSingleSampleTrialRun)
        traces = {}
        for lam in args.lambdas:
            method_id = _method_id(lam)
            csv_path = run.method_dir(method_id) / "iteration_metrics.csv"
            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            traces[lam] = rows
            result = run.results[method_id]
            assert result.metrics is not None and result.duration is not None
            summary.append(
                {
                    "scenario": scenario,
                    "lambda_reg": lam,
                    "iters": args.iters,
                    "psnr": result.metrics["psnr"],
                    "ssim": result.metrics["ssim"],
                    "haarpsi": result.metrics["haarpsi"],
                    "mse": result.metrics["mse"],
                    "duration": result.duration,
                    "objective": float(rows[-1]["objective"]),
                }
            )
        _save_curves(traces, scenario_config.runtime.root_output_dir)
        best = max(
            (row for row in summary if row["scenario"] == scenario),
            key=lambda row: float(row["haarpsi"]),
        )
        print(
            f"{scenario}: best final HaarPSI among tested lambdas is lambda={best['lambda_reg']:g} ({best['haarpsi']:.4f})"
        )

    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Scalar-TV lambda sweep saved to: {output_root}")


if __name__ == "__main__":
    main()
