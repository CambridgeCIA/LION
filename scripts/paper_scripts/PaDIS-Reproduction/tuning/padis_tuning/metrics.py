"""Metric summarisation and result writing for PaDIS tuning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from typing import Iterable


def finite_values(values: Iterable[float | None]) -> list[float]:
    """Return finite values."""
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def mean_or_none(values: Iterable[float | None]) -> float | None:
    """Return the mean or none."""
    finite = finite_values(values)
    if not finite:
        return None
    return sum(finite) / len(finite)


def min_or_none(values: Iterable[float | None]) -> float | None:
    """Return the minimum or none."""
    finite = finite_values(values)
    if not finite:
        return None
    return min(finite)


def max_or_none(values: Iterable[float | None]) -> float | None:
    """Return the maximum or none."""
    finite = finite_values(values)
    if not finite:
        return None
    return max(finite)


def summarize_metrics(metrics_path: pathlib.Path) -> dict:
    """Summarize metrics."""
    with open(metrics_path) as file:
        payload = json.load(file)
    metrics = payload.get("metrics", [])
    sampler = payload.get("sampler", {})
    method_settings = payload.get("method_settings", {})
    psnr_values = [item.get("psnr") for item in metrics]
    ssim_values = [item.get("ssim") for item in metrics]
    mae_values = [item.get("mae") for item in metrics]
    fdk_psnr_values = [item.get("fdk_psnr") for item in metrics]
    fdk_ssim_values = [item.get("fdk_ssim") for item in metrics]
    body_psnr_values = [item.get("body_psnr") for item in metrics]
    margins = [
        item.get("psnr") - item.get("fdk_psnr")
        for item in metrics
        if item.get("psnr") is not None and item.get("fdk_psnr") is not None
    ]
    primary_values = [
        value
        for item in metrics
        for value in (item.get("psnr"), item.get("ssim"), item.get("mae"))
    ]
    nonfinite_primary_metric_count = sum(
        1
        for value in primary_values
        if value is None or not math.isfinite(float(value))
    )
    return {
        "sample_count": len(metrics),
        "nonfinite_primary_metric_count": nonfinite_primary_metric_count,
        "all_finite_primary_metrics": nonfinite_primary_metric_count == 0,
        "mean_psnr": mean_or_none(psnr_values),
        "min_psnr": min_or_none(psnr_values),
        "mean_ssim": mean_or_none(ssim_values),
        "min_ssim": min_or_none(ssim_values),
        "mean_mae": mean_or_none(mae_values),
        "max_mae": max_or_none(mae_values),
        "mean_fdk_psnr": mean_or_none(fdk_psnr_values),
        "mean_fdk_ssim": mean_or_none(fdk_ssim_values),
        "mean_body_psnr": mean_or_none(body_psnr_values),
        "mean_psnr_margin_vs_fdk": mean_or_none(margins),
        "min_psnr_margin_vs_fdk": min_or_none(margins),
        "sampler": sampler,
        "method_settings": method_settings,
        "paper_experiment": payload.get("paper_experiment", {}),
    }


def command_extra_reconstruction_args(command: list[str]) -> list[str]:
    """Return raw reconstruction args appended after the candidate args."""
    marker = "scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_LIDC_reconstruction.py"
    if marker not in command:
        return []
    # The tuner appends candidate args and then any --reconstruction-arg values.
    # Keep only the explicit raw reconstruction args so summaries do not merge
    # otherwise identical candidates with different initialization/clipping.
    extra_markers = {
        "--initial-reconstruction",
        "--no-clip-initial",
        "--no-clip-output",
        "--clip-initial",
        "--clip-output",
    }
    extras: list[str] = []
    index = 0
    while index < len(command):
        value = command[index]
        if value in extra_markers:
            extras.append(value)
            if value == "--initial-reconstruction" and index + 1 < len(command):
                extras.append(command[index + 1])
                index += 2
                continue
        index += 1
    return extras


def write_jsonl(path: pathlib.Path, records: Iterable[dict]) -> None:
    """Write jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def aggregate_records(records: list[dict]) -> list[dict]:
    """Aggregate records."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if record["status"] not in ("completed", "nonfinite"):
            continue
        key = (record["candidate_group"], record["candidate"])
        groups.setdefault(key, []).append(record)

    rows = []
    for (candidate_group, candidate), group_records in sorted(groups.items()):
        summaries = [record["summary"] for record in group_records]
        psnr_values = [summary.get("mean_psnr") for summary in summaries]
        ssim_values = [summary.get("mean_ssim") for summary in summaries]
        mae_values = [summary.get("mean_mae") for summary in summaries]
        margin_values = [
            summary.get("mean_psnr_margin_vs_fdk") for summary in summaries
        ]
        elapsed_values = [record.get("elapsed_seconds") for record in group_records]
        rows.append(
            {
                "candidate_group": candidate_group,
                "candidate": candidate,
                "completed_jobs": len(group_records),
                "nonfinite_jobs": sum(
                    1
                    for summary in summaries
                    if not summary.get("all_finite_primary_metrics", False)
                ),
                "mean_psnr": mean_or_none(psnr_values),
                "min_job_psnr": min_or_none(psnr_values),
                "mean_ssim": mean_or_none(ssim_values),
                "min_job_ssim": min_or_none(ssim_values),
                "mean_mae": mean_or_none(mae_values),
                "mean_psnr_margin_vs_fdk": mean_or_none(margin_values),
                "min_job_psnr_margin_vs_fdk": min_or_none(margin_values),
                "total_elapsed_seconds": sum(finite_values(elapsed_values)),
                "experiments": ",".join(
                    sorted({record["experiment"] for record in group_records})
                ),
                "models": ",".join(
                    sorted({record["model"] for record in group_records})
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["candidate_group"],
            -float(row["mean_psnr"] if row["mean_psnr"] is not None else -1e9),
            -float(row["mean_ssim"] if row["mean_ssim"] is not None else -1e9),
        )
    )
    return rows


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    """Write csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_group",
        "candidate",
        "completed_jobs",
        "nonfinite_jobs",
        "mean_psnr",
        "min_job_psnr",
        "mean_ssim",
        "min_job_ssim",
        "mean_mae",
        "mean_psnr_margin_vs_fdk",
        "min_job_psnr_margin_vs_fdk",
        "total_elapsed_seconds",
        "experiments",
        "models",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(args: argparse.Namespace, records: list[dict]) -> None:
    """Write outputs."""
    run_root = args.output_root / args.run_name
    write_jsonl(run_root / "runs.jsonl", records)
    summary_rows = aggregate_records(records)
    with open(run_root / "summary.json", "w") as file:
        json.dump(summary_rows, file, indent=2)
    write_csv(run_root / "summary.csv", summary_rows)
