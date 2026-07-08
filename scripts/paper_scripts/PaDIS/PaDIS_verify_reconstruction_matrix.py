"""Verify PaDIS reconstruction-matrix outputs from metrics.json files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random


BOOTSTRAP_MEAN_METRICS = (
    "psnr",
    "ssim",
    "mae",
    "fdk_psnr",
    "fdk_margin",
    "relative_sinogram_residual",
)

CORE_CSV_FIELDS = (
    "method",
    "algorithm",
    "prior_mode",
    "experiment",
    "implementation",
    "geometry",
    "matrix_group",
    "num_samples",
    "checkpoint",
    "path",
)


def parse_csv(value: str | None) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_method_thresholds(items: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Expected method threshold in METHOD=VALUE form, got {item!r}."
            )
        method, value = item.split("=", 1)
        method = method.strip()
        if not method:
            raise ValueError(f"Empty method name in threshold {item!r}.")
        thresholds[method] = float(value)
    return thresholds


def parse_multi_csv(items: list[str]) -> set[str]:
    values: set[str] = set()
    for item in items:
        values.update(parse_csv(item))
    return values


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def minimum(values: list[float]) -> float | None:
    if not values:
        return None
    return float(min(values))


def maximum(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(values))


def finite_float(value) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def sample_standard_deviation(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    value_mean = float(sum(values) / len(values))
    variance = sum((item - value_mean) ** 2 for item in values) / (len(values) - 1)
    return float(math.sqrt(variance))


def percentile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def bootstrap_mean_summary(
    values: list[float],
    *,
    resamples: int,
    confidence: float,
    rng: random.Random,
) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "sample_sd": None,
            "bootstrap_se": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
        }
    if resamples <= 0:
        return {
            "n": len(values),
            "sample_sd": sample_standard_deviation(values),
            "bootstrap_se": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
        }

    sample_count = len(values)
    if sample_count == 1:
        return {
            "n": 1,
            "sample_sd": 0.0,
            "bootstrap_se": 0.0,
            "bootstrap_ci_low": float(values[0]),
            "bootstrap_ci_high": float(values[0]),
        }

    bootstrap_means = []
    for _ in range(resamples):
        bootstrap_means.append(
            sum(values[rng.randrange(sample_count)] for _ in range(sample_count))
            / sample_count
        )
    bootstrap_means.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": sample_count,
        "sample_sd": sample_standard_deviation(values),
        "bootstrap_se": sample_standard_deviation(bootstrap_means),
        "bootstrap_ci_low": percentile(bootstrap_means, alpha),
        "bootstrap_ci_high": percentile(bootstrap_means, 1.0 - alpha),
    }


def metric_values(metrics: list[dict], key: str) -> list[float]:
    values = []
    for item in metrics:
        value = finite_float(item.get(key))
        if value is not None:
            values.append(value)
    return values


def fdk_margin_values(metrics: list[dict]) -> list[float]:
    values = []
    for item in metrics:
        psnr = finite_float(item.get("psnr"))
        fdk_psnr = finite_float(item.get("fdk_psnr"))
        if psnr is not None and fdk_psnr is not None:
            values.append(psnr - fdk_psnr)
    return values


def bootstrap_values_for_metric(metrics: list[dict], name: str) -> list[float]:
    if name == "fdk_margin":
        return fdk_margin_values(metrics)
    if name == "relative_sinogram_residual":
        return metric_values(metrics, "recon_relative_sinogram_residual")
    return metric_values(metrics, name)


def stable_record_seed(summary: dict, seed: int) -> int:
    material = "|".join(
        [
            str(seed),
            summary.get("method", ""),
            summary.get("algorithm", ""),
            summary.get("prior_mode", ""),
            summary.get("experiment", ""),
            summary.get("implementation", ""),
            summary.get("geometry", ""),
            summary.get("matrix_group", ""),
            summary.get("checkpoint", ""),
            summary.get("path", ""),
        ]
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def add_bootstrap_uncertainty(
    summary: dict,
    metrics: list[dict],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> None:
    summary["bootstrap_resamples"] = int(resamples)
    summary["bootstrap_confidence"] = float(confidence)
    rng = random.Random(stable_record_seed(summary, seed))
    for name in BOOTSTRAP_MEAN_METRICS:
        values = bootstrap_values_for_metric(metrics, name)
        stats = bootstrap_mean_summary(
            values,
            resamples=resamples,
            confidence=confidence,
            rng=rng,
        )
        prefix = f"mean_{name}"
        summary[f"{prefix}_n"] = stats["n"]
        summary[f"{prefix}_sample_sd"] = stats["sample_sd"]
        summary[f"{prefix}_bootstrap_se"] = stats["bootstrap_se"]
        summary[f"{prefix}_bootstrap_ci_low"] = stats["bootstrap_ci_low"]
        summary[f"{prefix}_bootstrap_ci_high"] = stats["bootstrap_ci_high"]


def finite_or_infinite_psnr(key: str, value) -> bool:
    if not isinstance(value, (int, float)):
        return False
    if math.isfinite(float(value)):
        return True
    return key.endswith("psnr") and math.isinf(float(value)) and float(value) > 0


def load_record(path: Path) -> dict:
    with open(path) as f:
        payload = json.load(f)
    metrics = payload.get("metrics", [])
    if not metrics:
        raise ValueError(f"{path} does not contain any metrics.")

    summary = {
        "path": str(path),
        "checkpoint": str(payload.get("checkpoint", "")),
        "method": payload.get("method", "unknown"),
        "algorithm": str(payload.get("algorithm", "")),
        "prior_mode": str(payload.get("prior_mode", "")),
        "experiment": payload.get("experiment", "unknown"),
        "implementation": payload.get("implementation", "unknown"),
        "geometry": payload.get("geometry_tag", "unknown"),
        "matrix_group": payload.get("matrix_group", "main"),
        "num_samples": len(metrics),
        "mean_psnr": mean([item["psnr"] for item in metrics if "psnr" in item]),
        "min_psnr": minimum([item["psnr"] for item in metrics if "psnr" in item]),
        "mean_ssim": mean([item["ssim"] for item in metrics if "ssim" in item]),
        "min_ssim": minimum([item["ssim"] for item in metrics if "ssim" in item]),
        "mean_mae": mean([item["mae"] for item in metrics if "mae" in item]),
        "max_mae": maximum([item["mae"] for item in metrics if "mae" in item]),
        "mean_fdk_psnr": mean(
            [item["fdk_psnr"] for item in metrics if "fdk_psnr" in item]
        ),
        "mean_fdk_margin": mean(
            [
                item["psnr"] - item["fdk_psnr"]
                for item in metrics
                if "psnr" in item and "fdk_psnr" in item
            ]
        ),
        "min_fdk_margin": minimum(
            [
                item["psnr"] - item["fdk_psnr"]
                for item in metrics
                if "psnr" in item and "fdk_psnr" in item
            ]
        ),
        "mean_relative_sinogram_residual": mean(
            [
                item["recon_relative_sinogram_residual"]
                for item in metrics
                if "recon_relative_sinogram_residual" in item
            ]
        ),
    }
    return {"payload": payload, "summary": summary, "metrics": metrics}


def canonical_checkpoint(value) -> str:
    path_text = str(value or "")
    if not path_text:
        return ""
    return str(Path(path_text).expanduser().resolve(strict=False))


def identity_tuple(item: dict) -> tuple[str, str, str, str, str, str, str, str]:
    return (
        str(item.get("method", "")),
        str(item.get("algorithm", "")),
        str(item.get("prior_mode", "")),
        str(item.get("experiment", "")),
        str(item.get("implementation", "")),
        str(item.get("geometry", item.get("geometry_tag", ""))),
        str(item.get("matrix_group", "main")),
        canonical_checkpoint(item.get("checkpoint", "")),
    )


def identity_label(identity: tuple[str, str, str, str, str, str, str, str]) -> str:
    (
        method,
        algorithm,
        prior_mode,
        experiment,
        implementation,
        geometry,
        matrix_group,
        checkpoint,
    ) = identity
    return (
        f"method={method} algorithm={algorithm} prior_mode={prior_mode} "
        f"experiment={experiment} implementation={implementation} "
        f"geometry={geometry} matrix_group={matrix_group} checkpoint={checkpoint}"
    )


def load_expected_jobs(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of reconstruction jobs.")
    return payload


def expected_job_identities(jobs: list[dict]) -> Counter:
    return Counter(identity_tuple(item) for item in jobs)


def expected_samplers_by_identity(jobs: list[dict]) -> dict:
    samplers = {}
    for item in jobs:
        expected_sampler = item.get("expected_sampler")
        if expected_sampler is not None:
            samplers[identity_tuple(item)] = expected_sampler
    return samplers


def expected_method_settings_by_identity(jobs: list[dict]) -> dict:
    settings = {}
    for item in jobs:
        expected_settings = item.get("expected_method_settings")
        if expected_settings is not None:
            settings[identity_tuple(item)] = expected_settings
    return settings


def values_match(expected, actual) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-9)
    return actual == expected


def check_expected_mapping(
    summary: dict,
    payload: dict,
    *,
    payload_key: str,
    expected: dict,
    setting_label: str,
) -> list[str]:
    failures = []
    actual_settings = payload.get(payload_key, {})
    label = (
        f"{summary['method']} {summary['experiment']} "
        f"{summary['implementation']} {summary['geometry']}"
    )
    if not isinstance(actual_settings, dict):
        return [
            f"{label}: metrics payload does not contain a {payload_key} dictionary."
        ]
    for key, expected_value in sorted(expected.items()):
        if key not in actual_settings:
            failures.append(f"{label}: missing {setting_label} setting {key}")
            continue
        actual = actual_settings[key]
        if not values_match(expected_value, actual):
            failures.append(
                f"{label}: {setting_label} {key}={actual!r} "
                f"does not match expected {expected_value!r}"
            )
    return failures


def find_records(args) -> list[dict]:
    records = []
    methods = set(parse_csv(args.methods))
    experiments = set(parse_csv(args.experiments))
    implementations = set(parse_csv(args.implementations))
    geometries = set(parse_csv(args.geometries))

    for path in sorted(args.root.rglob("metrics.json")):
        record = load_record(path)
        summary = record["summary"]
        if methods and summary["method"] not in methods:
            continue
        if experiments and summary["experiment"] not in experiments:
            continue
        if implementations and summary["implementation"] not in implementations:
            continue
        if geometries and summary["geometry"] not in geometries:
            continue
        add_bootstrap_uncertainty(
            summary,
            record["metrics"],
            resamples=args.bootstrap_resamples,
            confidence=args.bootstrap_confidence,
            seed=args.bootstrap_seed,
        )
        records.append(record)
    return records


def output_csv_path(args: argparse.Namespace) -> Path | None:
    if args.output_csv is not None:
        return args.output_csv
    if args.output_json is not None:
        return args.output_json.with_suffix(".csv")
    return None


def write_records_csv(path: Path, records: list[dict]) -> None:
    rows = [record["summary"] for record in records]
    fieldnames = list(CORE_CSV_FIELDS)
    extras = sorted({key for row in rows for key in row} - set(fieldnames))
    fieldnames.extend(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_records(args, records: list[dict]) -> list[str]:
    failures = []
    if not records:
        return [f"No matching metrics.json files found under {args.root}."]
    if args.expected_records is not None and len(records) != args.expected_records:
        failures.append(
            f"Expected {args.expected_records} matching metrics.json files, "
            f"found {len(records)}."
        )
    expected_jobs_payload = load_expected_jobs(args.expected_jobs_json)
    expected_jobs = expected_job_identities(expected_jobs_payload)
    expected_samplers = expected_samplers_by_identity(expected_jobs_payload)
    expected_method_settings = expected_method_settings_by_identity(
        expected_jobs_payload
    )
    if args.expected_jobs_json is not None:
        found_jobs = Counter(identity_tuple(record["summary"]) for record in records)
        missing_jobs = expected_jobs - found_jobs
        unexpected_jobs = found_jobs - expected_jobs
        for identity, count in sorted(missing_jobs.items()):
            failures.append(
                f"Missing expected reconstruction record ({count}): "
                f"{identity_label(identity)}"
            )
        for identity, count in sorted(unexpected_jobs.items()):
            failures.append(
                f"Unexpected reconstruction record ({count}): "
                f"{identity_label(identity)}"
            )
        for record in records:
            identity = identity_tuple(record["summary"])
            if identity in expected_samplers:
                failures.extend(
                    check_expected_mapping(
                        record["summary"],
                        record["payload"],
                        payload_key="sampler",
                        expected=expected_samplers[identity],
                        setting_label="sampler",
                    )
                )
            if identity in expected_method_settings:
                failures.extend(
                    check_expected_mapping(
                        record["summary"],
                        record["payload"],
                        payload_key="method_settings",
                        expected=expected_method_settings[identity],
                        setting_label="method",
                    )
                )
    if args.expected_samples is not None:
        for record in records:
            summary = record["summary"]
            if summary["num_samples"] != args.expected_samples:
                failures.append(
                    f"{summary['method']} {summary['experiment']} "
                    f"{summary['implementation']} {summary['geometry']}: "
                    f"expected {args.expected_samples} samples, "
                    f"found {summary['num_samples']}."
                )

    seen_methods = {record["summary"]["method"] for record in records}
    for method in parse_csv(args.require_methods):
        if method not in seen_methods:
            failures.append(f"Missing required method: {method}")

    seen_experiments = {record["summary"]["experiment"] for record in records}
    for experiment in parse_csv(args.require_experiments):
        if experiment not in seen_experiments:
            failures.append(f"Missing required experiment: {experiment}")

    min_method_psnr = parse_method_thresholds(args.min_method_mean_psnr)
    min_method_ssim = parse_method_thresholds(args.min_method_mean_ssim)
    max_method_mae = parse_method_thresholds(args.max_method_mean_mae)
    require_method_mean_better_than_fdk = parse_multi_csv(
        args.require_method_mean_better_than_fdk
    )
    require_method_each_better_than_fdk = parse_multi_csv(
        args.require_method_each_better_than_fdk
    )

    required_metric_keys = ("mse", "psnr", "mae", "fdk_psnr")
    for record in records:
        summary = record["summary"]
        label = (
            f"{summary['method']} {summary['experiment']} "
            f"{summary['implementation']} {summary['geometry']}"
        )
        for index, item in enumerate(record["metrics"]):
            for key in required_metric_keys:
                if key not in item:
                    failures.append(f"{label} sample {index}: missing metric {key}")
                elif not finite_or_infinite_psnr(key, item[key]):
                    failures.append(
                        f"{label} sample {index}: non-finite metric {key}={item[key]!r}"
                    )

        if args.min_mean_psnr is not None and (
            summary["mean_psnr"] is None or summary["mean_psnr"] < args.min_mean_psnr
        ):
            failures.append(
                f"{label}: mean_psnr={summary['mean_psnr']} < {args.min_mean_psnr}"
            )
        if args.min_mean_ssim is not None and (
            summary["mean_ssim"] is None or summary["mean_ssim"] < args.min_mean_ssim
        ):
            failures.append(
                f"{label}: mean_ssim={summary['mean_ssim']} < {args.min_mean_ssim}"
            )
        if args.max_mean_mae is not None and (
            summary["mean_mae"] is None or summary["mean_mae"] > args.max_mean_mae
        ):
            failures.append(
                f"{label}: mean_mae={summary['mean_mae']} > {args.max_mean_mae}"
            )

        method = summary["method"]
        if method in min_method_psnr and (
            summary["mean_psnr"] is None
            or summary["mean_psnr"] < min_method_psnr[method]
        ):
            failures.append(
                f"{label}: mean_psnr={summary['mean_psnr']} < "
                f"{min_method_psnr[method]}"
            )
        if method in min_method_ssim and (
            summary["mean_ssim"] is None
            or summary["mean_ssim"] < min_method_ssim[method]
        ):
            failures.append(
                f"{label}: mean_ssim={summary['mean_ssim']} < "
                f"{min_method_ssim[method]}"
            )
        if method in max_method_mae and (
            summary["mean_mae"] is None or summary["mean_mae"] > max_method_mae[method]
        ):
            failures.append(
                f"{label}: mean_mae={summary['mean_mae']} > {max_method_mae[method]}"
            )
        if args.require_mean_better_than_fdk or (
            method in require_method_mean_better_than_fdk
        ):
            if summary["mean_psnr"] is None or summary["mean_fdk_psnr"] is None:
                failures.append(f"{label}: missing PSNR or FDK PSNR for FDK gate")
            elif summary["mean_psnr"] <= summary["mean_fdk_psnr"]:
                failures.append(
                    f"{label}: mean_psnr={summary['mean_psnr']} <= "
                    f"mean_fdk_psnr={summary['mean_fdk_psnr']}"
                )
        if args.require_each_better_than_fdk or (
            method in require_method_each_better_than_fdk
        ):
            if summary["min_fdk_margin"] is None:
                failures.append(f"{label}: missing sample PSNR margins for FDK gate")
            elif summary["min_fdk_margin"] <= 0:
                failures.append(
                    f"{label}: at least one sample did not beat FDK "
                    f"(min margin {summary['min_fdk_margin']})"
                )

    return failures


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV summary path. Defaults to the --output-json path with "
            "a .csv suffix when --output-json is set."
        ),
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2000,
        help="Bootstrap resamples per reconstruction record. Use 0 to disable.",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Central bootstrap confidence interval mass.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=33,
        help="Seed used for deterministic bootstrap uncertainty estimates.",
    )
    parser.add_argument("--methods", default=None)
    parser.add_argument("--experiments", default=None)
    parser.add_argument("--implementations", default=None)
    parser.add_argument("--geometries", default=None)
    parser.add_argument("--require-methods", default=None)
    parser.add_argument("--require-experiments", default=None)
    parser.add_argument(
        "--expected-records",
        type=int,
        default=None,
        help="Fail unless exactly this many matching metrics.json files are found.",
    )
    parser.add_argument(
        "--expected-jobs-json",
        type=Path,
        default=None,
        help=(
            "JSON output from PaDIS_run_reconstruction_matrix.py --list. "
            "Fail unless matching metrics exist for every listed job identity."
        ),
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=None,
        help="Fail unless every matching metrics.json has this many samples.",
    )
    parser.add_argument("--min-mean-psnr", type=float, default=None)
    parser.add_argument("--min-mean-ssim", type=float, default=None)
    parser.add_argument("--max-mean-mae", type=float, default=None)
    parser.add_argument(
        "--min-method-mean-psnr",
        action="append",
        default=[],
        help="Method-specific PSNR gate in METHOD=VALUE form. Repeatable.",
    )
    parser.add_argument(
        "--min-method-mean-ssim",
        action="append",
        default=[],
        help="Method-specific SSIM gate in METHOD=VALUE form. Repeatable.",
    )
    parser.add_argument(
        "--max-method-mean-mae",
        action="append",
        default=[],
        help="Method-specific MAE gate in METHOD=VALUE form. Repeatable.",
    )
    parser.add_argument(
        "--require-mean-better-than-fdk",
        action="store_true",
        help="Fail unless every selected record has mean PSNR greater than mean FDK PSNR.",
    )
    parser.add_argument(
        "--require-each-better-than-fdk",
        action="store_true",
        help="Fail unless every sample in every selected record beats its FDK PSNR.",
    )
    parser.add_argument(
        "--require-method-mean-better-than-fdk",
        action="append",
        default=[],
        help=(
            "Method name or comma-separated method list that must beat FDK in "
            "mean PSNR. Repeatable."
        ),
    )
    parser.add_argument(
        "--require-method-each-better-than-fdk",
        action="append",
        default=[],
        help=(
            "Method name or comma-separated method list where every sample must "
            "beat FDK. Repeatable."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.bootstrap_resamples < 0:
        raise ValueError("--bootstrap-resamples must be non-negative.")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("--bootstrap-confidence must be between 0 and 1.")
    records = find_records(args)
    failures = check_records(args, records)
    output = {
        "root": str(args.root),
        "num_records": len(records),
        "records": [record["summary"] for record in records],
        "failures": failures,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
    csv_path = output_csv_path(args)
    if csv_path is not None:
        write_records_csv(csv_path, records)
    print(json.dumps(output, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
