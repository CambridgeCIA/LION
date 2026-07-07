"""Summarize PaDIS hyperparameter tuning runs.

The tuner writes one ``runs.jsonl`` file per staged run. This helper combines
those files and ranks candidate settings separately for each
method/implementation/prior/model target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
from typing import Iterable


DEFAULT_TUNING_RUN_ROOT = pathlib.Path(
    "/home/thomas/DiS/Project/Data/experiments/PaDIS/hparam_tuning/runs"
)
REFERENCE_PSNR_TOLERANCE = 0.05
REFERENCE_SSIM_TOLERANCE = 0.002
REFERENCE_MAE_TOLERANCE = 0.0005
RANK_PSNR_TOLERANCE = 0.1
RANK_SSIM_TOLERANCE = 0.01


def parse_csv(value: str) -> tuple[str, ...] | None:
    value = value.strip()
    if value in ("", "all"):
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def mean_or_none(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return statistics.mean(finite)


def min_or_none(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return min(finite)


def load_records(
    run_root: pathlib.Path, run_names: tuple[str, ...] | None
) -> list[dict]:
    records: list[dict] = []
    paths = sorted(
        run_root.glob("*/runs.jsonl"),
        key=lambda item: (item.stat().st_mtime_ns, str(item)),
    )
    for path in paths:
        run_name = path.parent.name
        if run_names is not None and run_name not in run_names:
            continue
        run_mtime_ns = path.stat().st_mtime_ns
        with open(path) as file:
            for line_index, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["_run_name"] = run_name
                record["_run_mtime_ns"] = run_mtime_ns
                record["_line_index"] = line_index
                records.append(record)
    return records


def candidate_identity(record: dict) -> tuple:
    candidate_args = list(record.get("candidate_args") or ())
    for extra_arg in command_extra_reconstruction_args(record.get("command") or []):
        if extra_arg not in candidate_args:
            candidate_args.append(extra_arg)
    return (
        record.get("method"),
        record.get("implementation"),
        record.get("prior"),
        record.get("model"),
        record.get("candidate_group"),
        record.get("candidate"),
        tuple(candidate_args),
    )


def command_extra_reconstruction_args(command: list[str]) -> list[str]:
    extras: list[str] = []
    index = 0
    while index < len(command):
        value = command[index]
        if value in {
            "--no-clip-initial",
            "--no-clip-output",
            "--clip-initial",
            "--clip-output",
        }:
            extras.append(value)
        elif value == "--initial-reconstruction":
            extras.append(value)
            if index + 1 < len(command):
                extras.append(command[index + 1])
                index += 1
        index += 1
    return extras


def target_identity(record: dict) -> tuple:
    return (
        record.get("method"),
        record.get("implementation"),
        record.get("prior"),
        record.get("model"),
    )


def record_metric(record: dict, name: str) -> float | None:
    return finite_float((record.get("summary") or {}).get(name))


def record_sample_count(record: dict) -> int:
    value = (record.get("summary") or {}).get("sample_count")
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return max(count, 0)


def usable_record(record: dict) -> bool:
    if record.get("status") != "completed":
        return False
    summary = record.get("summary") or {}
    sampler = summary.get("sampler") or {}
    if sampler.get("stop_after_outer_steps") is not None:
        return False
    if not summary.get("all_finite_primary_metrics", False):
        return False
    return record_metric(record, "mean_psnr") is not None


def dedupe_records(records: Iterable[dict]) -> list[dict]:
    """Keep the newest completed record for the same candidate/run target."""
    by_key: dict[tuple, dict] = {}
    for index, record in enumerate(records):
        if not usable_record(record):
            continue
        key = (
            candidate_identity(record),
            record.get("experiment"),
            record.get("matrix_group"),
        )
        record["_order"] = index
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = record
            continue
        record_order = (
            int(record.get("_run_mtime_ns", 0)),
            int(record.get("_line_index", 0)),
            int(record.get("_order", 0)),
        )
        previous_order = (
            int(previous.get("_run_mtime_ns", 0)),
            int(previous.get("_line_index", 0)),
            int(previous.get("_order", 0)),
        )
        if record_order >= previous_order:
            by_key[key] = record
    return list(by_key.values())


def aggregate(
    records: list[dict], expected_experiments: tuple[str, ...] | None
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        grouped.setdefault(candidate_identity(record), []).append(record)

    rows: list[dict] = []
    expected = set(expected_experiments or ())
    for identity, group_records in grouped.items():
        (
            method,
            implementation,
            prior,
            model,
            candidate_group,
            candidate,
            candidate_args,
        ) = identity
        psnr_values = [record_metric(record, "mean_psnr") for record in group_records]
        ssim_values = [record_metric(record, "mean_ssim") for record in group_records]
        mae_values = [record_metric(record, "mean_mae") for record in group_records]
        margin_values = [
            record_metric(record, "mean_psnr_margin_vs_fdk") for record in group_records
        ]
        sample_count = sum(record_sample_count(record) for record in group_records)
        experiments = sorted(
            {str(record.get("experiment")) for record in group_records}
        )
        covered_expected = len(expected.intersection(experiments)) if expected else None
        expected_count = len(expected) if expected else None
        mean_psnr = mean_or_none(psnr_values)
        mean_ssim = mean_or_none(ssim_values)
        mean_mae = mean_or_none(mae_values)
        rows.append(
            {
                "method": method,
                "implementation": implementation,
                "prior": prior,
                "model": model,
                "candidate_group": candidate_group,
                "candidate": candidate,
                "candidate_args": " ".join(str(arg) for arg in candidate_args),
                "completed_jobs": len(group_records),
                "completed_samples": sample_count,
                "experiments": ",".join(experiments),
                "covered_expected_experiments": covered_expected,
                "expected_experiments": expected_count,
                "mean_psnr": mean_psnr,
                "min_psnr": min_or_none(psnr_values),
                "mean_ssim": mean_ssim,
                "min_ssim": min_or_none(ssim_values),
                "mean_mae": mean_mae,
                "mean_psnr_margin_vs_fdk": mean_or_none(margin_values),
                "min_psnr_margin_vs_fdk": min_or_none(margin_values),
                "run_names": ",".join(
                    sorted({record["_run_name"] for record in group_records})
                ),
            }
        )
    return rows


def row_float(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def reference_key(row: dict, implementation: str) -> tuple | None:
    if row.get("implementation") != implementation:
        return None
    return (row.get("method"), row.get("prior"), row.get("model"))


def reference_sort_key(row: dict) -> tuple:
    coverage = row.get("covered_expected_experiments")
    if coverage is None:
        coverage = row.get("completed_jobs") or 0
    expected = row.get("expected_experiments")
    full_coverage = expected is not None and coverage == expected
    psnr = row_float(row, "mean_psnr")
    ssim = row_float(row, "mean_ssim")
    mae = row_float(row, "mean_mae")
    is_current_default = str(row.get("candidate") or "") == "current_defaults"
    return (
        1 if full_coverage else 0,
        int(coverage),
        round((psnr or -1e9) / RANK_PSNR_TOLERANCE),
        round((ssim or -1e9) / RANK_SSIM_TOLERANCE),
        -(mae if mae is not None else 1e9),
        psnr or -1e9,
        ssim or -1e9,
        1 if is_current_default else 0,
        -len(str(row.get("candidate_args") or "").split()),
    )


def add_reference_columns(
    rows: list[dict],
    *,
    reference_implementation: str,
    compare_implementation: str,
) -> None:
    """Annotate rows with deltas against a matching reference row.

    Prefer references evaluated on the same experiment set and number of
    completed samples. Falling back to the best available reference is useful
    for sparse early sweeps, but same-sample references are the fair comparison
    once they exist.
    """
    references: dict[tuple, list[dict]] = {}
    for row in rows:
        key = reference_key(row, reference_implementation)
        if key is None:
            continue
        references.setdefault(key, []).append(row)

    def select_reference(candidates: list[dict], target: dict) -> dict:
        experiment_matched = [
            candidate
            for candidate in candidates
            if candidate.get("experiments") == target.get("experiments")
        ]
        narrowed = experiment_matched or candidates
        sample_matched = [
            candidate
            for candidate in narrowed
            if candidate.get("completed_samples") == target.get("completed_samples")
        ]
        narrowed = sample_matched or narrowed
        return max(narrowed, key=reference_sort_key)

    for row in rows:
        row["reference_implementation"] = ""
        row["reference_candidate"] = ""
        row["reference_candidate_args"] = ""
        row["reference_mean_psnr"] = None
        row["reference_mean_ssim"] = None
        row["reference_mean_mae"] = None
        row["delta_psnr_vs_reference"] = None
        row["delta_ssim_vs_reference"] = None
        row["delta_mae_vs_reference"] = None
        row["reference_status"] = ""
        if row.get("implementation") != compare_implementation:
            continue
        candidates = references.get(
            (row.get("method"), row.get("prior"), row.get("model"))
        )
        if not candidates:
            continue
        reference = select_reference(candidates, row)
        row["reference_implementation"] = reference_implementation
        row["reference_candidate"] = reference.get("candidate", "")
        row["reference_candidate_args"] = reference.get("candidate_args", "")
        for metric in ("mean_psnr", "mean_ssim", "mean_mae"):
            row[f"reference_{metric}"] = reference.get(metric)
        psnr = row_float(row, "mean_psnr")
        ssim = row_float(row, "mean_ssim")
        mae = row_float(row, "mean_mae")
        reference_psnr = row_float(reference, "mean_psnr")
        reference_ssim = row_float(reference, "mean_ssim")
        reference_mae = row_float(reference, "mean_mae")
        if psnr is not None and reference_psnr is not None:
            row["delta_psnr_vs_reference"] = psnr - reference_psnr
        if ssim is not None and reference_ssim is not None:
            row["delta_ssim_vs_reference"] = ssim - reference_ssim
        if mae is not None and reference_mae is not None:
            row["delta_mae_vs_reference"] = mae - reference_mae
        psnr_delta = row["delta_psnr_vs_reference"]
        ssim_delta = row["delta_ssim_vs_reference"]
        mae_delta = row["delta_mae_vs_reference"]
        if psnr_delta is None or ssim_delta is None or mae_delta is None:
            continue
        if psnr_delta >= 0.0 and ssim_delta >= 0.0 and mae_delta <= 0.0:
            row["reference_status"] = "beats_reference"
        elif (
            psnr_delta >= -REFERENCE_PSNR_TOLERANCE
            and ssim_delta >= -REFERENCE_SSIM_TOLERANCE
            and mae_delta <= REFERENCE_MAE_TOLERANCE
        ):
            row["reference_status"] = "matches_reference_tolerance"
        else:
            row["reference_status"] = "below_reference"


def sort_rows(rows: list[dict]) -> list[dict]:
    def key(row: dict) -> tuple:
        coverage = row["covered_expected_experiments"]
        if coverage is None:
            coverage = row["completed_jobs"]
        psnr = row_float(row, "mean_psnr")
        ssim = row_float(row, "mean_ssim")
        mae = row_float(row, "mean_mae")
        is_current_default = str(row.get("candidate") or "") == "current_defaults"
        return (
            str(row["method"]),
            str(row["implementation"]),
            str(row["prior"]),
            str(row["model"]),
            -int(coverage),
            -round((psnr or -1e9) / RANK_PSNR_TOLERANCE),
            -round((ssim or -1e9) / RANK_SSIM_TOLERANCE),
            float(mae if mae is not None else 1e9),
            -float(psnr if psnr is not None else -1e9),
            -float(ssim if ssim is not None else -1e9),
            0 if is_current_default else 1,
            len(str(row.get("candidate_args") or "").split()),
        )

    return sorted(rows, key=key)


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "implementation",
        "prior",
        "model",
        "candidate_group",
        "candidate",
        "candidate_args",
        "completed_jobs",
        "completed_samples",
        "experiments",
        "covered_expected_experiments",
        "expected_experiments",
        "mean_psnr",
        "min_psnr",
        "mean_ssim",
        "min_ssim",
        "mean_mae",
        "mean_psnr_margin_vs_fdk",
        "min_psnr_margin_vs_fdk",
        "reference_implementation",
        "reference_candidate",
        "reference_candidate_args",
        "reference_mean_psnr",
        "reference_mean_ssim",
        "reference_mean_mae",
        "delta_psnr_vs_reference",
        "delta_ssim_vs_reference",
        "delta_mae_vs_reference",
        "reference_status",
        "run_names",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_top(rows: list[dict], top_k: int) -> None:
    current_target: tuple | None = None
    emitted = 0
    for row in rows:
        target = (
            row["method"],
            row["implementation"],
            row["prior"],
            row["model"],
        )
        if target != current_target:
            current_target = target
            emitted = 0
            print()
            print(
                f"{row['method']} / {row['implementation']} / "
                f"{row['prior']} / {row['model']}"
            )
        if emitted >= top_k:
            continue
        emitted += 1
        coverage = ""
        if row["expected_experiments"] is not None:
            coverage = (
                f" coverage={row['covered_expected_experiments']}"
                f"/{row['expected_experiments']}"
            )
        print(
            f"  {emitted}. {row['candidate']} "
            f"PSNR={row['mean_psnr']:.3f} SSIM={row['mean_ssim']:.3f} "
            f"MAE={row['mean_mae']:.5f} samples={row['completed_samples']}"
            f"{coverage} "
            f"experiments={row['experiments']} args={row['candidate_args']}"
        )
        if row.get("reference_implementation"):
            print(
                "     vs "
                f"{row['reference_implementation']}:{row['reference_candidate']} "
                f"dPSNR={row['delta_psnr_vs_reference']:.3f} "
                f"dSSIM={row['delta_ssim_vs_reference']:.3f} "
                f"dMAE={row['delta_mae_vs_reference']:.5f} "
                f"status={row['reference_status']}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=pathlib.Path, default=DEFAULT_TUNING_RUN_ROOT
    )
    parser.add_argument(
        "--run-names",
        default="all",
        help="Comma-separated tuning run names to include, or all.",
    )
    parser.add_argument(
        "--expected-experiments",
        default="ct_20,ct_8",
        help="Comma-separated experiments expected in the current validation stage.",
    )
    parser.add_argument(
        "--include-extra-experiments",
        action="store_true",
        help=(
            "When --expected-experiments is set, keep records from other "
            "experiments in the aggregate instead of restricting the report to "
            "the expected validation-stage experiments."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-csv",
        type=pathlib.Path,
        default=None,
        help="Optional CSV output path. Defaults to run-root/hparam_recommendations.csv.",
    )
    parser.add_argument(
        "--reference-implementation",
        default="public_repo",
        help="Implementation used as fixed comparator for matching method/prior/model rows.",
    )
    parser.add_argument(
        "--compare-implementation",
        default="lion_physics",
        help="Implementation whose rows should receive reference delta columns.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_names = parse_csv(args.run_names)
    expected_experiments = parse_csv(args.expected_experiments)
    output_csv = args.output_csv or args.run_root / "hparam_recommendations.csv"

    records = dedupe_records(load_records(args.run_root, run_names))
    if expected_experiments is not None and not args.include_extra_experiments:
        expected_set = set(expected_experiments)
        records = [
            record
            for record in records
            if str(record.get("experiment")) in expected_set
        ]
    rows = aggregate(records, expected_experiments)
    add_reference_columns(
        rows,
        reference_implementation=args.reference_implementation,
        compare_implementation=args.compare_implementation,
    )
    rows = sort_rows(rows)
    write_csv(output_csv, rows)
    print(f"Wrote {len(rows)} candidate summaries to {output_csv}")
    print_top(rows, args.top_k)


if __name__ == "__main__":
    main()
