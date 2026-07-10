"""Load and export tuned PaDIS reconstruction defaults."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import json
import math
import pathlib
import sys
from typing import Iterable


SCHEMA_VERSION = 1
CONSENSUS_EXPERIMENT = "consensus"
DEFAULT_CONSENSUS_EXPERIMENTS = ("ct_20", "ct_8")
CONSENSUS_PSNR_TOLERANCE = 0.1
CONSENSUS_SSIM_TOLERANCE = 0.01
DEFAULT_RECONSTRUCTION_HPARAM_DEFAULTS_JSON = (
    pathlib.Path(__file__).resolve().parent
    / "config"
    / "reconstruction_hparam_defaults.json"
)
HIGH_VIEW_FALLBACKS = {
    "ct_60": ("ct_20",),
    "ct_fanbeam_180": ("ct_20",),
    "ct_512_60": ("ct_20",),
}


@dataclass(frozen=True)
class HparamSelection:
    args: tuple[str, ...]
    method: str
    implementation: str
    prior: str
    model: str
    experiment: str
    source_experiment: str
    candidate: str
    candidate_group: str
    run_name: str
    mean_psnr: float
    mean_ssim: float | None
    mean_mae: float | None
    exact_experiment: bool
    exact_model: bool

    def to_json(self) -> dict:
        return {
            "args": list(self.args),
            "method": self.method,
            "implementation": self.implementation,
            "prior": self.prior,
            "model": self.model,
            "experiment": self.experiment,
            "source_experiment": self.source_experiment,
            "candidate": self.candidate,
            "candidate_group": self.candidate_group,
            "run_name": self.run_name,
            "mean_psnr": self.mean_psnr,
            "mean_ssim": self.mean_ssim,
            "mean_mae": self.mean_mae,
            "exact_experiment": self.exact_experiment,
            "exact_model": self.exact_model,
        }


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


def completed_finite_record(record: dict) -> bool:
    if record.get("status") != "completed":
        return False
    summary = record.get("summary") or {}
    sampler = summary.get("sampler") or {}
    if sampler.get("stop_after_outer_steps") is not None:
        return False
    if not summary.get("all_finite_primary_metrics", False):
        return False
    return finite_float(summary.get("mean_psnr")) is not None


def mean_metric(records: Iterable[dict], metric: str) -> float | None:
    values = [
        finite_float((record.get("summary") or {}).get(metric)) for record in records
    ]
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return sum(finite) / len(finite)


def sample_count(record: dict) -> int:
    value = (record.get("summary") or {}).get("sample_count")
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def command_extra_reconstruction_args(command: list[str]) -> list[str]:
    extras: list[str] = []
    index = 0
    while index < len(command):
        value = command[index]
        if value in {
            "--clip-initial",
            "--no-clip-initial",
            "--clip-output",
            "--no-clip-output",
            "--initial-fdk-padded",
            "--no-initial-fdk-padded",
        }:
            extras.append(value)
        elif value in {
            "--initial-reconstruction",
            "--initial-fdk-filter-type",
            "--initial-fdk-frequency-scaling",
            "--initial-fdk-batch-size",
        }:
            extras.append(value)
            if index + 1 < len(command):
                extras.append(command[index + 1])
                index += 1
        index += 1
    return extras


def candidate_args(record: dict) -> tuple[str, ...]:
    args = list(record.get("candidate_args") or ())
    for extra_arg in command_extra_reconstruction_args(record.get("command") or []):
        if extra_arg not in args:
            args.append(extra_arg)
    return tuple(str(item) for item in args)


def record_identity(record: dict) -> tuple:
    return (
        record.get("method"),
        record.get("implementation"),
        record.get("prior"),
        record.get("model"),
        record.get("experiment"),
        record.get("matrix_group", "main"),
        record.get("candidate_group"),
        record.get("candidate"),
        candidate_args(record),
    )


def record_rank(record: dict) -> tuple:
    summary = record.get("summary") or {}
    return (
        finite_float(summary.get("mean_psnr")) or -1.0e18,
        finite_float(summary.get("mean_ssim")) or -1.0e18,
        -(finite_float(summary.get("mean_mae")) or 1.0e18),
        int(record.get("_mtime_ns", 0)),
        int(record.get("_line_index", 0)),
    )


def parse_globs(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or items == ("all",):
        return ("*",)
    return items


def load_records(run_root: pathlib.Path, run_glob: str = "fixedval_*") -> list[dict]:
    run_root = pathlib.Path(run_root).expanduser()
    if not run_root.is_dir():
        return []
    patterns = parse_globs(run_glob)
    records: list[dict] = []
    paths = sorted(
        run_root.glob("*/runs.jsonl"),
        key=lambda item: (item.stat().st_mtime_ns, str(item)),
    )
    for path in paths:
        run_name = path.parent.name
        if not any(fnmatch.fnmatch(run_name, pattern) for pattern in patterns):
            continue
        mtime_ns = path.stat().st_mtime_ns
        with open(path) as file:
            for line_index, line in enumerate(file):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_run_name"] = run_name
                record["_mtime_ns"] = mtime_ns
                record["_line_index"] = line_index
                records.append(record)
    return records


def json_entry_record(entry: dict, *, line_index: int, source_name: str) -> dict:
    summary = dict(entry.get("summary") or {})
    for key in ("mean_psnr", "mean_ssim", "mean_mae"):
        if key in entry:
            summary[key] = entry[key]
    summary.setdefault(
        "all_finite_primary_metrics",
        finite_float(summary.get("mean_psnr")) is not None,
    )
    required = ("method", "implementation", "prior", "model", "experiment", "args")
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(
            f"Invalid hparam defaults entry {line_index}: missing "
            f"{', '.join(missing)}."
        )
    return {
        "candidate_group": str(entry.get("candidate_group") or ""),
        "candidate": str(entry.get("candidate") or ""),
        "candidate_args": [str(item) for item in entry.get("args") or ()],
        "method": str(entry["method"]),
        "implementation": str(entry["implementation"]),
        "prior": str(entry["prior"]),
        "model": str(entry["model"]),
        "experiment": str(entry["experiment"]),
        "matrix_group": str(entry.get("matrix_group", "main")),
        "status": "completed",
        "summary": summary,
        "command": [str(item) for item in entry.get("command") or ()],
        "_run_name": str(entry.get("run_name") or source_name),
        "_mtime_ns": int(entry.get("_mtime_ns", 0)),
        "_line_index": line_index,
    }


def load_json_records(path: pathlib.Path) -> list[dict]:
    path = pathlib.Path(path).expanduser()
    with open(path) as file:
        payload = json.load(file)
    if isinstance(payload, list):
        entries = payload
        source_name = path.name
    elif isinstance(payload, dict):
        schema_version = int(payload.get("schema_version", 0))
        if schema_version and schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported hparam defaults schema_version {schema_version}; "
                f"expected {SCHEMA_VERSION}."
            )
        entries = payload.get("defaults")
        source_name = str(payload.get("source_name") or path.name)
    else:
        raise ValueError(f"Invalid hparam defaults JSON payload in {path}.")
    if not isinstance(entries, list):
        raise ValueError(f"Hparam defaults JSON must contain a defaults list: {path}.")
    mtime_ns = path.stat().st_mtime_ns
    records = []
    for line_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid hparam defaults entry {line_index}: expected object."
            )
        record = json_entry_record(
            entry,
            line_index=line_index,
            source_name=source_name,
        )
        record["_mtime_ns"] = mtime_ns
        records.append(record)
    return records


def dedupe_records(records: Iterable[dict]) -> list[dict]:
    by_key: dict[tuple, dict] = {}
    for record in records:
        if not completed_finite_record(record):
            continue
        if str(record.get("matrix_group", "main")) != "main":
            continue
        key = record_identity(record)
        previous = by_key.get(key)
        if previous is None or record_rank(record) >= record_rank(previous):
            by_key[key] = record
    return list(by_key.values())


def default_record_key(record: dict) -> tuple[str, str, str, str, str]:
    return (
        str(record.get("method") or ""),
        str(record.get("implementation") or ""),
        str(record.get("prior") or ""),
        str(record.get("model") or ""),
        str(record.get("experiment") or ""),
    )


def selected_default_records(records: Iterable[dict]) -> list[dict]:
    by_key: dict[tuple[str, str, str, str, str], dict] = {}
    for record in dedupe_records(records):
        key = default_record_key(record)
        previous = by_key.get(key)
        if previous is None or record_rank(record) >= record_rank(previous):
            by_key[key] = record
    return [by_key[key] for key in sorted(by_key)]


def consensus_candidate_key(record: dict) -> tuple:
    return (
        record.get("method"),
        record.get("implementation"),
        record.get("prior"),
        record.get("model"),
        record.get("candidate_group"),
        record.get("candidate"),
        candidate_args(record),
    )


def consensus_target_key(record: dict) -> tuple[str, str, str, str]:
    return (
        str(record.get("method") or ""),
        str(record.get("implementation") or ""),
        str(record.get("prior") or ""),
        str(record.get("model") or ""),
    )


def aggregate_consensus_record(
    group: list[dict],
    *,
    expected_experiments: tuple[str, ...],
) -> dict:
    first = group[0]
    experiments = sorted({str(record.get("experiment") or "") for record in group})
    expected = set(expected_experiments)
    covered = len(expected.intersection(experiments)) if expected else len(experiments)
    expected_count = len(expected) if expected else 0
    summary = {
        "all_finite_primary_metrics": True,
        "mean_psnr": mean_metric(group, "mean_psnr"),
        "mean_ssim": mean_metric(group, "mean_ssim"),
        "mean_mae": mean_metric(group, "mean_mae"),
        "sample_count": sum(sample_count(record) for record in group),
        "covered_expected_experiments": covered,
        "expected_experiments": expected_count,
        "source_experiments": experiments,
    }
    return {
        "candidate_group": str(first.get("candidate_group") or ""),
        "candidate": str(first.get("candidate") or ""),
        "candidate_args": list(candidate_args(first)),
        "method": str(first.get("method") or ""),
        "implementation": str(first.get("implementation") or ""),
        "prior": str(first.get("prior") or ""),
        "model": str(first.get("model") or ""),
        "experiment": CONSENSUS_EXPERIMENT,
        "matrix_group": str(first.get("matrix_group", "main")),
        "status": "completed",
        "summary": summary,
        "command": [],
        "_run_name": ",".join(
            sorted({str(record.get("_run_name") or "") for record in group})
        ),
        "_mtime_ns": max(int(record.get("_mtime_ns", 0)) for record in group),
        "_line_index": max(int(record.get("_line_index", 0)) for record in group),
    }


def consensus_rank(record: dict) -> tuple:
    summary = record.get("summary") or {}
    covered = int(summary.get("covered_expected_experiments") or 0)
    expected = int(summary.get("expected_experiments") or 0)
    full_coverage = bool(expected and covered == expected)
    psnr = finite_float(summary.get("mean_psnr")) or -1.0e18
    psnr_bin = round(psnr / CONSENSUS_PSNR_TOLERANCE)
    ssim = finite_float(summary.get("mean_ssim")) or -1.0e18
    ssim_bin = round(ssim / CONSENSUS_SSIM_TOLERANCE)
    is_current_default = str(record.get("candidate") or "") == "current_defaults"
    return (
        1 if full_coverage else 0,
        covered,
        psnr_bin,
        ssim_bin,
        -(finite_float(summary.get("mean_mae")) or 1.0e18),
        psnr,
        ssim,
        1 if is_current_default else 0,
        -len(candidate_args(record)),
        int(record.get("_mtime_ns", 0)),
        int(record.get("_line_index", 0)),
    )


def selected_consensus_records(
    records: Iterable[dict],
    *,
    expected_experiments: tuple[str, ...] = DEFAULT_CONSENSUS_EXPERIMENTS,
) -> list[dict]:
    expected = set(expected_experiments)
    grouped: dict[tuple, list[dict]] = {}
    for record in dedupe_records(records):
        experiment = str(record.get("experiment") or "")
        if expected and experiment not in expected:
            continue
        grouped.setdefault(consensus_candidate_key(record), []).append(record)

    candidates = [
        aggregate_consensus_record(group, expected_experiments=expected_experiments)
        for group in grouped.values()
    ]
    by_target: dict[tuple[str, str, str, str], dict] = {}
    for record in candidates:
        key = consensus_target_key(record)
        previous = by_target.get(key)
        if previous is None or consensus_rank(record) >= consensus_rank(previous):
            by_target[key] = record
    supplemental_exact_records = []
    for record in selected_default_records(records):
        key = consensus_target_key(record)
        if key not in by_target:
            supplemental_exact_records.append(record)
    return [
        *[by_target[key] for key in sorted(by_target)],
        *sorted(
            supplemental_exact_records,
            key=lambda record: (
                consensus_target_key(record),
                str(record.get("experiment") or ""),
            ),
        ),
    ]


def record_json_entry(record: dict) -> dict:
    summary = record.get("summary") or {}
    entry = {
        "method": str(record.get("method") or ""),
        "implementation": str(record.get("implementation") or ""),
        "prior": str(record.get("prior") or ""),
        "model": str(record.get("model") or ""),
        "experiment": str(record.get("experiment") or ""),
        "args": list(candidate_args(record)),
        "candidate": str(record.get("candidate") or ""),
        "candidate_group": str(record.get("candidate_group") or ""),
        "run_name": str(record.get("_run_name") or ""),
        "mean_psnr": finite_float(summary.get("mean_psnr")),
        "mean_ssim": finite_float(summary.get("mean_ssim")),
        "mean_mae": finite_float(summary.get("mean_mae")),
    }
    if str(record.get("experiment") or "") == CONSENSUS_EXPERIMENT:
        entry["source_experiments"] = list(summary.get("source_experiments") or ())
        entry["covered_expected_experiments"] = int(
            summary.get("covered_expected_experiments") or 0
        )
        entry["expected_experiments"] = int(summary.get("expected_experiments") or 0)
    return entry


def build_defaults_payload(
    run_root: pathlib.Path,
    *,
    run_glob: str = "fixedval_*",
    selection_scope: str = "per_experiment",
    expected_experiments: tuple[str, ...] = DEFAULT_CONSENSUS_EXPERIMENTS,
) -> dict:
    run_root = pathlib.Path(run_root).expanduser()
    loaded_records = load_records(run_root, run_glob)
    if selection_scope == "per_experiment":
        records = selected_default_records(loaded_records)
        description = (
            "Selected PaDIS reconstruction hyperparameter defaults generated "
            "from fixed-validation tuning runs. For ct_60 and ct_512_60, the "
            "reconstruction matrix falls back to the tuned ct_20 defaults when "
            "an exact experiment default is absent."
        )
    elif selection_scope == "consensus":
        records = selected_consensus_records(
            loaded_records,
            expected_experiments=expected_experiments,
        )
        description = (
            "Selected PaDIS reconstruction hyperparameter defaults generated "
            "as consensus settings across fixed-validation experiments. The "
            "same selected args are used across the reconstruction experiment "
            "grid for each method/implementation/prior/model target. Exact "
            "experiment records are also retained for targets, such as native "
            "512 models, that have no ct_20/ct_8 consensus record."
        )
    else:
        raise ValueError(
            f"Unknown selection scope {selection_scope!r}; expected "
            "'per_experiment' or 'consensus'."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "description": description,
        "generated_from_run_root": str(run_root),
        "run_glob": run_glob,
        "selection_metric": "mean_psnr",
        "selection_scope": selection_scope,
        "consensus_experiment": CONSENSUS_EXPERIMENT,
        "expected_consensus_experiments": list(expected_experiments),
        "high_view_fallbacks": {
            key: list(value) for key, value in HIGH_VIEW_FALLBACKS.items()
        },
        "defaults": [record_json_entry(record) for record in records],
    }


class HparamDefaults:
    def __init__(self, records: Iterable[dict]):
        self.records = tuple(dedupe_records(records))

    @classmethod
    def from_run_root(
        cls, run_root: pathlib.Path | None, run_glob: str = "fixedval_*"
    ) -> "HparamDefaults":
        if run_root is None:
            return cls(())
        return cls(load_records(run_root, run_glob))

    @classmethod
    def from_json(cls, path: pathlib.Path) -> "HparamDefaults":
        return cls(load_json_records(path))

    def select(
        self,
        *,
        method: str,
        implementation: str,
        prior: str,
        model: str,
        experiment: str,
    ) -> HparamSelection | None:
        source_experiments = (
            experiment,
            *HIGH_VIEW_FALLBACKS.get(experiment, ()),
            CONSENSUS_EXPERIMENT,
        )
        model_preferences = [model]
        canonical_model = {
            "patch": "patch_lidc_default",
            "whole_image": "whole_lidc_default",
        }.get(prior)
        if canonical_model is not None and canonical_model != model:
            model_preferences.append(canonical_model)

        # Model identity takes precedence over experiment specificity. This
        # prevents an exact-experiment record for a full-data model from
        # leaking into default-data and trained-ablation model rows.
        for preferred_model in model_preferences:
            for source_experiment in source_experiments:
                selection = self._select_for_experiment(
                    method=method,
                    implementation=implementation,
                    prior=prior,
                    model=model,
                    experiment=experiment,
                    source_experiment=source_experiment,
                    required_model=preferred_model,
                )
                if selection is not None:
                    return selection
        return None

    def _select_for_experiment(
        self,
        *,
        method: str,
        implementation: str,
        prior: str,
        model: str,
        experiment: str,
        source_experiment: str,
        required_model: str,
    ) -> HparamSelection | None:
        matches = [
            record
            for record in self.records
            if record.get("method") == method
            and record.get("implementation") == implementation
            and record.get("prior") == prior
            and record.get("experiment") == source_experiment
            and record.get("model") == required_model
        ]
        if not matches:
            return None
        record = max(matches, key=record_rank)
        summary = record.get("summary") or {}
        return HparamSelection(
            args=candidate_args(record),
            method=method,
            implementation=implementation,
            prior=prior,
            model=str(record.get("model") or ""),
            experiment=experiment,
            source_experiment=source_experiment,
            candidate=str(record.get("candidate") or ""),
            candidate_group=str(record.get("candidate_group") or ""),
            run_name=str(record.get("_run_name") or ""),
            mean_psnr=float(summary.get("mean_psnr")),
            mean_ssim=finite_float(summary.get("mean_ssim")),
            mean_mae=finite_float(summary.get("mean_mae")),
            exact_experiment=source_experiment == experiment,
            exact_model=record.get("model") == model,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=pathlib.Path,
        required=True,
        help="Folder containing hparam sweep subfolders with runs.jsonl files.",
    )
    parser.add_argument(
        "--run-glob",
        default="fixedval_*",
        help=(
            "Comma-separated glob(s) for sweep run folder names. Default "
            "fixedval_* uses corrected fixed-validation tuning runs."
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_RECONSTRUCTION_HPARAM_DEFAULTS_JSON,
        help=(
            "JSON path to write. Defaults to the repo-local reconstruction "
            "defaults file."
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing --output.",
    )
    parser.add_argument(
        "--selection-scope",
        choices=("per_experiment", "consensus"),
        default="per_experiment",
        help=(
            "per_experiment selects the best exact experiment record. consensus "
            "selects one robust candidate per method/implementation/prior/model "
            "using --expected-experiments and writes it as a consensus fallback."
        ),
    )
    parser.add_argument(
        "--expected-experiments",
        default=",".join(DEFAULT_CONSENSUS_EXPERIMENTS),
        help=(
            "Comma-separated validation experiments used by --selection-scope "
            "consensus. Default: ct_20,ct_8."
        ),
    )
    parser.add_argument(
        "--require-records",
        action="store_true",
        help="Fail if no completed finite tuning records are selected.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    expected_experiments = tuple(
        item.strip() for item in args.expected_experiments.split(",") if item.strip()
    )
    payload = build_defaults_payload(
        args.run_root,
        run_glob=args.run_glob,
        selection_scope=args.selection_scope,
        expected_experiments=expected_experiments,
    )
    if args.require_records and not payload["defaults"]:
        raise ValueError(
            f"No completed finite hparam defaults found under {args.run_root} "
            f"with run glob {args.run_glob!r}."
        )
    text = json.dumps(payload, indent=2) + "\n"
    if args.stdout:
        print(text, end="")
        return
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(
        f"Wrote {len(payload['defaults'])} hparam default(s) to {output}",
        file=sys.stderr,
    )


def _value(args: tuple[str, ...], index: int, flag: str) -> tuple[str, int]:
    if index + 1 >= len(args):
        raise ValueError(f"{flag} requires a value.")
    return args[index + 1], index + 2


def apply_reconstruction_args_to_settings(
    *,
    reconstruction_args: Iterable[str],
    sampler_settings: dict | None = None,
    method_settings: dict | None = None,
) -> tuple[dict, dict]:
    sampler = dict(sampler_settings or {})
    method = dict(method_settings or {})
    args = tuple(str(item) for item in reconstruction_args)
    index = 0
    while index < len(args):
        flag = args[index]
        if flag in {
            "--num-steps",
            "--inner-steps",
            "--patch-size",
            "--pad-width",
            "--patch-batch-size",
            "--patch-overlap",
            "--initial-fdk-batch-size",
            "--stop-after-outer-steps",
            "--trace-interval",
        }:
            value, index = _value(args, index, flag)
            sampler[flag[2:].replace("-", "_")] = int(value)
            continue
        if flag in {
            "--sigma-min",
            "--sigma-max",
            "--rho",
            "--zeta",
            "--dps-epsilon",
            "--sampling-epsilon",
            "--initial-fdk-frequency-scaling",
            "--langevin-noise-scale",
            "--pc-snr",
            "--data-consistency-scale",
            "--adjoint-data-consistency-scale",
            "--data-consistency-scale-floor",
            "--data-consistency-scale-power",
            "--operator-norm",
        }:
            value, index = _value(args, index, flag)
            sampler[flag[2:].replace("-", "_")] = float(value)
            continue
        if flag in {
            "--noise-schedule",
            "--data-consistency-gradient",
            "--adjoint-data-step-schedule",
            "--patch-assembly",
            "--fixed-overlap-layout",
            "--pc-corrector-step-rule",
            "--pc-corrector-denoise-sigma",
            "--data-consistency-normalization",
            "--data-consistency-scale-schedule",
        }:
            value, index = _value(args, index, flag)
            sampler[flag[2:].replace("-", "_")] = value
            continue
        if flag == "--initial-reconstruction":
            value, index = _value(args, index, flag)
            sampler["initial_reconstruction"] = value
            if value == "noise":
                sampler["initial_fdk_filter_type"] = None
                sampler["initial_fdk_frequency_scaling"] = 1.0
                sampler["initial_fdk_padded"] = True
            continue
        if flag == "--initial-fdk-filter-type":
            value, index = _value(args, index, flag)
            sampler["initial_fdk_filter_type"] = None if value == "none" else value
            continue
        if flag == "--ve-ddnm-nfe-layout":
            value, index = _value(args, index, flag)
            sampler["ve_ddnm_nfe_layout"] = value
            if value == "paper_1000x1":
                sampler["num_steps"] = 1000
                sampler["inner_steps"] = 1
            elif value == "public_inner":
                sampler["num_steps"] = 100
                sampler["inner_steps"] = 10
            continue
        if flag in {
            "--clip-initial",
            "--clip-output",
            "--clip-denoised",
            "--clip-state",
            "--disable-data-consistency",
            "--disable-langevin-noise",
            "--disable-prior-score",
            "--initial-fdk-padded",
            "--patch-checkpoint-denoiser",
            "--fixed-overlap-checkpoint-denoiser",
            "--langevin-ddnm",
            "--ddnm-pseudoinverse-clip",
            "--ddnm-projected-pseudoinverse-clip",
            "--ddnm-corrected-clip",
            "--pc-reuse-predictor-layout",
            "--trace-images",
        }:
            sampler[flag[2:].replace("-", "_")] = True
            index += 1
            continue
        if flag in {
            "--no-clip-initial",
            "--no-clip-output",
            "--no-initial-fdk-padded",
            "--no-patch-checkpoint-denoiser",
            "--no-fixed-overlap-checkpoint-denoiser",
            "--no-ddnm-pseudoinverse-clip",
            "--no-ddnm-projected-pseudoinverse-clip",
            "--no-ddnm-corrected-clip",
            "--no-pc-reuse-predictor-layout",
        }:
            key = flag[5:].replace("-", "_")
            sampler[key] = False
            index += 1
            continue
        if flag == "--tv-lambda":
            value, index = _value(args, index, flag)
            method["tv_lambda"] = float(value)
            continue
        if flag == "--tv-iterations":
            value, index = _value(args, index, flag)
            method["tv_iterations"] = int(value)
            continue
        if flag == "--tv-lipschitz":
            value, index = _value(args, index, flag)
            method["tv_lipschitz"] = float(value)
            continue
        if flag == "--tv-non-negativity":
            method["tv_non_negativity"] = True
            index += 1
            continue
        if flag in {"--pnp-iterations", "--pnp-cg-iterations"}:
            value, index = _value(args, index, flag)
            method[flag[2:].replace("-", "_")] = int(value)
            continue
        if flag in {"--pnp-eta", "--pnp-cg-tolerance", "--pnp-noise-level"}:
            value, index = _value(args, index, flag)
            method[flag[2:].replace("-", "_")] = float(value)
            continue
        if flag == "--no-pnp-clip":
            method["pnp_clip"] = False
            index += 1
            continue
        index += 1
    return sampler, method


if __name__ == "__main__":
    main()
