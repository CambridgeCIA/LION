"""Reconcile PaDIS reconstruction runner state after manifest reordering."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
from pathlib import Path
import shutil


MARKER_SUFFIXES = ("done", "failed", "seconds")


def load_manifest(path: Path) -> list[dict]:
    with open(path) as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return payload


def stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def command_option(command: list[str], name: str) -> str | None:
    for index, item in enumerate(command):
        if item == name and index + 1 < len(command):
            return command[index + 1]
    return None


def job_key(job: dict) -> str:
    command = [str(item) for item in job.get("command") or []]
    command_without_python = command[1:] if command else []
    fields = {
        "model": job.get("model"),
        "checkpoint": job.get("checkpoint"),
        "method": job.get("method"),
        "algorithm": job.get("algorithm"),
        "prior_mode": job.get("prior_mode"),
        "implementation": job.get("implementation"),
        "geometry": job.get("geometry"),
        "experiment": job.get("experiment"),
        "matrix_group": job.get("matrix_group", "main"),
        "extra_reconstruction_args": job.get("extra_reconstruction_args") or [],
        "sampler_overrides": job.get("sampler_overrides") or {},
        "expected_sampler": job.get("expected_sampler") or {},
        "expected_method_settings": job.get("expected_method_settings") or {},
        "command": command_without_python,
    }
    return stable_json(fields)


def unique_key_to_index(jobs: list[dict]) -> dict[str, int]:
    key_to_indices: dict[str, list[int]] = {}
    for index, job in enumerate(jobs):
        key_to_indices.setdefault(job_key(job), []).append(index)
    return {
        key: indices[0] for key, indices in key_to_indices.items() if len(indices) == 1
    }


def marker_name(index: int, suffix: str) -> str:
    if suffix == "seconds":
        return f"reconstruction_{index:06d}.reconstruction.seconds"
    return f"reconstruction_{index:06d}.reconstruction.{suffix}"


def marker_path(directory: Path, index: int, suffix: str) -> Path:
    return directory / marker_name(index, suffix)


def metrics_path_for_job(job: dict) -> Path | None:
    command = [str(item) for item in job.get("command") or []]
    output_folder = command_option(command, "--output-folder")
    experiment = command_option(command, "--experiment")
    split = command_option(command, "--split")
    method = command_option(command, "--method")
    algorithm = command_option(command, "--algorithm")
    if not all((output_folder, experiment, split, method, algorithm)):
        return None
    return (
        Path(output_folder) / experiment / split / method / algorithm / "metrics.json"
    )


def expected_sample_count(job: dict) -> int | None:
    command = [str(item) for item in job.get("command") or []]
    value = command_option(command, "--max-samples")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def completed_output_exists(job: dict) -> bool:
    metrics_path = metrics_path_for_job(job)
    if metrics_path is None or not metrics_path.is_file():
        return False
    expected_samples = expected_sample_count(job)
    if expected_samples is None:
        return True
    try:
        with open(metrics_path) as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    metrics = payload.get("metrics")
    return isinstance(metrics, list) and len(metrics) >= expected_samples


def build_index_mapping(old_jobs: list[dict], new_jobs: list[dict]) -> dict[int, int]:
    old_unique = unique_key_to_index(old_jobs)
    new_unique = unique_key_to_index(new_jobs)
    mapping = {}
    for key, old_index in old_unique.items():
        new_index = new_unique.get(key)
        if new_index is not None:
            mapping[old_index] = new_index
    return mapping


def move_marker_to_backup(path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / path.name
    counter = 1
    while target.exists():
        target = backup_dir / f"{path.name}.{counter}"
        counter += 1
    shutil.move(str(path), str(target))
    return target


def migrate_marker_kind(
    *,
    suffix: str,
    directory: Path | None,
    backup_root: Path,
    old_jobs: list[dict],
    index_mapping: dict[int, int],
    skip_output_check: bool,
) -> tuple[int, int]:
    if directory is None or not directory.is_dir():
        return (0, 0)
    pattern = "reconstruction_*.reconstruction.seconds"
    if suffix != "seconds":
        pattern = f"reconstruction_*.reconstruction.{suffix}"

    staged: list[tuple[int, Path]] = []
    stale_count = 0
    for path in sorted(directory.glob(pattern)):
        old_index_text = path.name.split(".")[0].replace("reconstruction_", "")
        try:
            old_index = int(old_index_text)
        except ValueError:
            stale_count += 1
            move_marker_to_backup(path, backup_root / suffix / "unparsed")
            continue
        new_index = index_mapping.get(old_index)
        if new_index is None:
            stale_count += 1
            move_marker_to_backup(path, backup_root / suffix / "unmatched")
            continue
        if (
            suffix == "done"
            and not skip_output_check
            and not completed_output_exists(old_jobs[old_index])
        ):
            stale_count += 1
            move_marker_to_backup(path, backup_root / suffix / "incomplete")
            continue
        staged_path = move_marker_to_backup(path, backup_root / suffix / "staged")
        staged.append((new_index, staged_path))

    migrated_count = 0
    for new_index, staged_path in staged:
        destination = marker_path(directory, new_index, suffix)
        if destination.exists():
            move_marker_to_backup(destination, backup_root / suffix / "overwritten")
        shutil.copy2(staged_path, destination)
        migrated_count += 1
    return migrated_count, stale_count


def write_done_marker_from_output(
    *,
    done_dir: Path,
    failed_dir: Path | None,
    index: int,
    job: dict,
) -> bool:
    destination = marker_path(done_dir, index, "done")
    if destination.exists():
        return False
    metrics_path = metrics_path_for_job(job)
    destination.parent.mkdir(parents=True, exist_ok=True)
    now = _datetime.datetime.now(_datetime.UTC).isoformat()
    with open(destination, "w") as file:
        file.write(f"completed={now}\n")
        file.write("phase=reconstruction\n")
        file.write(f"task_index={index}\n")
        file.write("source=completed_output_reconcile\n")
        if metrics_path is not None:
            file.write(f"metrics_path={metrics_path}\n")

    if failed_dir is not None:
        failed_marker = marker_path(failed_dir, index, "failed")
        if failed_marker.exists():
            failed_marker.unlink()
    return True


def synthesize_done_markers_from_completed_outputs(
    *,
    jobs: list[dict],
    done_dir: Path,
    failed_dir: Path | None,
    skip_output_check: bool,
) -> int:
    if skip_output_check:
        return 0
    created = 0
    for index, job in enumerate(jobs):
        if completed_output_exists(job) and write_done_marker_from_output(
            done_dir=done_dir,
            failed_dir=failed_dir,
            index=index,
            job=job,
        ):
            created += 1
    return created


def reconcile(args: argparse.Namespace) -> dict:
    new_jobs = load_manifest(args.new_json)
    if not args.old_json.is_file():
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.new_json, args.output_json)
        done_from_outputs = synthesize_done_markers_from_completed_outputs(
            jobs=new_jobs,
            done_dir=args.done_dir,
            failed_dir=args.failed_dir,
            skip_output_check=args.skip_output_check,
        )
        return {
            "old_manifest_found": False,
            "new_jobs": len(new_jobs),
            "matched_jobs": 0,
            "migrated_markers": {},
            "stale_markers": {},
            "done_markers_from_completed_outputs": done_from_outputs,
        }

    old_jobs = load_manifest(args.old_json)
    index_mapping = build_index_mapping(old_jobs, new_jobs)
    stamp = _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_root = args.state_dir / "manifest_rebuild_backups" / stamp
    migrated: dict[str, int] = {}
    stale: dict[str, int] = {}
    marker_dirs = {
        "done": args.done_dir,
        "failed": args.failed_dir,
        "seconds": args.runtime_dir,
    }
    for suffix in MARKER_SUFFIXES:
        migrated_count, stale_count = migrate_marker_kind(
            suffix=suffix,
            directory=marker_dirs[suffix],
            backup_root=backup_root,
            old_jobs=old_jobs,
            index_mapping=index_mapping,
            skip_output_check=args.skip_output_check,
        )
        migrated[suffix] = migrated_count
        stale[suffix] = stale_count

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.new_json, args.output_json)
    done_from_outputs = synthesize_done_markers_from_completed_outputs(
        jobs=new_jobs,
        done_dir=args.done_dir,
        failed_dir=args.failed_dir,
        skip_output_check=args.skip_output_check,
    )
    return {
        "old_manifest_found": True,
        "old_jobs": len(old_jobs),
        "new_jobs": len(new_jobs),
        "matched_jobs": len(index_mapping),
        "backup_root": str(backup_root),
        "migrated_markers": migrated,
        "stale_markers": stale,
        "done_markers_from_completed_outputs": done_from_outputs,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-json", type=Path, required=True)
    parser.add_argument("--new-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--done-dir", type=Path, required=True)
    parser.add_argument("--failed-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-output-check",
        action="store_true",
        help="Migrate done markers without checking for completed metrics.json files.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = reconcile(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
