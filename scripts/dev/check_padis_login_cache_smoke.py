"""Lightweight login-node cache smoke check for PaDIS LIDC archives.

This intentionally avoids full train-cache materialisation. It verifies that the
default 256x256 train and validation archives match the current dataset metadata,
then stages and loads only the much smaller validation archive through the same
code path used by the Slurm training jobs.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.util
import json
import os
import pathlib
import resource
import shutil
import sys
import time
from types import ModuleType

import torch

LION_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(LION_ROOT) not in sys.path:
    sys.path.insert(0, str(LION_ROOT))

from LION.CTtools.ct_geometry import Geometry  # noqa: E402
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI  # noqa: E402


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _load_lidc256_module() -> ModuleType:
    script_path = (
        LION_ROOT / "scripts" / "paper_scripts" / "PaDIS" / "PaDIS_LIDC_256.py"
    )
    spec = importlib.util.spec_from_file_location(
        "padis_lidc_256_login_smoke", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonable(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dataset(args: argparse.Namespace, mode: str):
    geometry = Geometry.default_parameters(image_scaling=0.5)
    params = LIDC_IDRI.default_parameters(geometry=geometry, task="image_prior")
    params.device = torch.device("cpu")
    params.max_num_slices_per_patient = int(args.max_slices_per_patient)
    params.pcg_slices_nodule = float(args.pcg_slices_nodule)
    if args.data_folder is not None:
        params.folder = args.data_folder
    return LIDC_IDRI(mode, parameters=params, geometry_parameters=geometry)


def _archive_details(
    padis_lidc256: ModuleType,
    dataset,
    mode: str,
    archive_folder: pathlib.Path,
):
    cache_path, metadata = padis_lidc256.cache_path_for_dataset(
        dataset, mode, archive_folder
    )
    slices_to_load = metadata.get("slices_to_load", {})
    metadata_summary = {
        key: value for key, value in metadata.items() if key != "slices_to_load"
    }
    metadata_summary["slices_to_load_patients"] = len(slices_to_load)
    metadata_summary["selected_slices"] = sum(
        len(slice_indices) for slice_indices in slices_to_load.values()
    )
    metadata_summary["metadata_sha256"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()
    archive_path = archive_folder / f"{cache_path.name}.zst"
    details = {
        "mode": mode,
        "dataset_len": len(dataset),
        "cache_name": cache_path.name,
        "archive_path": archive_path,
        "archive_exists": archive_path.is_file(),
        "metadata": metadata_summary,
    }
    if archive_path.is_file():
        details["archive_size_mib"] = archive_path.stat().st_size / 1024**2
    return details


def run(args: argparse.Namespace) -> dict[str, object]:
    """Exercise login-node cache discovery without starting GPU training."""
    start = time.perf_counter()
    padis_lidc256 = _load_lidc256_module()
    train_dataset = _dataset(args, "train")
    validation_dataset = _dataset(args, "validation")

    train_details = _archive_details(
        padis_lidc256, train_dataset, "train", args.cache_archive_folder
    )
    validation_details = _archive_details(
        padis_lidc256,
        validation_dataset,
        "validation",
        args.cache_archive_folder,
    )
    missing = [
        str(details["archive_path"])
        for details in (train_details, validation_details)
        if not details["archive_exists"]
    ]
    if missing:
        raise FileNotFoundError(
            "Missing default PaDIS cache archive(s):\n  " + "\n  ".join(missing)
        )

    stage_details: dict[str, object] = {
        "skipped": bool(args.skip_stage_validation_cache)
    }
    if not args.skip_stage_validation_cache:
        if shutil.which("zstd") is None:
            raise RuntimeError(
                "zstd is required to stage the validation cache archive."
            )
        stage_start = time.perf_counter()
        images = padis_lidc256.materialize_image_prior_dataset(
            validation_dataset,
            "validation",
            args.cache_folder,
            rebuild_cache=False,
            source_cache_folder=None,
            cache_archive_folder=args.cache_archive_folder,
            write_archive=False,
            require_cache_hit=True,
        )
        stage_details = {
            "skipped": False,
            "seconds": time.perf_counter() - stage_start,
            "cache_folder": args.cache_folder,
            "images_shape": list(images.shape),
            "images_dtype": str(images.dtype),
            "images_size_mib": images.numel() * images.element_size() / 1024**2,
            "images_min": float(torch.amin(images).item()),
            "images_max": float(torch.amax(images).item()),
        }
        if not args.keep_cache:
            shutil.rmtree(args.cache_folder, ignore_errors=True)
            stage_details["cache_removed"] = True

    return {
        "status": "passed",
        "seconds": time.perf_counter() - start,
        "rss_gib": _rss_gib(),
        "lion_root": LION_ROOT,
        "data_folder": args.data_folder,
        "cache_archive_folder": args.cache_archive_folder,
        "train_archive": train_details,
        "validation_archive": validation_details,
        "validation_stage": stage_details,
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the cache-smoke command-line parser."""
    data_root = pathlib.Path(
        os.environ.get("LION_DATA_PATH", "/home/tjh200/rds/hpc-work/Datasets")
    )
    default_cache_folder = (
        pathlib.Path("/tmp")
        / getpass.getuser()
        / f"lion_padis_login_cache_smoke_{os.getpid()}"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument(
        "--cache-archive-folder",
        type=pathlib.Path,
        default=data_root / "processed" / "LIDC-IDRI-cache" / "padis_256" / "archives",
    )
    parser.add_argument(
        "--cache-folder", type=pathlib.Path, default=default_cache_folder
    )
    parser.add_argument("--max-slices-per-patient", type=int, default=4)
    parser.add_argument("--pcg-slices-nodule", type=float, default=0.5)
    parser.add_argument("--skip-stage-validation-cache", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--json", type=pathlib.Path, default=None)
    return parser


def main() -> None:
    """Run the cache smoke check and print its JSON report."""
    args = build_parser().parse_args()
    summary = run(args)
    text = json.dumps(_jsonable(summary), indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
