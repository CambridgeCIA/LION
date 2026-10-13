"""Combine scalar-TV box-plot CSVs with an earlier matched PCM dataset run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from LION.pcm.results import RESULT_INDEX_COLUMNS, RESULT_METRIC_COLUMNS

_ACQUISITION_CONFIG_FIELDS = (
    "data",
    "scaling",
    "sampling",
    "uses_normalized_op",
    "noise",
    "base_seed",
)
_ZERO_FILLED_METRICS = tuple(
    metric for metric in RESULT_METRIC_COLUMNS if metric != "duration"
)


def _method_csvs(run_dir: Path, method_id: str) -> dict[str, Path]:
    csv_dir = run_dir / method_id
    files = {path.name: path for path in sorted(csv_dir.glob("*.csv"))}
    if not files:
        raise ValueError(f"No {method_id} CSVs found in {csv_dir}")
    return files


def _indexed_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(RESULT_INDEX_COLUMNS) | set(_ZERO_FILLED_METRICS)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing result columns: {sorted(missing)}")
    if df.duplicated(list(RESULT_INDEX_COLUMNS)).any():
        raise ValueError(f"Duplicate sample/trial/sampling keys in {path}")
    return df.set_index(list(RESULT_INDEX_COLUMNS)).sort_index()


def _assert_matched_keys(a: pd.DataFrame, b: pd.DataFrame, *, context: str) -> None:
    if not a.index.equals(b.index):
        raise ValueError(f"Unmatched sample/trial/sampling/rng_seed rows for {context}")


def merge_scalar_tv_box_results(
    reference_run_dir: Path, scalar_tv_run_dir: Path, output_dir: Path
) -> Path:
    """Validate paired acquisitions and build a *new*, CSV-only box-plot run.

    This never modifies the reference or scalar-TV runs. The zero-filled quality metrics
    must agree, which checks that both runs used the same sampling and noisy measurements.
    """
    reference_run_dir = Path(reference_run_dir).expanduser().resolve()
    scalar_tv_run_dir = Path(scalar_tv_run_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists (nothing overwritten): {output_dir}"
        )
    if reference_run_dir == scalar_tv_run_dir:
        raise ValueError("Reference and scalar-TV runs must be different directories")
    if (reference_run_dir / "scalar_tv").exists():
        raise ValueError(
            f"Reference run already contains scalar_tv: {reference_run_dir}"
        )

    with (reference_run_dir / "config.yaml").open(encoding="utf-8") as f:
        reference_config = yaml.safe_load(f)
    with (scalar_tv_run_dir / "config.yaml").open(encoding="utf-8") as f:
        scalar_tv_config = yaml.safe_load(f)

    for field in _ACQUISITION_CONFIG_FIELDS:
        if reference_config.get(field) != scalar_tv_config.get(field):
            raise ValueError(f"Incompatible experiment settings: {field!r} differs")
    for field in ("num_trials", "num_trials_skip"):
        if reference_config["runtime"].get(field) != scalar_tv_config["runtime"].get(
            field
        ):
            raise ValueError(f"Incompatible experiment runtime: {field!r} differs")

    old_zf = _method_csvs(reference_run_dir, "zero_filled")
    new_zf = _method_csvs(scalar_tv_run_dir, "zero_filled")
    tv_csvs = _method_csvs(scalar_tv_run_dir, "scalar_tv")
    if old_zf.keys() != new_zf.keys() or old_zf.keys() != tv_csvs.keys():
        raise ValueError(
            "Category CSVs differ between the reference and scalar-TV runs"
        )

    sample_counts: dict[str, int] = {}
    for filename in old_zf:
        reference = _indexed_csv(old_zf[filename])
        same_zero_filled = _indexed_csv(new_zf[filename])
        scalar_tv = _indexed_csv(tv_csvs[filename])
        _assert_matched_keys(
            reference, same_zero_filled, context=f"zero-filled {filename}"
        )
        _assert_matched_keys(reference, scalar_tv, context=f"scalar-TV {filename}")
        for metric in _ZERO_FILLED_METRICS:
            if not np.allclose(
                reference[metric].to_numpy(dtype=float),
                same_zero_filled[metric].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Zero-filled {metric} differs in {filename}: acquisitions may not match"
                )
        sample_counts[filename] = len(reference)

    # All checks passed. Copy only CSVs; leave the original runs and artifacts untouched.
    output_dir.mkdir(parents=True)
    for method_dir in sorted(reference_run_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        for source in sorted(method_dir.glob("*.csv")):
            destination = output_dir / method_dir.name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for source in tv_csvs.values():
        destination = output_dir / "scalar_tv" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    merged_config = dict(reference_config)
    merged_config["methods"] = {
        **reference_config["methods"],
        "scalar_tv": scalar_tv_config["methods"]["scalar_tv"],
    }
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged_config, f, sort_keys=False)
    with (output_dir / "merge_sources.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "reference_run": str(reference_run_dir),
                "scalar_tv_run": str(scalar_tv_run_dir),
                "paired_rows_per_category": sample_counts,
            },
            f,
            indent=2,
        )
    return output_dir
