"""PCM experiment result contract."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ZERO_FILLED_METHOD_ID = "zero_filled"


@dataclass(frozen=True)
class PCMResultRow:
    """One row in a PCM result CSV."""

    sample_id: int | None
    trial_num: int
    sampling_rate: float
    coarse_j: int
    rng_seed: int
    mse: float
    psnr: float
    ssim: float
    haarpsi: float
    pearson_corr: float
    duration: float


RESULT_COLUMNS = tuple(field.name for field in fields(PCMResultRow))
RESULT_INDEX_COLUMNS = (
    "sample_id",
    "trial_num",
    "sampling_rate",
    "coarse_j",
    "rng_seed",
)
RESULT_METRIC_COLUMNS = tuple(
    column for column in RESULT_COLUMNS if column not in RESULT_INDEX_COLUMNS
)


def result_csv_path(root: Path, method_id: str, category: str | None = None) -> Path:
    """Return the CSV path for one method."""
    filename = f"{category}.csv" if category is not None else "results.csv"
    return root / method_id / filename


def append_result(
    *, root: Path, method_id: str, category: str | None, result: PCMResultRow
) -> None:
    """Append one result row to the corresponding CSV."""
    csv_path = result_csv_path(root, method_id, category)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


def iter_result_csvs(root: Path):
    """Yield ``(method_id, category, csv_path)`` for result CSVs under ``root``."""
    for method_dir in sorted(path for path in Path(root).iterdir() if path.is_dir()):
        for csv_path in sorted(method_dir.glob("*.csv")):
            category = None if csv_path.name == "results.csv" else csv_path.stem
            yield method_dir.name, category, csv_path
