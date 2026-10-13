from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True)
class DataConfig:
    name: str
    path: Path
    save_generated: bool = False

    pregenerated_dir: Path | None = None
    pregenerated_seed: int | None = None

    permutation_file: Path | None = None
    permutation_seed: int | None = None  # Old default value: 3841
    generate_permutation_if_missing: bool = False

    def __post_init__(self) -> None:
        """Normalise path-like fields."""
        object.__setattr__(self, "path", Path(self.path))
        if self.pregenerated_dir is not None:
            object.__setattr__(self, "pregenerated_dir", Path(self.pregenerated_dir))
        if self.permutation_file is not None:
            object.__setattr__(self, "permutation_file", Path(self.permutation_file))
        if self.generate_permutation_if_missing and self.permutation_seed is None:
            raise ValueError(
                "permutation_seed is required when generate_permutation_if_missing=true"
            )
        if (self.pregenerated_dir is None) != (self.pregenerated_seed is None):
            raise ValueError(
                "pregenerated_dir and pregenerated_seed must either both be set or both be null"
            )


@dataclass(kw_only=True)
class SupervisedExtraDataConfig:
    """Configuration for supervised training data."""

    train_split: int
    val_split: int
    test_split: int
    augment_spatial: bool


@dataclass(kw_only=True)
class SelfSupervisedExtraDataConfig:
    sample_ids: list[int] | None = None
    start_id: int | None = None
    end_id: int | None = None
    permutation_start: int | None = None
    permutation_end: int | None = None

    p_target_train: float
    p_target_val: float | None

    base_train_partition_seed: int
    base_val_partition_seed: int

    this_sample_id: int = -1

    def __post_init__(self) -> None:
        if self.sample_ids is not None and not self.sample_ids:
            raise ValueError("sample_ids must not be empty")
        has_id_range = self.start_id is not None or self.end_id is not None
        has_permutation_range = (
            self.permutation_start is not None or self.permutation_end is not None
        )
        active_modes = (
            int(self.sample_ids is not None)
            + int(has_id_range)
            + int(has_permutation_range)
        )
        if active_modes > 1:
            raise ValueError(
                "Use exactly one selection mode: sample_ids, start_id/end_id, or permutation_start/permutation_end"
            )
