"""Type definitions used by training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

StateDict = dict[str, Any]


@dataclass(kw_only=True)
class TrainingState:
    """Mutable one-indexed training progress shared by trainer and logger."""

    one_indexed_epoch: int = 0
    one_indexed_global_step: int = 0
