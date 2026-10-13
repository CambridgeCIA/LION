from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from LION.utils.parameter import LIONParameter


@dataclass(kw_only=True)
class AdamParameter(LIONParameter):
    """Configuration for optimizer construction."""

    name: Literal["Adam", "AdamW"]
    learning_rate: float
    betas: list[float]
    eps: float
    weight_decay: float


@dataclass(kw_only=True)
class CosineAnnealingLRParameter(LIONParameter):
    T_max: int | None
    eta_min: float
