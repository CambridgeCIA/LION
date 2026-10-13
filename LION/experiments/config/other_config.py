from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from LION.utils.parameter import LIONParameter


@dataclass(kw_only=True)
class NormParameter(LIONParameter):
    """Configuration for p-norm data fidelity term.

    Attributes:
        ord: The order of the norm. Currently supports 1 and 2. If 2, assume squared 2-norm.
        factor: A scaling factor for the norm. If None, defaults to 1/ord.
        normalize: Whether to normalize by the number of measurements.
    """

    ord: Literal[1, 2] = 2
    factor: float | None = None
    normalize: bool
