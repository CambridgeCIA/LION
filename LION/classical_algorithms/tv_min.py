from __future__ import annotations

from typing import Callable, Optional

import torch

from LION.classical_algorithms._batching import batched_reconstruction
from LION.CTtools.ct_geometry import Geometry
from LION.operators.Operator import Operator
from LION.utils.tomosipo_import import import_tomosipo


def tv_min(
    sino: torch.Tensor,
    op: Operator | Geometry,
    lam: float,
    num_iterations: int = 500,
    L: Optional[float] = None,
    non_negativity: bool = False,
    progress_bar: bool = False,
    callbacks: Optional[list[Callable]] = None,
) -> torch.Tensor:
    """Compute TV-minimization reconstructions on a batched input.

    See ts_algorithms.tv_min2d for more details.
    """
    ts_tv_min = import_tomosipo("ts_algorithms", "tv_min").tv_min2d

    callbacks = [] if callbacks is None else callbacks

    def recon_fn(op, sino_i, i):
        return ts_tv_min(
            op, sino_i, lam, num_iterations, L, non_negativity, progress_bar, callbacks
        )

    return batched_reconstruction(recon_fn, sino, op)
