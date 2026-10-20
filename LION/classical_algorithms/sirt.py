from __future__ import annotations

from typing import Callable, Optional

import torch

from LION.classical_algorithms._batching import batched_reconstruction, per_sample
from LION.CTtools.ct_geometry import Geometry
from LION.operators.Operator import Operator
from LION.utils.tomosipo_import import import_tomosipo


def sirt(
    sino: torch.Tensor,
    op: Operator | Geometry,
    num_iterations: int,
    min_constraint: Optional[float] = None,
    max_constraint: Optional[float] = None,
    x_init: Optional[torch.Tensor] = None,
    volume_mask: Optional[torch.Tensor] = None,
    projection_mask: Optional[torch.Tensor] = None,
    progress_bar: bool = False,
    callbacks: Optional[list[Callable]] = None,
) -> torch.Tensor:
    """Compute SIRT reconstructions on a batched input.

    ``x_init`` may be a single initial guess, shared by all samples, or a batch
    of them (one per sample). See ts_algorithms.sirt for more details.
    """
    ts_sirt = import_tomosipo("ts_algorithms", "sirt").sirt

    callbacks = [] if callbacks is None else callbacks

    def recon_fn(op, sino_i, i):
        return ts_sirt(
            op,
            sino_i,
            num_iterations,
            min_constraint,
            max_constraint,
            per_sample(x_init, op, i),
            volume_mask,
            projection_mask,
            progress_bar,
            callbacks,
        )

    return batched_reconstruction(recon_fn, sino, op)
