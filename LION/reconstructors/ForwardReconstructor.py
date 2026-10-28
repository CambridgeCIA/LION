# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""A model's own forward pass, as a reconstructor."""

from __future__ import annotations

import torch

from LION.optimizers.LIONsolver import model_forward
from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.utils.parameter import LIONParameter


class ForwardReconstructor(LIONReconstructor):
    """Reconstruct with a model's forward pass, as its solvers train it
    (``LION.optimizers.LIONsolver.model_forward``): image-domain models (e.g.
    FBPConvNet) get an FDK reconstruction first, sinogram-domain models (e.g.
    LPD) the measurements; with the model's normalisation (its
    ``set_normalisation()``, saved with it), the input is normalised and the
    output mapped back.

    ``ForwardReconstructor(operator, model)``; it has no parameters. A batch is
    reconstructed in one pass. ``x0`` is not used.
    """

    @staticmethod
    def default_parameters() -> LIONParameter:
        return LIONParameter()

    def reconstruct(
        self, y: torch.Tensor, x0: torch.Tensor | None = None, prog_bar: bool = False
    ) -> torch.Tensor:
        """Reconstruct measurements ``y``: one sample (the operator's range
        shape) or a batch of them, in one forward pass."""
        if self.model is None:
            raise ValueError(
                "ForwardReconstructor has no model: set it first "
                "(reconstructor.model = model, or instantiate(source, model))"
            )
        if y.dim() == len(self.op.range_shape) + 1:
            return model_forward(self.model, y, self.op)
        return self.reconstruct_sample(y, x0, prog_bar)

    def reconstruct_sample(self, y, x0=None, prog_bar=False):
        return model_forward(self.model, y.unsqueeze(0), self.op).squeeze(0)
