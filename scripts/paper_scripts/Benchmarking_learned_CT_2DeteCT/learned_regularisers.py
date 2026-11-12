# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Zakhar Shumaylov, Ander Biguri
# Modifications: LION's current API; from the 2024 code of the benchmark
# =============================================================================
"""AR and ACR as the 2DeteCT benchmark paper's code (Zakhar Shumaylov, 2024)
trained and used them, where it differs from LION's ``ACR`` and
``LearnedRegularizerReconstructor``: ``ACR2024``, and the reconstruction,
``GradientDescent2024``."""

import torch

from LION.models.learned_regularizer.ACR import ACR
from LION.reconstructors.LearnedRegularizerReconstructor import (
    LearnedRegularizerReconstructor,
)
from LION.utils.parameter import LIONParameter


class ACR2024(ACR):
    """LION's ``ACR`` as the 2024 code had it:

    - the ``||.||²`` term is of the last layer's output, not of the image;
    - its weight starts at softplus(-36), not softplus(-9);
    - the non-negative weights start as torch's default (then clipped at 0):
      the 2024 code drew them in [0, 1e-3], but assigned them to the computed
      weights, not to those learned, so it had no effect."""

    @staticmethod
    def default_parameters():
        """``ACR``'s, with ``l2_penalty`` -36."""
        params = ACR.default_parameters()
        params.l2_penalty = -36.0
        return params

    def initialize_weights(self, min_val=0.0, max_val=1e-3):
        """Nothing: the 2024 code's had no effect (see the class)."""
        return self

    def forward(self, x):
        """The regulariser's value for each image of ``x``: (B, 1)."""
        z = self.first_activation(self.first_layer(x))
        for i in range(self.model_parameters.layers):
            z = getattr(self, f"ICNN_layer_{i}")(z, x)
        z = self.last_layer(z)
        return self.pool(z).reshape(-1, 1) + self.L2(z)


class GradientDescent2024(LearnedRegularizerReconstructor):
    """The 2024 code's reconstruction with a learned regulariser R: from the
    initial reconstruction (FDK), ``n_steps`` steps of gradient descent with
    momentum on ½‖Ax - y‖² + α R(x). Each step's size is reduced (by
    ``backtracking``) until the Armijo condition holds for a plain gradient
    step, and never grows back; with ``min_step``, the descent stops once the
    step is smaller."""

    @staticmethod
    def default_parameters() -> LIONParameter:
        """``LearnedRegularizerReconstructor``'s, with ``momentum`` 0.5 and
        ``min_step`` None (no stop)."""
        params = LearnedRegularizerReconstructor.default_parameters()
        params.momentum = 0.5
        params.min_step = None
        return params

    def reconstruct_sample(self, y, x0=None, verbose=1):
        """Reconstruct one measurement ``y``, as the 2024 code (see the
        class)."""
        p = self.parameters
        x = (self.initial_guess(y) if x0 is None else x0).detach().clone()
        step = p.step_size
        velocity = None
        for _ in range(p.n_steps):
            energy = self.energy(x, y)
            grad = self.gradient(x, y)
            decrease = 0.5 * torch.sum(grad**2)
            while self.energy(x - step * grad, y) > energy - step * decrease:
                step *= p.backtracking
                if step < torch.finfo(x.dtype).tiny:
                    return x
            if p.min_step is not None and step < p.min_step:
                return x
            velocity = grad if velocity is None else p.momentum * velocity + grad
            x = x - step * velocity
        return x
