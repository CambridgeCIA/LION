"""
This module implements the loss function for training the score-based model.

Author: Tianzhen Peng

References
----------
.. [Song2021] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., 
   Ermon, S., & Poole, B. (2021). "Score-Based Generative Modeling 
   through Stochastic Differential Equations." ICLR. https://openreview.net/forum?id=PxTIG12RRHS

.. [Song2022] Song, Y., Shen, L., Xing, L., & Ermon, S. (2022). 
   "Solving Inverse Problems in Medical Imaging with Score-Based 
   Generative Models." ICLR. https://openreview.net/forum?id=vaRCHVj0uGI
"""

from typing import Optional
import torch
import torch.nn as nn
from .sde import SimpleForwardSDE

class SMLoss(nn.Module):
    """
    The loss function for training the score-based model. Equation (7) in [Song2021]. The weighting function lambda(t) is set to beta(t)^2 ([Song2022]), equivalent to the "typical choice" in [Song2021].
    """
    def __init__(self, score_fn, sde: SimpleForwardSDE, eps: float = 1e-5):
        """
        Args:
            score_fn (callable): the score function to be trained, which accepts (xt, t) as input and outputs the score function.
            sde (SimpleForwardSDE): the SDE to be used for training.
            eps (float): a small constant to avoid numerical issues.
        """
        super().__init__()
        self.score_fn = score_fn
        self.sde = sde
        self.eps = eps


    def forward(self, x0: torch.Tensor, generator: Optional[torch.Generator] = None):
        """
        Compute the loss for training the score-based model.

        Args:
            x0: torch.Tensor of shape (batch_size, ...). A batch of data samples.
            generator: Optional torch.Generator for deterministic randomness.

        Returns:
            torch.Tensor: the computed loss.
        """
        t = torch.rand(x0.shape[0], device=x0.device, generator=generator) * (1. - self.eps) + self.eps
        z = torch.randn_like(x0, generator=generator)
        alpha_t, beta_t = self.sde.transition_dist(x0, t)
        xt = alpha_t * x0 + beta_t * z
        score_pred = self.score_fn(xt, t)
        loss = ((score_pred * beta_t + z) ** 2).sum(dim=tuple(range(1, len(x0.shape)))).mean()
        return loss