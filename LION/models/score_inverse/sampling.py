"""
This module implements the abstract base class for predictors (Predictor) and correctors (Corrector) in the sampling process of Score-Based Generative Models.

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

from abc import ABC, abstractmethod
import torch
from torch.linalg import vector_norm
from .sde import SimpleForwardSDE, SimpleReverseSDE, VESDE
import math

class Predictor(ABC):
    """
    Predictor for estimate the sample of the reverse-time SDE at the next time step. [Song2021]
    """
    def __init__(self, sde: SimpleForwardSDE, score_fn: callable, N: int):
        """
        Args:
            sde (SimpleForwardSDE): the forward SDE.
            score_fn (callable): a function that takes in x and t and outputs the score function.
            N (int): number of discretization steps for the reverse-time SDE.
        """
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.N = N
        self.rsde = SimpleReverseSDE(sde, score_fn)

    @abstractmethod
    def step(self, x: torch.Tensor, t: torch.Tensor):
        """
        Update the solution of the reverse-time SDE.

        Args:
            x: torch.Tensor of shape (batch_size, ...).
            t: torch.Tensor of shape (batch_size,).

        Returns:
            x_next: torch.Tensor of shape (batch_size, ...) representing the predicted solution at the next time step (t - 1 / N).
            x_mean: torch.Tensor of shape (batch_size, ...) representing the mean of predicted solution at the next time step (t - 1 / N).
        """
        pass

class EulerMaruyama(Predictor):
    """
    Euler-Maruyama predictor for solving the reverse-time SDE.
    """
    def step(self, x: torch.Tensor, t: torch.Tensor):
        dt = 1.0 / self.N
        x_mean = x - self.rsde.drift_coeff(x, t) * dt
        x_next = x_mean + self.rsde.diffusion_coeff(x, t) * math.sqrt(dt) * torch.randn_like(x)
        return x_next, x_mean

class ReverseDiffusionVE(Predictor):
    """
    Reverse diffusion predictor for VESDE (Algorithm 2 in [Song2021]).
    """
    def __init__(self, sde: VESDE, score_fn: callable, N: int):
        super().__init__(sde, score_fn, N)

    def step(self, x: torch.Tensor, t: torch.Tensor):
        dt = 1.0 / self.N
        sigma = self.sde.beta(t)
        sigma_next = self.sde.beta(torch.clamp(t - dt, min=0.0))
        var_diff = (sigma**2 - sigma_next**2).view(-1, *([1] * (len(x.shape) - 1)))
        var_diff = torch.clamp(var_diff, min=0.0) # to prevent numerical issues
        x_mean = x + var_diff * self.score_fn(x, t)
        x_next = x_mean + torch.sqrt(var_diff) * torch.randn_like(x)
        return x_next, x_mean
        

class Corrector(ABC):
    """
    A score-based MCMC that corrects the marginal distribution of the estimated samples. [Song2021]
    """
    def __init__(self, score_fn: callable, M: int):
        """
        Args:
            score_fn (callable): a function that takes in x and t and outputs the score function.
            M (int): number of MCMC steps.
        """
        super().__init__()
        self.score_fn = score_fn
        self.M = M

    @abstractmethod
    def correct(self, x: torch.Tensor, t: torch.Tensor):
        """
        Update the solution of the reverse-time SDE.

        Args:
            x: torch.Tensor of shape (batch_size, ...).
            t: torch.Tensor of shape (batch_size,).

        Returns:
            torch.Tensor of shape (batch_size, ...) representing the corrected solution at time t.
        """
        pass

class LangevinDynamics(Corrector):
    """
    Langevin dynamics corrector for VESDE. Algorithm 4 in [Song2021].
    """
    def __init__(self, score_fn: callable, M: int, snr: float):
        """
        Args:
            score_fn (callable): a function that takes in x and t and outputs the score function.
            M (int): number of MCMC steps.
            snr (float): parameter that controls the step size of Langevin dynamics. See Appendix G in [Song2021] for details.
        """
        super().__init__(score_fn, M)
        self.snr = snr
        
    def correct(self, x: torch.Tensor, t: torch.Tensor):
        for _ in range(self.M):
            z = torch.randn_like(x)
            g = self.score_fn(x, t)
            z_norm = vector_norm(z, dim=tuple(range(1, len(x.shape))), keepdim=True)
            z_norm_mean = z_norm.mean() # Appendix G in [Song2021] suggests using the mean across the minibatch.
            g_norm = vector_norm(g, dim=tuple(range(1, len(x.shape))), keepdim=True)
            g_norm = torch.clamp(g_norm, min=1e-10) # to prevent division by zero
            eps = 2 * (self.snr * z_norm_mean / g_norm) ** 2
            x = x + eps * g + torch.sqrt(2 * eps) * z
        return x

def pc_sampler(x: torch.Tensor, predictor: Predictor, corrector: Corrector, hijack=None, verbose=False, verbose_freq=10):
    """
    Predictor-corrector sampler for solving the reverse-time SDE. Algorithm 1 in [Song2021].

    Args:
        x: torch.Tensor of shape (batch_size, ...) representing the initial solution at time 1.
        predictor (Predictor): the predictor for estimating the solution of the reverse-time SDE at the next time step.
        corrector (Corrector): the corrector for correcting the marginal distribution of the estimated samples.
        hijack (callable, optional): a function that takes in x and t and outputs the hijacked solution at time t before application of predictor.
        verbose (bool): whether to print verbose output.
        verbose_freq (int): the frequency of verbose output.
    Returns:
        torch.Tensor of shape (batch_size, ...) representing the final solution at time 0.
    """
    for i in range(predictor.N):
        if verbose and i % verbose_freq == 0:
            print(f"Step {i}/{predictor.N}")
        t = torch.tensor(1.0 - i / predictor.N, device=x.device).expand(x.shape[0])
        if hijack is not None:
            x = hijack(x, t)
        x, x_mean = predictor.step(x, t)
        x = corrector.correct(x, t - 1 / predictor.N)
    return x_mean