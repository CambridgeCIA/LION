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
    def __init__(self, sde: SimpleForwardSDE, score_fn: callable, N: int, eps: float = 1e-5):
        """
        Args:
            sde (SimpleForwardSDE): the forward SDE.
            score_fn (callable): a function that takes in x and t and outputs the score function.
            N (int): number of discretization steps for the reverse-time SDE.
            eps (float): the minimum time step (e.g. 1e-5) to integrate down to.
        """
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.N = N
        self.eps = eps
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
        dt = (1.0 - self.eps) / self.N
        x_mean = x - self.rsde.drift_coeff(x, t) * dt
        x_next = x_mean + self.rsde.diffusion_coeff(x, t) * math.sqrt(dt) * torch.randn_like(x)
        return x_next, x_mean

class ReverseDiffusionVE(Predictor):
    """
    Reverse diffusion predictor for VESDE (Algorithm 2 in [Song2021]).
    """
    def __init__(self, sde: VESDE, score_fn: callable, N: int, eps: float = 1e-5):
        super().__init__(sde, score_fn, N, eps)

    def step(self, x: torch.Tensor, t: torch.Tensor):
        dt = (1.0 - self.eps) / self.N
        sigma = self.sde.beta(t)
        sigma_next = self.sde.beta(torch.clamp(t - dt, min=self.eps))
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
            eps = 2 * (self.snr * z_norm_mean / g_norm) ** 2 # corrector step size
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
    dt = (1.0 - predictor.eps) / predictor.N
    timesteps = 1.0 - torch.arange(predictor.N, device=x.device) * dt
    for i in range(predictor.N):
        if verbose and i % verbose_freq == 0:
            print(f"Step {i}/{predictor.N}")
        t = timesteps[i].expand(x.shape[0])
        if hijack is not None:
            x = hijack(x, t)
        x, x_mean = predictor.step(x, t)
        if i < predictor.N - 1:
            t_next = timesteps[i + 1].expand(x.shape[0])
            x = corrector.correct(x, t_next)
    return x_mean

def tweedie_denoise(x: torch.Tensor, t: torch.Tensor, sde: SimpleForwardSDE, score_fn: callable):
    """
    Tweedie denoising after the final step of sampling. Not mentioned in [Song2021] or [Song2022], but appeared in `score_inverse_problems/cs.py`.

    Args:
        x: torch.Tensor of shape (batch_size, ...) representing the input.
        t: torch.Tensor of shape (batch_size,) representing the time step.
        sde: the forward SDE.
        score_fn: the score function.

    Returns:
        torch.Tensor of shape (batch_size, ...) representing the denoised output.
    """
    alpha_t, beta_t = sde.transition_dist(x, t)
    return (x + beta_t**2 * score_fn(x, t)) / alpha_t

def pc_sampler_new(x: torch.Tensor, predictor: Predictor, corrector: Corrector, hijack=None, denoise=True, verbose=False, verbose_freq=10):
    """
    A new version of predictor-corrector sampler for solving the reverse-time SDE consistent with `score_inverse_problems/cs.py`. The main differences are ...

    Args:
        x: torch.Tensor of shape (batch_size, ...) representing the initial solution at time 1.
        predictor (Predictor): the predictor for estimating the solution of the reverse-time SDE at the next time step.
        corrector (Corrector): the corrector for correcting the marginal distribution of the estimated samples.
        hijack (callable, optional): a function that takes in x and t and outputs the hijacked solution at time t before application of predictor.
        denoise (bool): whether to apply tweedie denoising after the final step.
        verbose (bool): whether to print verbose output.
        verbose_freq (int): the frequency of verbose output.
    Returns:
        torch.Tensor of shape (batch_size, ...) representing the final solution at time 0.
    """
    dt = (1.0 - predictor.eps) / predictor.N
    timesteps = 1.0 - torch.arange(predictor.N, device=x.device) * dt
    for i in range(predictor.N):
        if verbose and i % verbose_freq == 0:
            print(f"Step {i}/{predictor.N}")
        t = timesteps[i].expand(x.shape[0])
        if hijack is not None:
            x = hijack(x, t)
        x = corrector.correct(x, t)
        if hijack is not None:
            x = hijack(x, t)
        x, x_mean = predictor.step(x, t)
    if denoise:
        t = torch.tensor(predictor.eps, device=x.device).expand(x.shape[0])
        x = tweedie_denoise(x, t, predictor.sde, predictor.score_fn)
        if hijack is not None:
            x = hijack(x, torch.zeros_like(t))
    return x

def get_score_conditional(score_fn: callable, sde: SimpleForwardSDE, y: torch.Tensor, op: callable, op_adj: callable, rate=1.0):
    """
    Compute the conditional score function Grad_x log p_t(x | y) using the unconditional score function Grad_x log p_t(x) and the likelihood Grad_x log p_t(y(t) | x(t)). See Appendix I.4 in [Song2021] for details.

    Args:
        score_fn (callable): the unconditional score function.
        sde: the forward SDE.
        y: torch.Tensor of shape (batch_size, ...) representing the observed data at time 0.
        sde: the forward SDE.
        op: the forward operator that maps x to y. Should be a function that takes in a torch.Tensor of shape (batch_size, ...) and outputs a torch.Tensor of shape (batch_size, ...).
        op_adj: the adjoint of the forward operator (excluding the batch dimension). Should be a function that takes in a torch.Tensor of shape (batch_size, ...) and outputs a torch.Tensor of shape (batch_size, ...).
        rate: multiplier for likelihood std scaling.

    Returns:
        torch.Tensor of shape (batch_size, ...) representing the conditional score function evaluated at (x, t).
    """
    def score_conditional(x, t):
        alpha_t, beta_t = sde.transition_dist(y, t)
        y_t = alpha_t * y + beta_t * op(torch.randn_like(x))
        beta_t = beta_t.view(-1, *[1] * (len(x.shape) - 1))
        grad_log_likelihood = -op_adj(op(x) - y_t) / ((rate * beta_t) ** 2)
        return score_fn(x, t) + grad_log_likelihood

    return score_conditional