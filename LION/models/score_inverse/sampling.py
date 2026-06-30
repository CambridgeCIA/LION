"""
Predictor-Corrector (PC) samplers for reverse-time SDE integration.

Processes & Samplers:
  1. pc_sampler (Standard): Implements the standard loop: Hijack -> Predictor -> Corrector.
  2. pc_sampler_new (JAX-Equivalent): Follows the sequence: Hijack -> Corrector -> Hijack -> Predictor, terminating with Tweedie denoising and a final Hijack data-consistency projection.
  3. get_score_conditional: Modifies the score function using the conditional likelihood gradient for standard conditional sampling.
  4. get_hijack / get_hijack_new: Return a hijack function for data consistency, get_hijack uses the invertible operator T in [Song2022], while get_hijack_new uses a generalized right inverse.

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
    A new version of predictor-corrector sampler for solving the reverse-time SDE,
    consistent with the ``projection_sampler`` in ``score_inverse_problems/cs.py``.

    The key differences from :func:`pc_sampler` are:

    * The per-step update order is ``hijack → correct → hijack → predict`` instead
      of ``hijack → predict → correct``.  Applying hijack before the corrector
      ensures the corrector operates on a measurement-consistent iterate.
    * After the final predictor step, an optional Tweedie denoising step is applied
      (see :func:`tweedie_denoise`), followed by a final hijack to re-impose
      data consistency on the denoised estimate.

    Args:
        x: torch.Tensor of shape (batch_size, ...) representing the initial noisy
            sample drawn from the prior at time ``t = 1``.
        predictor (Predictor): the predictor for estimating the solution of the
            reverse-time SDE at the next time step.
        corrector (Corrector): the corrector for correcting the marginal distribution
            of the estimated samples.
        hijack (callable, optional): a function ``hijack(x, t) -> x`` that enforces
            data consistency at time ``t``.  If ``None``, no hijacking is applied.
        denoise (bool): whether to apply Tweedie denoising after the final predictor
            step.  Defaults to ``True``.
        verbose (bool): whether to print progress messages.  Defaults to ``False``.
        verbose_freq (int): print a progress message every this many steps.
            Defaults to ``10``.

    Returns:
        torch.Tensor of shape (batch_size, ...) representing the final sample at
        time ``t ≈ 0`` (or the Tweedie-denoised estimate if ``denoise=True``).
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
    else:
        return x_mean

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

def get_hijack(sde: SimpleForwardSDE, full_op: callable, full_op_inv: callable, y: torch.Tensor, mask: torch.Tensor, lb=1.0, clean_hijack=False, clamp_factor=None, data_range=(0.0, 1.0)):
    """
    Return a hijack function for data consistency, implementing equation (9) in [Song2022], consistent with the ``inpaint_update_fn`` in ``score_inverse_problems/cs.py``.

    .. math::

        x_{\\text{hijacked}} = T^{-1}\\!\\left[\\lambda \\Lambda \\odot y_t
        + (1 - \\lambda \\Lambda) \\odot T(x_t)\\right]

    Args:
        sde (SimpleForwardSDE): the forward SDE used to compute the transition distribution ``(α_t, β_t)``.
        full_op (callable): the invertible forward operator ``T`` in [Song2022], mapping image-space tensors of shape ``(batch_size, ...)`` to the measurement domain.
        full_op_inv (callable): the exact inverse ``T^{-1}`` of ``full_op``.
        y (torch.Tensor): the clean observed data in the measurement domain, of shape ``(batch_size, ...)``.  Used to form the noisy measurement ``y_t`` at each time step.
        mask (torch.Tensor): the binary mask ``Λ`` in [Song2022] selecting the observed frequency components.  Broadcast-compatible with the output of ``full_op``.
        lb (float): hijack weight ``λ`` in [Song2022]. ``lb=1.0`` gives the strongest correction. Defaults to ``1.0``.
        clean_hijack (bool): if ``True``, use ``y_t = α_t · y`` (no noise injection) instead of the SDE-consistent noisy measurement.  Should be ``False`` to remain consistent with [Song2022].  Defaults to ``False``.
        clamp_factor (float or None): if not ``None``, clamp the hijacked iterate to ``[data_range[0] - clamp_factor · β_t, data_range[1] + clamp_factor · β_t]`` to prevent overflow. Defaults to ``None``.
        data_range (tuple[float, float]): the expected range ``(min, max)`` of clean image values.  Active only when ``clamp_factor`` is not ``None``. Defaults to ``(0.0, 1.0)``.

    Returns:
        callable: a function ``hijack(x, t) -> x_hijacked`` that enforces measurement consistency at time step ``t``.
    """
    def hijack(x, t):
        alpha_t, beta_t = sde.transition_dist(y, t)
        if clean_hijack:
            y_t = alpha_t * y
        else:
            y_t = alpha_t * y + beta_t * full_op(torch.randn_like(x))
        Tx = full_op(x)
        x_hijacked = full_op_inv(lb * mask * y_t + (1.0 - lb * mask) * Tx)
        
        if clamp_factor is not None:
            x_hijacked = torch.clamp(x_hijacked, data_range[0] - clamp_factor * beta_t, data_range[1] + clamp_factor * beta_t)
        return x_hijacked
    return hijack

def get_hijack_new(sde: SimpleForwardSDE, op: callable, pseudo_inv: callable, y: torch.Tensor, lb=1.0, clean_hijack=False, clamp_factor=None, data_range=(0.0, 1.0)):
    """
    Return a hijack function for data consistency using a pseudo-inverse operator, generalising equation (9) in [Song2022] to settings where an invertible operator ``T`` is not available.

    Unlike :func:`get_hijack`, this variant does not require an invertible operator ``T``.  Instead it asks for a pseudo-inverse ``A†`` of the forward operator ``A``:

    .. math::

        x_{\\text{hijacked}} = x_t - \\lambda \\cdot A^\\dagger\\!(A(x_t) - y_t)

    When ``A†`` is a right inverse (``A · A† = I``), this is algebraically equivalent to :func:`get_hijack`.  In practice, approximate right-inverses such as SIRT preconditioned adjoint (see :class:`~LION.models.score_inverse.sirt_adj.SIRTAdj`) work well empirically even though they are not exact right-inverses.

    Args:
        sde (SimpleForwardSDE): the forward SDE used to compute the transition distribution ``(α_t, β_t)``.
        op (callable): the forward operator ``A`` mapping image-space tensors of shape ``(batch_size, ...)`` to sinogram-space tensors.
        pseudo_inv (callable): an approximate right-inverse ``A†`` of ``op``,mapping sinogram-space tensors back to image space.
        y (torch.Tensor): the clean observed sinogram of shape ``(batch_size, ...)``. Used to form the noisy measurement ``y_t`` at each time step.
        lb (float): hijack weight ``λ``. ``lb=1.0`` gives the strongest correction. Defaults to ``1.0``.
        clean_hijack (bool): if ``True``, use ``y_t = α_t · y`` (no noise injection) instead of the SDE-consistent noisy measurement.  Should be ``False`` to remain consistent with [Song2022].  Defaults to ``False``.
        clamp_factor (float or None): if not ``None``, clamp the hijacked iterate to ``[data_range[0] - clamp_factor · β_t, data_range[1] + clamp_factor · β_t]`` to prevent overflow. Defaults to ``None``.
        data_range (tuple[float, float]): the expected range ``(min, max)`` of clean image values.  Active only when ``clamp_factor`` is not ``None``. Defaults to ``(0.0, 1.0)``.

    Returns:
        callable: a function ``hijack(x, t) -> x_hijacked`` that enforces
        approximate measurement consistency at time step ``t``.
    """
    def hijack(x, t):
        alpha_t, beta_t = sde.transition_dist(y, t)
        if clean_hijack:
            y_t = alpha_t * y
        else:
            y_t = alpha_t * y + beta_t * op(torch.randn_like(x))
        x_hijacked = x - lb * pseudo_inv(op(x) - y_t)
        
        if clamp_factor is not None:
            x_hijacked = torch.clamp(x_hijacked, data_range[0] - clamp_factor * beta_t, data_range[1] + clamp_factor * beta_t)
        return x_hijacked
    return hijack