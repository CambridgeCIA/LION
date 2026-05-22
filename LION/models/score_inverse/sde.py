"""
This module implements the abstract base class for Stochastic Differential Equations (SDEs), their reverse-time counterparts, and an implementation of the Variance Exploding SDE (VESDE).

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
import numpy as np
import math

class SDE(ABC):
    """
    Forward SDE on time interval [0, 1].
    """
    @abstractmethod
    def drift_coeff(self, x: torch.Tensor, t: torch.Tensor):
        """
        Compute the drift coefficient of the SDE.

        Args:
            x: torch.Tensor of shape (batch_size, ...).
            t: torch.Tensor of shape (batch_size,).

        Returns:
            torch.Tensor of shape (batch_size, ...) representing the drift coefficient.
        """
        pass
    
    @abstractmethod
    def diffusion_coeff(self, x: torch.Tensor, t: torch.Tensor):
        """
        Compute the diffusion coefficient of the SDE.

        Args:
            x: torch.Tensor of shape (batch_size, ...).
            t: torch.Tensor of shape (batch_size,).

        Returns:
            torch.Tensor of shape (batch_size, ...) representing the diffusion coefficient.
        """
        pass

class SimpleForwardSDE(SDE):
    """
    Forward linear SDE that defines the forward diffusion process. Equation (1) in [Song2022].
    The drift coefficients and diffusion coefficients are scalar functions of time t.
    """
    @abstractmethod
    def f(self, t: torch.Tensor):
        """
        The coefficient of x_t in the drift term in the forward SDE, which is a scalar function of time t.
        
        Args:
            t: torch.Tensor of shape (batch_size,).
        Returns:
            torch.Tensor of shape (batch_size,) representing the coefficient of x_t in the drift term
        """
        pass
    
    @abstractmethod
    def g(self, t: torch.Tensor):
        """
        The diffusion term in the forward SDE, which is a scalar function of time t.
        
        Args:
            t: torch.Tensor of shape (batch_size,).
        Returns:
            torch.Tensor of shape (batch_size,) representing the diffusion term in the forward SDE.
        """
        pass
    
    def drift_coeff(self, x, t):
        return self.f(t).view(-1, *[1] * (len(x.shape) - 1)) * x
    
    def diffusion_coeff(self, x, t):
        return self.g(t).view(-1, *[1] * (len(x.shape) - 1)).to(x.device)

    @property
    @abstractmethod
    def noise_std(self):
        """
        The std of the isotropic Gaussian distribution assumed to be the approximated marginal distribution of the SDE at time t = 1.

        Returns:
            float representing the std of the noise distribution.
        """
        pass
    
    def sample_noise(self, shape: torch.Size):
        """
        Sample noise from the isotropic Gaussian distribution assumed to be the approximated marginal distribution of the SDE at time t = 1.

        Args:
            shape: torch.Size representing the shape of the noise to be sampled.

        Returns:
            torch.Tensor of shape `shape` representing the sampled noise.
        """
        return self.noise_std * torch.randn(shape)
    
    @abstractmethod
    def alpha(self, t: torch.Tensor):
        """
        Compute the alpha coefficient of the transition distribution p_0t(. | x) ~ N(alpha(t) * x, beta^2(t) I).

        Args:
            t: torch.Tensor of shape (batch_size,).

        Returns:
            torch.Tensor of shape (batch_size,) representing the alpha coefficient.
        """
        pass
    
    @abstractmethod
    def beta(self, t: torch.Tensor):
        """
        Compute the beta coefficient of the transition distribution p_0t(. | x) ~ N(alpha(t) * x, beta^2(t) I).

        Args:
            t: torch.Tensor of shape (batch_size,).

        Returns:
            torch.Tensor of shape (batch_size,) representing the beta coefficient.
        """
        pass
    
    def transition_dist(self, x: torch.Tensor, t: torch.Tensor):
        """
        Jointly compute alpha and beta coefficients of the transition distribution with shape (batch_size, 1, ..., 1), suitable for broadcasting.

        Args:
            x: torch.Tensor of shape (batch_size, ...).
            t: torch.Tensor of shape (batch_size,).
        """
        alpha_t = self.alpha(t).view(-1, *[1] * (len(x.shape) - 1)).to(x.device)
        beta_t = self.beta(t).view(-1, *[1] * (len(x.shape) - 1)).to(x.device)
        return alpha_t, beta_t

    def sample_transition(self, x: torch.Tensor, t: torch.Tensor):
        """
        Sample from the transition distribution p_0t(. | x) ~ N(alpha(t) * x, beta^2(t) I).

        Args:
            x: torch.Tensor of shape (batch_size, ...).
            t: torch.Tensor of shape (batch_size,).
        
        Returns:
            torch.Tensor of shape (batch_size, ...) representing the sampled transition.
        """
        alpha_t, beta_t = self.transition_dist(x, t)
        return alpha_t * x + beta_t * torch.randn_like(x)
    
    def score_transition(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """
        Compute Grad_y log p_0t(y | x), where p_0t(. | x) ~ N(alpha(t) * x, beta^2(t) I).

        Args:
            x: torch.Tensor of shape (batch_size, ...).
            y: torch.Tensor of shape (batch_size, ...).
            t: torch.Tensor of shape (batch_size,).
        """
        alpha_t, beta_t = self.transition_dist(x, t)
        return -(y - alpha_t * x) / (beta_t ** 2)

class SimpleReverseSDE(SDE):
    """
    Reverse-time SDE from a given forward SDE and score function. Equation (2) and (3) in [Song2022].
    """
    def __init__(self, forward_sde: SimpleForwardSDE, score_fn):
        """
        Args:
            forward_sde (SimpleForwardSDE): the forward SDE that defines the forward diffusion process.
            score_fn (callable): a function that takes in (x, t) and returns the score function evaluated at (x, t).
        """
        super().__init__()
        self.forward_sde = forward_sde
        self.score_fn = score_fn

    def drift_coeff(self, x: torch.Tensor, t: torch.Tensor):
        return self.forward_sde.drift_coeff(x, t) - self.forward_sde.diffusion_coeff(x, t)**2 * self.score_fn(x, t)
    
    def diffusion_coeff(self, x: torch.Tensor, t: torch.Tensor):
        return self.forward_sde.diffusion_coeff(x, t)

class VESDE(SimpleForwardSDE):
    """
    Variance Exploding SDE. Equation (30) in [Song2021].
    """
    def __init__(self, sigma_min: float, sigma_max: float):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def f(self, t: torch.Tensor):
        return torch.zeros_like(t)

    def g(self, t: torch.Tensor):
        return self.sigma_min * (self.sigma_max / self.sigma_min)**t * math.sqrt(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)))
    
    @property
    def noise_std(self):
        return self.sigma_max
    
    def alpha(self, t: torch.Tensor):
        return torch.ones_like(t)
    
    def beta(self, t: torch.Tensor):
        return self.sigma_min * (self.sigma_max / self.sigma_min)**t