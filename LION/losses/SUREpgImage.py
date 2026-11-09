# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Alvaro Exposito Martinez
# =============================================================================
import torch


class SUREpgLoss:
    """PG-SURE: an unbiased estimate of a denoiser's mean squared error under
    mixed Poisson-Gaussian noise (Le Montagner, Angelini and Olivo-Marin, "An
    unbiased risk estimator for image denoising in the presence of mixed
    Poisson-Gaussian noise", IEEE TIP 23(3), 2014), as a self-supervised loss.
    ``zeta``: the Poisson gain; ``sigma2``: the Gaussian variance; ``eps1``,
    ``eps2``: the Monte Carlo perturbations of the first and second
    derivatives; ``kappa``: the asymmetry of the second's random perturbation.
    """

    def __init__(
        self,
        zeta: float,
        sigma2: float,
        eps1: float = 1e-3,
        eps2: float = 1e-3,
        kappa: float = 1.0,
    ):
        self.zeta = zeta
        self.sigma2 = sigma2
        self.eps1 = eps1
        self.eps2 = eps2
        self.kappa = kappa

        self.p = (1 / 2) * (1 + self.kappa / (self.kappa**2 + 4) ** 0.5)
        self.q = 1 - self.p
        self.a = (self.q / self.p) ** 0.5
        self.b = (self.p / self.q) ** 0.5

    def __call__(self, model, y):
        """The loss of the denoiser ``model`` on the noisy images ``y`` (B, C,
        H, W), per pixel."""
        B = y.shape[0]
        N_per_img = y.shape[1] * y.shape[2] * y.shape[3]

        fy = model(y)
        loss = (
            ((fy - y) ** 2).sum(dim=(1, 2, 3))
            - self.zeta * y.sum(dim=(1, 2, 3))
            - self.sigma2 * N_per_img
        )

        # 1st derivative MC
        delta1 = torch.randn_like(y)
        fy_perturbated = model(y + self.eps1 * delta1)

        u = self.zeta * y + self.sigma2
        mc1 = (delta1 * u * (fy_perturbated - fy)).sum(dim=(1, 2, 3))
        loss += 2.0 * mc1 / self.eps1

        # 2nd derivative MC
        u_rand = torch.rand_like(y)
        delta2 = torch.where(
            u_rand < self.p, -self.a * torch.ones_like(y), +self.b * torch.ones_like(y)
        )

        fy_plus = model(y + self.eps2 * delta2)
        fy_minus = model(y - self.eps2 * delta2)

        mc2 = (delta2 * (fy_plus - 2 * fy + fy_minus)).sum(dim=(1, 2, 3))
        loss -= (2 * self.sigma2 * self.zeta / (self.eps2**2 * self.kappa)) * mc2

        return loss.mean() / N_per_img
