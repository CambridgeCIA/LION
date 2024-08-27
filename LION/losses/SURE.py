from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch


class SURE(nn.Module):
    def __init__(self, noise_std: float, epsilon: Optional[float] = None) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.noise_std = noise_std
        self.mse = nn.MSELoss()

    def _monte_carlo_divergence(self, f, y: torch.Tensor):
        # y is B, C, W, H
        B, C, W, H = y.shape
        N = C * W * H
        b = torch.normal(0.0, 1.0, (B, N)).to(y.device)

        if self.epsilon is None:
            epsilon = SURE.default_epsilon(y)
        else:
            epsilon = self.epsilon

        # TODO: Find a better way to do this
        diff = f((y.reshape((B, N)) + epsilon * b).reshape((B, C, W, H))) - f(y)
        return (1 / epsilon) * (1 / N) * (torch.sum(b * (diff).reshape((B, N)), dim=1))

    def forward(self, model, noisy):
        # mean loss over batch
        return torch.mean(
            self.mse(noisy, model(noisy))
            - self.noise_std**2
            + 2 * (self.noise_std**2) * self._monte_carlo_divergence(model, noisy)
        )

    @staticmethod
    def default_epsilon(y):
        return torch.max(y) / 1000
