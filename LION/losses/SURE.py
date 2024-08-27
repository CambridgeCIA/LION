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

    def _monte_carlo_divergence(self, model, y: torch.Tensor):
        # y is B, C, W, H
        B, C, W, H = y.shape
        N = C * W * H
        b = torch.normal(0.0, 1.0, (B, N)).to(y.device)
        y = y.reshape((B, N))

        def f(x):
            return model(x.reshape((B, C, W, H))).reshape((B, N))

        if self.epsilon is None:
            epsilon = SURE.default_epsilon(y)
        else:
            epsilon = self.epsilon

        diff = f(y + epsilon * b) - f(y)
        return torch.sum((1 / (N * epsilon)) * torch.sum(b * diff))

    def forward(self, model, noisy):
        # print("MSE: ", self.mse(noisy, model(noisy)))
        # # print("Noise_var: ", self.noise_std**2)
        # # print("Divergence term", 2 * (self.noise_std**2) * self._monte_carlo_divergence(model, noisy))
        # print("SURE loss ", torch.mean(
        #     self.mse(noisy, model(noisy))
        #     - self.noise_std**2
        #     + 2 * (self.noise_std**2) * self._monte_carlo_divergence(model, noisy)
        # ))
        # if torch.mean(
        #     self.mse(noisy, model(noisy))
        #     - self.noise_std**2
        #     + 2 * (self.noise_std**2) * self._monte_carlo_divergence(model, noisy)
        # ) < 0:
        #     print("NEGATIVE SURE LOSS")
        #     quit()
        # mean loss over batch
        return torch.abs(
            torch.mean(
                self.mse(noisy, model(noisy))
                - self.noise_std**2
                + 2 * (self.noise_std**2) * self._monte_carlo_divergence(model, noisy)
            )
        )

    @staticmethod
    def default_epsilon(y):
        return torch.max(y) / 1000
