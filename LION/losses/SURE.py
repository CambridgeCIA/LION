from typing import Optional
import torch.nn as nn
import torch


class SURE(nn.Module):
    def __init__(self, epsilon: Optional[float] = None) -> None:
        super().__init__()
        self.epsilon = epsilon

    def _monte_carlo_divergence(self, f, y: torch.Tensor):
        # y is B, C, W, H
        B, C, W, H = y.shape
        N = C * W * H
        b = torch.normal(0.0, 1.0, (B, N)).to(y.device)

        if self.epsilon is None:
            epsilon = torch.max(y) / 1000
        else:
            epsilon = self.epsilon

        # TODO: Find a better way to do this
        diff = f((y.reshape((B, N)) + epsilon * b).reshape((B, C, W, H))) - f(y)
        return (1 / epsilon) * (torch.sum(b * (diff).reshape((B, N)), dim=1))

    def forward(self, model, noisy):
        return self._monte_carlo_divergence(model, noisy)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    my_sure = SURE().to(device)

    def model(x):
        return x**2

    phantom = torch.rand((2, 1, 2, 2)).to(device)
    print(my_sure(model, phantom))
