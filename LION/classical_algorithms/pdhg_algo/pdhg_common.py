from __future__ import annotations

from collections.abc import Sequence

import torch


class PdhgTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def solver(self):
        raise NotImplementedError(
            "The 'solver' property must be implemented in subclasses of PdhgTorch."
        )


def prepare_lambda(lambda_reg: float | torch.Tensor, dim: Sequence[int], device):
    if not isinstance(lambda_reg, torch.Tensor):
        lambda_reg = torch.tensor(lambda_reg, device=device)
    elif lambda_reg.device != device:
        lambda_reg = lambda_reg.to(device)
    for _ in range(len(dim)):
        lambda_reg = lambda_reg.unsqueeze(-1)
    return lambda_reg


def prepare_param(self, param, requires_grad=False):
    if not isinstance(param, torch.Tensor):
        param = torch.tensor(param, requires_grad=requires_grad)
    if param.device != self.device:
        param = param.to(self.device)
    return param
