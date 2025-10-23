"""Conjugate Gradient Solver"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass
class ConjugateGradient:
    max_iter: int
    tol: float

    def solve(
            self, matmul_closure: Callable[[Tensor], Tensor], d: Tensor, x0: Tensor
    ) -> Tensor:
        x = x0.clone()
        r = d - matmul_closure(x)
        d = r.clone()
        rr = torch.sum(r**2)

        for _ in range(self.max_iter):
            z = matmul_closure(d)

            dz = torch.sum(d * z)
            # Check for breakdown
            if abs(dz) < 1e-14:
                break
            alpha = rr / dz
            x += alpha * d
            r -= alpha * z

            if torch.norm(r) / torch.norm(d) < self.tol:
                break

            rr_next = torch.sum(r**2)
            beta = rr_next / rr
            d = r + beta * d
            rr = rr_next

        return x
