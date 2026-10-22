from __future__ import annotations

import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as skim_psnr

from LION.exceptions.exceptions import ShapeMismatchException


class PSNR(nn.Module):
    """PSNR: A wrapper for skimage.metrics.peak_signal_noise_ratio that operates on (potentially) batched pytorch tensors"""

    # a score, not a loss: solvers keep the model with the highest validation value
    higher_is_better = True

    def __init__(self):
        super().__init__()

    """
        Args:
            x: torch.Tensor: input image
            target: torch.Tensor: target image to compare x against
            batched: whether x and target are batched or not
            reduce: if batched=False has no effect. Otherwise specifies what operation should be done to aggregate SSIMs along the batch axis.
                Defaults to None, in which case a tensor of shape (batch size) is returned

    """

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        reduce: str | None = None,
        batched=True,
    ) -> torch.Tensor:
        if x.shape != target.shape:
            raise ShapeMismatchException(
                f"x (shape {x.shape}) and target (shape {target.shape}) tensors must match to compare with PSNR"
            )
        # Squeeze per sample, not the whole array: squeezing a batch of size 1
        # would drop the batch axis and x_[i] would then index image rows.
        x_ = x.detach().cpu().numpy()
        target_ = target.detach().cpu().numpy()
        if batched:
            vals = torch.empty(x.shape[0])
            for i in range(x.shape[0]):
                xi, ti = x_[i].squeeze(), target_[i].squeeze()
                vals[i] = skim_psnr(ti, xi, data_range=ti.max() - ti.min())

            if reduce is None:
                return vals
            elif reduce == "mean":
                return torch.mean(vals)
            else:
                raise ValueError(
                    f"expected one of 'mean' or None for parameter 'reduce', got {reduce}"
                )
        else:
            x_, target_ = x_.squeeze(), target_.squeeze()
            # skimage expects (image_true, image_test)
            return torch.tensor(
                skim_psnr(target_, x_, data_range=target_.max() - target_.min())
            )
