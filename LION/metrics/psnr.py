# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Charlie Shoebridge, Ander Biguri
# =============================================================================
from __future__ import annotations

import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as skim_psnr


def _range(target, data_range):
    """data_range if given, else the range of the target."""
    return target.max() - target.min() if data_range is None else data_range


class PSNR(nn.Module):
    """PSNR, from ``skimage.metrics.peak_signal_noise_ratio``, for
    (potentially) batched pytorch tensors. A score: higher is better."""

    # a score, not a loss: solvers keep the model with the highest validation value
    higher_is_better = True

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        reduce: str | None = None,
        batched=True,
        data_range: float | None = None,
    ) -> torch.Tensor:
        """The PSNR of ``x`` against ``target`` (the same shape): one per
        sample if ``batched`` (the first axis), or their mean if ``reduce`` is
        "mean". ``data_range``: the range of the image values (e.g. 1 for
        images in [0, 1]); None: each target's (max - min)."""
        if x.shape != target.shape:
            raise ValueError(
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
                vals[i] = skim_psnr(ti, xi, data_range=_range(ti, data_range))

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
                skim_psnr(target_, x_, data_range=_range(target_, data_range))
            )
