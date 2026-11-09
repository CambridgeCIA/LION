# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Charlie Shoebridge, Ander Biguri, Shanglei Chai
# =============================================================================
from __future__ import annotations
from skimage.metrics import structural_similarity as skim_ssim
import torch
import torch.nn as nn

from LION.metrics.psnr import _range


class SSIM(nn.Module):
    """SSIM, from ``skimage.metrics.structural_similarity``, for (potentially)
    batched pytorch tensors. A score: higher is better."""

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
        channel_axis: int | None = None,
        data_range: float | None = None,
    ) -> torch.Tensor:
        """The SSIM of ``x`` against ``target`` (the same shape): one per
        sample if ``batched`` (the first axis), or their mean if ``reduce`` is
        "mean". ``channel_axis``: the channel axis of colour images (None:
        grayscale, or axis 0 of a 3D sample, LION being channels-first);
        ``data_range``: the range of the image values (e.g. 1 for images in [0,
        1]); None: each target's (max - min)."""
        if x.shape != target.shape:
            raise ValueError(
                f"x (shape {x.shape}) and target (shape {target.shape}) tensors must match to compare with SSIM"
            )
        # Squeeze per sample, not the whole array: squeezing a batch of size 1
        # would drop the batch axis and x_[i] would then index image rows.
        x_ = x.detach().cpu().numpy()
        target_ = target.detach().cpu().numpy()
        # LION is channels-first, so a 3D sample means (C, W, H) -> channel axis 0
        if batched:
            # shape either B, C, W, H, ... or B, W, H, ...
            # if it's not, then that's your fault not mine, you told me it was batched
            vals = torch.empty((x.shape[0]))
            for i in range(x.shape[0]):
                xi, ti = x_[i].squeeze(), target_[i].squeeze()
                sample_axis = channel_axis
                if sample_axis is None:
                    sample_axis = 0 if xi.ndim >= 3 else None
                vals[i] = torch.tensor(
                    skim_ssim(
                        xi,
                        ti,
                        data_range=_range(ti, data_range),
                        channel_axis=sample_axis,
                    )
                )

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
            if channel_axis is None:
                channel_axis = 0 if x_.ndim >= 3 else None
            return torch.tensor(
                skim_ssim(
                    x_,
                    target_,
                    data_range=_range(target_, data_range),
                    channel_axis=channel_axis,
                )
            )
