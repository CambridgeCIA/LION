from __future__ import annotations
from skimage.metrics import structural_similarity as skim_ssim
import torch
import torch.nn as nn

from LION.exceptions.exceptions import ShapeMismatchException


class SSIM(nn.Module):
    """SSIM: A wrapper for skimage.metrics.structural_similarity that operates on (potentially) batched pytorch tensors"""

    def __init__(self):
        super().__init__()

    """
        Args:
            x: torch.Tensor: input image.
            target: torch.Tensor: target image to compare x against.
            batched: whether x and target are batched or not.
            reduce: if batched=False has no effect. Otherwise specifies what operation should be done to aggregate SSIMs along the batch axis. 
                Defaults to None, in which case a tensor of shape (batch size) is returned.
            channel_axis: if working with color images, which axis of image tensor corresponds to channel dimension. 
                Defaults to None, corresponding to a grayscale image.

    """

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        reduce=str | None,
        batched=True,
        channel_axis: int | None = 1,
    ) -> torch.Tensor:
        if x.shape != target.shape:
            raise ShapeMismatchException(
                f"x (shape {x.shape}) and target (shape {target.shape}) tensors must match to compare with SSIM"
            )
        x_ = x.detach().cpu().numpy().squeeze()
        target_ = target.detach().cpu().numpy().squeeze()
        if batched:
            # shape either B, C, W, H, ... or B, W, H, ...
            # if it's not, then that's your fault not mine, you told me it was batched
            vals = torch.empty((x.shape[0]))
            for i in range(x.shape[0]):
                vals[i] = skim_ssim(
                    x_[i],
                    target_[i],
                    data_range=target_[i].max() - target_[i].min(),
                    channel_axis=channel_axis,
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
            return torch.tensor(
                skim_ssim(
                    x,
                    target,
                    data_range=target_.max() - target_.min(),
                    channel_axis=channel_axis,
                )
            )
