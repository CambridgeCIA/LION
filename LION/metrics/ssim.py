from __future__ import annotations
from skimage.metrics import structural_similarity as skim_ssim
import torch
import torch.nn as nn

from LION.exceptions.exceptions import ShapeMismatchException
from LION.metrics.psnr import _range


class SSIM(nn.Module):
    """SSIM: A wrapper for skimage.metrics.structural_similarity that operates on (potentially) batched pytorch tensors"""

    # a score, not a loss: solvers keep the model with the highest validation value
    higher_is_better = True

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
            data_range: the range of the image values (e.g. 1 for images in [0, 1]).
                Defaults to None: the range (max - min) of each target.

    """

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        reduce: str | None = None,
        batched=True,
        channel_axis: int | None = None,
        data_range: float | None = None,
    ) -> torch.Tensor:
        if x.shape != target.shape:
            raise ShapeMismatchException(
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
