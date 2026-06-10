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
        reduce: str | None = None,
        batched=True,
        channel_axis: int | None = 1,
        data_range: float | None = None,
    ) -> torch.Tensor:
        if x.shape != target.shape:
            raise ShapeMismatchException(
                f"x (shape {x.shape}) and target (shape {target.shape}) tensors must match to compare with SSIM"
            )
            
        if batched:
            x_ = x.detach().cpu().numpy()
            target_ = target.detach().cpu().numpy()
            vals = torch.empty((x.shape[0]))
            for i in range(x.shape[0]):
                xi = x_[i].squeeze()
                ti = target_[i].squeeze()
                
                if xi.shape != x_[i].shape and channel_axis is not None:
                    import warnings
                    warnings.warn(
                        f"SSIM warning: Squeezing changed batch element shape from {x_[i].shape} to {xi.shape} "
                        f"while channel_axis was {channel_axis}. The channel_axis might be handled unpredictably. "
                        f"It is highly recommended to perform squeezing (for non-batch dimensions only) yourself "
                        f"before applying SSIM, and pass channel_axis=None.",
                        UserWarning,
                        stacklevel=2
                    )
                
                curr_channel_axis = channel_axis
                if len(xi.shape) == 2:
                    curr_channel_axis = None
                elif curr_channel_axis is not None and curr_channel_axis >= 0:
                    curr_channel_axis -= 1
                    
                dr = data_range if data_range is not None else (ti.max() - ti.min())
                vals[i] = float(skim_ssim(
                    xi,
                    ti,
                    data_range=dr,
                    channel_axis=curr_channel_axis,
                ))

            if reduce is None:
                return vals
            elif reduce == "mean":
                return torch.mean(vals)
            else:
                raise ValueError(
                    f"expected one of 'mean' or None for parameter 'reduce', got {reduce}"
                )
        else:
            x_ = x.detach().cpu().numpy().squeeze()
            target_ = target.detach().cpu().numpy().squeeze()
            
            if x_.shape != tuple(x.shape) and channel_axis is not None:
                import warnings
                warnings.warn(
                    f"SSIM warning: Squeezing changed shape from {x.shape} to {x_.shape} "
                    f"while channel_axis was {channel_axis}. The channel_axis might be handled unpredictably. "
                    f"It is highly recommended to perform squeezing yourself before applying SSIM, "
                    f"and pass channel_axis=None.",
                    UserWarning,
                    stacklevel=2
                )
                
            curr_channel_axis = channel_axis
            if len(x_.shape) == 2:
                curr_channel_axis = None
                
            dr = data_range if data_range is not None else (target_.max() - target_.min())
            return torch.tensor(
                skim_ssim(
                    x_,
                    target_,
                    data_range=dr,
                    channel_axis=curr_channel_axis,
                )
            )
