from skimage.metrics import peak_signal_noise_ratio as skim_psnr
import torch
import torch.nn as nn

from LION.exceptions.exceptions import ShapeMismatchException


class PSNR(nn.Module):
    """PSNR: A wrapper for skimage.metrics.peak_signal_noise_ratio that operates on (potentially) batched pytorch tensors"""

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
        self, x: torch.Tensor, target: torch.Tensor, reduce=str | None, batched=True
    ) -> torch.Tensor:
        if x.shape != target.shape:
            raise ShapeMismatchException(
                f"x (shape {x.shape}) and target (shape {target.shape}) tensors must match to compare with SSIM"
            )
        x_ = x.detach().cpu().numpy().squeeze()
        target_ = target.detach().cpu().numpy().squeeze()
        if batched:
            vals = torch.empty((x.shape[0]))
            for i in range(x.shape[0]):
                vals[i] = skim_psnr(
                    target_[i], x_[i], data_range=target_[i].max() - target_[i].min()
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
                skim_psnr(x, target, data_range=target_.max() - target_.min())
            )
