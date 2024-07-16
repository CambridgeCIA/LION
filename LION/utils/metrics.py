# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Hong Ye Tan
# Modifications: -
# Created: 11 July 2024
# =============================================================================

from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch

def SSIM(x, y):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


def PSNR(x_test, x_true):
    # x_true: reference image
    mse = torch.mean( (x_test - x_true) ** 2, dim=[1,2,3] )
    PIXEL_MAX = torch.amax(x_true, dim=[1,2,3])
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def avgPSNR(x_true, x_test):
    return torch.mean(PSNR(x_true, x_test))