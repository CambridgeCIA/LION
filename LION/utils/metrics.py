# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Hong Ye Tan
# Modifications: -
# Created: 11 July 2024
# =============================================================================

from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
import numpy as np
import torch

####
# All functions here are in order preds, target
####

# wrapper around SSIM and PSNR
def SSIM(pred, gt):
    return structural_similarity_index_measure(pred, gt, reduction="none")


def avgSSIM(pred, gt):
    return structural_similarity_index_measure(pred, gt)


def PSNR(pred, gt, data_range=1.0):
    # x_true: reference image
    return peak_signal_noise_ratio(
        pred, gt, reduction="none", data_range=data_range, dim=[1, 2, 3]
    )


def avgPSNR(pred, gt):
    return peak_signal_noise_ratio(pred, gt, data_range=1.0)
