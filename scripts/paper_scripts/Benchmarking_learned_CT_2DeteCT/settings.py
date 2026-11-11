# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Maximilian B. Kiss, Ander Biguri
# =============================================================================
"""What the 2DeteCT benchmark paper (Kiss et al., 2025) trained and tested,
for the methods whose code is in LION: shared by ``train.py`` and
``evaluate.py``, as the code that produced the paper's results (2024), which
the paper's text doesn't always detail."""

import functools

import torch

from LION.experiments import deteCT_benchmark
from LION.models.CNNs.dncnn import DnCNN
from LION.models.CNNs.MSDNet import MSDNet
from LION.models.CNNs.UNets.Unet import UNet
from LION.models.iterative_unrolled.LG import LG
from LION.models.iterative_unrolled.LPD import LPD
from LION.operators.ct import fdk, tv_min
from LION.operators.ct._batching import batched_reconstruction
from LION.utils.paths import results_path
from LION.utils.tomosipo_import import import_tomosipo

# The paper's nine problems (its Tables 5-7), as LION's experiments
EXPERIMENTS = {
    "full": deteCT_benchmark.FullDataCTRecon,
    "limited_120": deteCT_benchmark.LimitedAngle120CTRecon,
    "limited_90": deteCT_benchmark.LimitedAngle90CTRecon,
    "limited_60": deteCT_benchmark.LimitedAngle60CTRecon,
    "sparse_360": deteCT_benchmark.SparseAngle360CTRecon,
    "sparse_120": deteCT_benchmark.SparseAngle120CTRecon,
    "sparse_60": deteCT_benchmark.SparseAngle60CTRecon,
    "low_dose": deteCT_benchmark.LowDoseCTRecon,
    "beam_hardening": deteCT_benchmark.BeamHardeningCTRecon,
}


# The learned methods, with the parameters they were trained with
def unet_parameters():
    """FBP+U-Net: LION's U-Net, on FDK (the 2024 class wasn't kept)."""
    return UNet.default_parameters()


def msdnet_parameters():
    """FBP+MS-D Net: LION's MS-D network (100 layers of width 1, dilations 1
    to 10, as the 2024 default), on FDK."""
    return MSDNet.default_parameters()


def dncnn_parameters():
    """FBP+DnCNN: 20 layers of 64 channels, bias-free, with batch
    normalisation and leaky ReLUs, learning the residual (FBPDnCNN, 2024)."""
    parameters = DnCNN.default_parameters()
    parameters.bias_free = True
    return parameters


def lg_parameters():
    """LG: 5 iterations, blocks [8, 32, 32, 6] (LION's default, as in 2024)."""
    return LG.default_parameters()


def lpd_parameters():
    """LPD as trained in 2024: 5 iterations of 3-layer blocks of 32 channels,
    no biases, fixed step sizes, the first primal channel projected in the
    dual step."""
    parameters = LPD.default_parameters()
    parameters.n_iters = 5
    parameters.conv_bias = False
    parameters.dual_step_primal_channel = 0
    return parameters


# name: (model class, its parameters, uses the operator, learning rate)
MODELS = {
    "FBP+UNet": (UNet, unet_parameters, False, 1e-4),
    "FBP+MSDNet": (MSDNet, msdnet_parameters, False, 1e-4),
    "FBP+DnCNN": (DnCNN, dncnn_parameters, False, 1e-4),
    "LG": (LG, lg_parameters, True, 1e-5),
    "LPD": (LPD, lpd_parameters, True, 1e-4),
}

# The training: Adam (betas 0.9 and 0.99), batch size 1, 100 epochs, the mean
# squared error to the target; validated every 10 epochs, and the model with
# the lowest validation loss is the one tested
EPOCHS = 100
BATCH_SIZE = 1
BETAS = (0.9, 0.99)
LOSS = torch.nn.MSELoss
VALIDATION_EVERY = 10


# The classical methods
def agd(sinogram, op):
    """AGD: Nesterov's accelerated gradient descent on the least-squares
    problem, 100 iterations (ts_algorithms' ``nag_ls``)."""
    nag_ls = import_tomosipo("ts_algorithms", "agd").nag_ls
    return batched_reconstruction(
        lambda op, y, i: nag_ls(op, y, num_iterations=100), sinogram, op
    )


# ChP: TV minimisation by Chambolle-Pock (ts_algorithms' ``tv_min2d``, 500
# iterations), with the regularisation weight chosen for each problem
TV_WEIGHTS = {
    "full": 1e-10,
    "limited_120": 1e-8,
    "limited_90": 1e-7,
    "limited_60": 1e-8,
    "sparse_360": 1e-6,
    "sparse_120": 5e-6,
    "sparse_60": 5e-6,
    "low_dose": 5e-6,
    "beam_hardening": 1e-8,
}


def classical_methods(experiment_name):
    """``{name: function(sinogram, operator)}``: FBP, AGD and ChP for one
    problem."""
    return {
        # ts_algorithms' FDK, without setting negative values to zero
        "FBP": functools.partial(fdk, clip=False),
        "AGD": agd,
        "ChP": functools.partial(tv_min, lam=TV_WEIGHTS[experiment_name], verbose=0),
    }


def folder(experiment_name):
    """Where the models trained on one problem, and their evaluation, go."""
    return results_path(f"paper_scripts/2detect_benchmark/{experiment_name}")
