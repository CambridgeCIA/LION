# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Maximilian B. Kiss, Ferdia Sherry, Zakhar Shumaylov, Ander Biguri
# =============================================================================
"""What the 2DeteCT benchmark paper (Kiss et al., 2025) trained and tested:
shared by ``train.py``, ``train_denoisers.py``, ``train_regularisers.py`` and
``evaluate.py``, as the code that produced the paper's results (2024), which
the paper's text doesn't always detail."""

import functools

import torch

from LION.experiments import deteCT_benchmark
from LION.models.CNNs.dncnn import DnCNN
from LION.models.CNNs.drunet import DRUNet
from LION.models.CNNs.MSDNet import MSDNet
from LION.models.CNNs.UNets.Unet import UNet
from LION.models.iterative_unrolled.LG import LG
from LION.models.iterative_unrolled.LPD import LPD
from LION.models.learned_regularizer.AR import AR
from LION.models.learned_regularizer.TDV import TDV
from LION.models.PnP.gradient_step_denoiser import GSD
from LION.operators.ct import fdk, tv_min
from LION.operators.ct._batching import batched_reconstruction
from LION.reconstructors.PnP import PnP
from LION.utils.math import power_method
from LION.utils.paths import results_path
from LION.utils.tomosipo_import import import_tomosipo

from learned_regularisers import ACR2024, GradientDescent2024

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
    """FBP+U-Net: LION's U-Net (the paper's), on FDK."""
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
    """LG: 5 iterations of blocks of [7, 32, 32, 6] channels (LION's
    default)."""
    return LG.default_parameters()


def lgtv_parameters():
    """LGTV: LG whose blocks also get a TV gradient, [8, 32, 32, 6]."""
    return LG.lgtv_parameters()


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
    "LGTV": (LG, lgtv_parameters, True, 1e-5),
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


# The Plug-and-Play methods (Ferdia Sherry's code of 2024): a Gaussian
# denoiser trained on the targets, then used in a reconstruction. Each
# denoiser is trained at each noise level (the paper's); for each problem, the
# one whose PnP reconstructions are best on the validation set is tested.
NOISE_LEVELS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07]

# The denoisers see the targets scaled to [0, 1] by the training targets'
# maximum (the 2024 code's normalisation.json); the noise is added after
NORMALISATION = (0.0, 0.023659633472561836)


def pnp_dncnn_parameters():
    """DnCNN-PnP's denoiser: 20 layers of 64 channels, with biases, batch
    normalisation and leaky ReLUs, learning the residual (the 2024 script's
    defaults)."""
    parameters = DnCNN.default_parameters()
    parameters.bias_free = False
    return parameters


def pnp_drunet_parameters():
    """DRUNet-PnP's denoiser: 64 channels, 4 blocks per scale, with biases,
    leaky ReLUs, without the noise level as an input (the 2024 script's
    defaults)."""
    parameters = DRUNet.default_parameters()
    parameters.bias_free = False
    return parameters


def gs_parameters():
    """GS-PnP's denoiser, a gradient-step denoiser x - grad g(x), g(x) = ||x -
    N(x)||² / 2: N a DRUNet scaled down to 32 channels and 2 blocks per scale
    (the 2024 class's default; the paper: "scaling down ... the backbone
    DRUNet"), with ELUs (g must be smooth)."""
    parameters = GSD.default_parameters()
    parameters.backbone_params.act = "elu"
    return parameters


# name: (model class, its parameters)
DENOISERS = {
    "DnCNN-PnP": (DnCNN, pnp_dncnn_parameters),
    "DRUNet-PnP": (DRUNet, pnp_drunet_parameters),
    "GS-PnP": (GSD, gs_parameters),
}

# The denoisers' training: Adam (betas 0.9 and 0.9), learning rate 1e-4, 25
# epochs, one image at a time as 5 random patches of 256 x 256, half of them
# with a random rectangle erased; the mean squared error to the clean patch.
# The model with the best validation is kept.
DENOISER_EPOCHS = 25
DENOISER_LEARNING_RATE = 1e-4
DENOISER_BETAS = (0.9, 0.9)
DENOISER_PATCHES = (5, 256)

# The reconstructions: 100 iterations from zero, with step size 1 / ||A||²
PNP_ITERATIONS = 100


def zeros(sinogram, op):
    """The PnP reconstructions' starting image: zero."""
    return torch.zeros(op.domain_shape, device=sinogram.device)


def pnp(name, denoiser, op):
    """The PnP reconstruction of ``name`` with a trained ``denoiser``: for
    DnCNN-PnP and DRUNet-PnP, forward-backward splitting, x = D(x - step
    Aᵀ(A x - y)) (LION's ``PnP``); for GS-PnP, gradient descent on ||A x -
    y||² / 2 + g(x), then one denoising (``gs_pnp``)."""
    if name == "GS-PnP":
        return functools.partial(gs_pnp, denoiser=denoiser)
    parameters = PnP.default_parameters("FBS")
    parameters.max_iter = PNP_ITERATIONS
    parameters.initial_reconstruction = zeros
    return PnP(op, denoiser, parameters)


_norms = {}  # ||A|| of each operator, computed once


def gs_pnp(sinogram, op, denoiser, iterations=PNP_ITERATIONS):
    """GS-PnP as the 2024 code ran it: gradient descent on ||A x - y||² / 2 +
    g(x) from zero, step size 1 / ||A||², then ``denoiser`` applied once.
    In the descent, g is evaluated on the images as they are, not scaled to
    [0, 1]; the final denoising scales them."""
    if id(op) not in _norms:
        _norms[id(op)] = power_method(op, device=sinogram.device).item()
    step_size = 1.0 / _norms[id(op)] ** 2
    images = []
    for y in sinogram:
        x = zeros(y, op)
        for _ in range(iterations):
            data_gradient = op.adjoint(op(x) - y)
            _, regulariser_gradient = denoiser.grad(x.unsqueeze(0))
            x = (x - step_size * (data_gradient + regulariser_gradient[0])).detach()
        x = x.unsqueeze(0)
        with torch.no_grad():
            x = denoiser.unnormalise(denoiser(denoiser.normalise(x)), x)
        images.append(x[0])
    return torch.stack(images)


# The learned regularisers and TDV (Zakhar Shumaylov's code of 2024): AR and
# ACR, regularisers trained adversarially (clean images against FDK
# reconstructions) then used in a variational reconstruction; TDV, a
# variational network trained end to end.


def tdv_parameters():
    """TDV: 10 steps, a regulariser of 5 macro blocks of 5 scales, 16
    channels; reflection padding, as LION had it in 2024."""
    parameters = TDV.default_parameters()
    parameters.padding = "reflect"
    return parameters


#: The classes defined in this folder that saved models and records name
ALLOW = ("learned_regularisers.", "settings.")

# name: (model class, its parameters, batch size)
REGULARISERS = {
    "AR": (AR, AR.default_parameters, 2),
    "ACR": (ACR2024, ACR2024.default_parameters, 5),
    "TDV": (TDV, tdv_parameters, 1),
}

# Their training: Adam (betas 0.9 and 0.99), learning rate 1e-4, 25 epochs,
# validated and checkpointed every epoch. AR and ACR: the WGAN loss with
# gradient penalty (weight 0.1), validated by the mean squared error of their
# reconstructions of the first 50 validation slices; their images scaled by
# the minimum and maximum of the first 100 training targets. TDV: the mean
# squared error to the target, validated on the whole validation set.
REGULARISER_EPOCHS = 25
REGULARISER_LEARNING_RATE = 1e-4
REGULARISER_BETAS = (0.9, 0.99)
REGULARISER_VALIDATION_SLICES = 50
NORMALISATION_SLICES = 100

# The three models of each tested (the 2024 code tested them, and each
# problem's table used one): at the lowest validation loss, at the lowest
# training loss, and after the last epoch
CHECKPOINTS = {"min val": "_best", "min train": "_mintrain", "last": ""}


def reconstruction(name, op, alpha, model=None):
    """AR's or ACR's reconstruction with weight ``alpha`` (with the trained
    regulariser ``model``, or the one being trained): 150 steps from
    FDK, momentum 0.5, step size 0.2 / ||A||² (AR, which stops when it is
    below 1e-3 / ||A||²) or 1 / ||A||² (ACR), backtracking 0.95
    (``GradientDescent2024``)."""
    norm = power_method(op).item()
    parameters = GradientDescent2024.default_parameters()
    parameters.alpha = float(alpha)
    parameters.n_steps = 150
    parameters.step_size = (0.2 if name == "AR" else 1.0) / norm**2
    parameters.min_step = 1e-3 / norm**2 if name == "AR" else None
    return GradientDescent2024(op, model, parameters)


# The weight of AR's and ACR's regulariser, α, is estimated on the first 50
# validation slices (``estimate_alpha``), and divided by 10 for beam hardening
BEAM_HARDENING_ALPHA_FACTOR = 0.1


def denoiser_name(name, noise_level):
    """A denoiser's file name, e.g. ``DnCNN-PnP_sigma0_005`` (no dot: it would
    be read as a suffix)."""
    return f"{name}_sigma{noise_level:.3f}".replace(".", "_")


def denoiser_folder():
    """Where the denoisers go: they are trained once, for every problem."""
    return results_path("paper_scripts/2detect_benchmark/denoisers")


def folder(experiment_name):
    """Where the models trained on one problem, and their evaluation, go."""
    return results_path(f"paper_scripts/2detect_benchmark/{experiment_name}")
