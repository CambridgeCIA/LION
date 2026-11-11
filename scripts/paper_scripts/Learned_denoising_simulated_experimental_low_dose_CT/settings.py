# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Maximilian B. Kiss, Ander Biguri
# =============================================================================
"""What the noise paper (Kiss et al., "Learned denoising with simulated and
experimental low-dose CT data", Scientific Reports, 2025) trained and tested,
shared by ``train.py`` and ``evaluate.py``: as the code that produced its
results (2024), which the paper's text doesn't always detail.

The 2DeteCT dataset (``LION.data_loaders.deteCT``), with mode 2 (high dose) as
the clean target and one of two noisy inputs:

- ``"simulated"``: mode 2 sinograms with added noise (Poisson and 5% detector
  cross-talk), one I0 = 200 for every slice (``deteCT.noise_parameters(I0=200)``;
  LION's default is one I0 per detector, so it is set here);
- ``"experimental"``: the measured low-dose mode 1 sinograms.

Two tasks: denoising the sinograms (``"sino2sino"``: sinogram to the mode 2
sinogram), and end to end (``"sino2recon"``: sinogram to the mode 2
reconstruction, through FDK). Sinograms are flat- and dark-field corrected and
log-transformed, as the code ran (the paper's text says the networks denoise
before the logarithm). The data are split 80/10/10, the loader's default.
"""

import functools

import torch

from LION.data_loaders.deteCT import deteCT
from LION.models.CNNs.MSDNet import MSDNet
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.operators.ct import fdk
from LION.utils.paths import results_path

from unet import UNet

#: The I0 of the simulated noise: the paper compared 200, 250, 300 and 350 and
#: chose 200, whose reconstructions' PSNR was closest to mode 1's.
SIMULATED_I0 = 200
#: Fraction of the signal of the neighbouring detector pixels (from XCIST).
CROSS_TALK = 0.05

NOISE = ("simulated", "experimental")
TASKS = ("sino2sino", "sino2recon")


def data_parameters(noise, task):
    """The 2DeteCT parameters for one of the paper's settings: ``noise`` is
    "simulated" or "experimental", ``task`` "sino2sino" or "sino2recon"."""
    if noise not in NOISE or task not in TASKS:
        raise ValueError(f"noise must be one of {NOISE} and task one of {TASKS}")
    params = deteCT.default_parameters()
    params.task = task
    params.target_mode = "mode2"
    if noise == "simulated":
        params.input_mode = "mode2"
        params.add_noise = True
        params.noise_params = deteCT.noise_parameters(
            I0=SIMULATED_I0, cross_talk=CROSS_TALK
        )
    else:
        params.input_mode = "mode1"
    return params


def dataset(noise, task, split="train"):
    """The 2DeteCT dataset of one setting, for ``split`` "train", "validation"
    or "test"."""
    return deteCT(split, parameters=data_parameters(noise, task))


def operator_params():
    """2DeteCT's geometry (the FDK of the end-to-end models and of the
    evaluation)."""
    return deteCT.get_default_geometry()


# The models: sinogram denoisers, and networks on the FDK of the sinograms
def _on(model_class, parameters, input_type):
    parameters.model_input_type = input_type
    return model_class(parameters)


#: name: (task, a function making the untrained model)
MODELS = {
    # the network on the measured sinogram
    "UNet": (
        "sino2sino",
        lambda: _on(UNet, UNet.default_parameters(), ModelInputType.SINOGRAM),
    ),
    "MSDNet": (
        "sino2sino",
        lambda: _on(MSDNet, MSDNet.default_parameters(), ModelInputType.SINOGRAM),
    ),
    # the network on the FDK of the sinogram (the solver computes it)
    "FBP+UNet": (
        "sino2recon",
        lambda: _on(UNet, UNet.default_parameters(), ModelInputType.IMAGE),
    ),
    "FBP+MSDNet": (
        "sino2recon",
        lambda: _on(MSDNet, MSDNet.default_parameters(), ModelInputType.IMAGE),
    ),
}

#: The classes defined in this folder that saved models name
ALLOW = ("unet.",)

# The training: Adam (learning rate 1e-4, betas 0.9 and 0.99), batch size 1,
# 100 epochs, the mean squared error to the target; validated every 10 epochs,
# and the model with the lowest validation loss is the one tested
EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.99)
LOSS = torch.nn.MSELoss
VALIDATION_EVERY = 10


def name(model, noise):
    """A trained model's name: e.g. "UNet_simulated"."""
    return f"{model}_{noise}"


def folder():
    """Where the trained models and the evaluations go."""
    return results_path("paper_scripts/noise_paper")


# The evaluation's methods that aren't a trained model alone
def noisy(sinogram, op):
    """The noisy sinogram itself: the noise before denoising (Table 1)."""
    return sinogram


def fbp(sinogram, op):
    """FDK of the sinogram, negative values kept (as the paper's evaluation)."""
    return fdk(sinogram, op, clip=False)


_loaded = {}


def fbp_of_denoised(sinogram, op, model_file):
    """FDK (negative values kept) of the sinogram denoised by the model saved
    in ``model_file``: a sinogram denoiser's reconstruction (Table 2)."""
    if model_file not in _loaded:
        model, _, _ = LIONmodel.load(model_file, allow=ALLOW, supress_warnings=True)
        _loaded[model_file] = model.eval()
    model = _loaded[model_file]
    with torch.no_grad():
        denoised = model(sinogram.to(next(model.parameters()).device))
    return fbp(denoised, op)


def denoised_then_fbp(model_file):
    """``fbp_of_denoised`` for one model, as an evaluation method."""
    return functools.partial(fbp_of_denoised, model_file=str(model_file))
