# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri, Max Kiss
# =============================================================================
"""The data of Kiss et al., "Learned denoising with simulated and experimental
low-dose CT data" (Scientific Reports, 2025): work in progress, see README.md.

The 2DeteCT dataset (``LION.data_loaders.deteCT``), with mode 2 (high dose) as
the clean target and one of two noisy inputs:

- ``"simulated"``: mode 2 sinograms with added noise (Poisson and 5% detector
  cross-talk), one I0 = 200 for every slice, as in the paper. This is
  ``deteCT.noise_parameters(I0=200)``; LION's default is now one I0 per
  detector, so it is set here explicitly.
- ``"experimental"``: the measured low-dose mode 1 sinograms.

The tasks of the paper are ``"sino2sino"`` (sinogram denoising) and
``"sino2recon"`` (end to end, with an FBP layer). The paper splits the data
80/10/10 (training/validation/testing), the loader's default.
"""

from LION.data_loaders.deteCT import deteCT

#: The I0 of the simulated noise: the paper compared 200, 250, 300 and 350 and
#: chose 200, whose reconstructions' PSNR was closest to mode 1's.
SIMULATED_I0 = 200
#: Fraction of the signal of the neighbouring detector pixels (from XCIST).
CROSS_TALK = 0.05

NOISE = ("simulated", "experimental")
TASKS = ("sino2sino", "sino2recon")


def data_parameters(noise: str, task: str):
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


def dataset(noise: str, task: str, mode: str = "train") -> deteCT:
    """The 2DeteCT dataset of one setting, for ``mode`` "train", "validation"
    or "test". The paper tests every model on both test sets."""
    return deteCT(mode, parameters=data_parameters(noise, task))
