# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Reconstructors
==============

A **reconstructor** turns measurements into images with a trained model: the
model inside an algorithm (a denoiser as a prior, a learned regulariser), or
the model alone. It is made with everything that decides its result (the
operator, the model and the algorithm's settings), then run on data:

.. code-block:: python

    reconstructor = PnP(operator, model, parameters)
    images = reconstructor.reconstruct(measurements)

So a reconstructor with its settings is one method: evaluations score it,
records name it, and the same settings give the same result. This tutorial
puts LION's side by side, then saves one. Reconstructors are optional: a
model can always be called directly (:doc:`d05_supervised_training`).

Previous: :doc:`d09_saving_and_records`. Next: :doc:`d11_evaluation`.

Needs a GPU and the processed LIDC-IDRI data in ``LION_DATA_PATH``; the PnP
part downloads the authors' trained DRUNet (130 MB).

.. note::

   The models trained in these tutorials are toys: a few epochs on a few
   slices, to show how LION is used, so their results are poor (often worse
   than FDK). Trained as their papers train them (many epochs, all the data),
   they do what the papers report.
"""

# %%
# Classical reconstructions
# -------------------------
# The simplest are functions of the measurements and the operator (or its
# parameters): ``fdk``, ``sirt``, ``tv_min`` for CT, and FISTA, conjugate
# gradient and SPGL1 for any operator (``LION.classical_algorithms``).
# Evaluations take such functions as methods too (:doc:`d11_evaluation`).

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.metrics.psnr import PSNR
from LION.operators.ct import fdk, sirt

experiment = ct_experiments.LowDoseCTRecon()
geometry = experiment.operator_params
sinogram, image = experiment.get_dataset("validation")[0]

reconstructions = {
    "FDK": fdk(sinogram, geometry),
    "SIRT": sirt(sinogram, geometry, num_iterations=100),
}

# %%
# The forward reconstructor
# -------------------------
# ``ForwardReconstructor`` runs a model trained to reconstruct (FBPConvNet,
# LPD, ...) as its solver did: the initial reconstruction for image models,
# the model's normalisation, without gradients. Here a FBPConvNet trained for
# a moment:

from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.optimizers.SupervisedSolver import SupervisedSolver
from LION.reconstructors.ForwardReconstructor import ForwardReconstructor

training = Subset(experiment.get_dataset("train"), range(40))
training_loader = DataLoader(training, batch_size=4, shuffle=True)

folder = Path(tempfile.mkdtemp())
fbpconvnet = FBPConvNet(geometry)
solver = SupervisedSolver(
    fbpconvnet, torch.optim.Adam(fbpconvnet.parameters()), torch.nn.MSELoss()
)
solver.set_training(training_loader)
solver.set_saving(folder, "FBPConvNet")
solver.train(2)
solver.save_final_results()

fbpconvnet, _, _ = FBPConvNet.load(folder / "FBPConvNet")  # from its file
forward = ForwardReconstructor(geometry, fbpconvnet)
reconstructions["FBPConvNet"] = forward.reconstruct(sinogram)

# %%
# Plug-and-Play
# -------------
# ``PnP`` uses a denoiser as the prior of an iterative algorithm (ADMM, HQS or
# FBS). Here the DRUNet its authors trained on natural images in [0, 1], so the
# images are normalised to [0, 1] for it (:doc:`d07_denoisers_and_pnp` trains
# one):

from LION.models.CNNs.drunet import DRUNet
from LION.reconstructors.PnP import PnP

denoiser = DRUNet.pretrained()
denoiser.set_normalisation("dataset", training)

parameters = PnP.default_parameters("ADMM")
parameters.max_iter = 20
parameters.noise_level = 10 / 255
pnp = PnP(geometry, denoiser, parameters)
reconstructions["PnP-ADMM"] = pnp.reconstruct(sinogram)

# %%
# Learned regularisers
# --------------------
# ``LearnedRegularizerReconstructor`` minimises ½||Ax - y||² + α R(x) by
# gradient descent, for a regulariser R: a trained network
# (:doc:`d08_learned_regularisers`) or any function of an image, e.g. a
# smoothed total variation:


def smoothed_tv(x, eps=1e-3):
    dx = x[..., 1:, :-1] - x[..., :-1, :-1]
    dy = x[..., :-1, 1:] - x[..., :-1, :-1]
    return torch.sqrt(dx**2 + dy**2 + eps**2).sum()


from LION.reconstructors.LearnedRegularizerReconstructor import (
    LearnedRegularizerReconstructor,
)

parameters = LearnedRegularizerReconstructor.default_parameters()
parameters.alpha = 10.0
parameters.n_steps = 50
regularised = LearnedRegularizerReconstructor(geometry, smoothed_tv, parameters)
reconstructions["smoothed TV"] = regularised.reconstruct(sinogram)

# %%
psnr = PSNR()
fig, axes = plt.subplots(1, len(reconstructions), figsize=(4 * len(reconstructions), 4))
for ax, (name, x) in zip(axes, reconstructions.items()):
    ax.imshow(x[0].cpu(), cmap="gray")
    ax.set_title(f"{name}, {psnr(x, image, batched=False):.1f} dB")
    ax.axis("off")

# %%
# What they share
# ---------------
# Every reconstructor:
#
# - is made with ``(operator, model, parameters)``: the operator, or its
#   parameters (a CT geometry); the model; the algorithm's settings, from its
#   class's ``default_parameters()``;
# - reconstructs with ``reconstruct(y, x0=None, verbose=1)``: one sample (the
#   operator's range shape) or a batch, each on its own; ``x0`` a start, for
#   algorithms that take one;
# - starts, when it starts from an image, from its parameters'
#   ``initial_reconstruction`` (the operator's default, FDK for CT):

print(pnp.parameters.initial_reconstruction)
batch = torch.stack([sinogram, sinogram])
print(forward.reconstruct(batch).shape)
start = sirt(sinogram, geometry, num_iterations=20)
x = pnp.reconstruct(sinogram, x0=start, verbose=0)

# %%
# In a solver's validation
# ------------------------
# A solver validates through a reconstructor when the model is used inside
# one: e.g. a learned regulariser's (:doc:`d08_learned_regularisers`), or a
# denoiser validated by its PnP reconstructions. The solver puts the model
# being trained into it:
#
# .. code-block:: python
#
#    solver.set_validation(validation_loader, 1, PSNR(), reconstructor=pnp)

# %%
# Saving and loading
# ------------------
# A reconstructor saves its settings and its model by the file it was loaded
# from (the weights stay there); ``load`` makes both again. ``instantiate``
# makes it without its model, to give it another:

from LION.reconstructors.LIONreconstructor import LIONReconstructor

forward.save(folder / "FBPConvNet_reconstructor.json")
again = LIONReconstructor.load(folder / "FBPConvNet_reconstructor.json")
print(type(again).__name__, type(again.model).__name__)

other_model = FBPConvNet(geometry)
again = LIONReconstructor.instantiate(folder / "FBPConvNet_reconstructor.json")
again.model = other_model

# %%
# Your own
# --------
# A reconstructor derives from ``LIONReconstructor`` and defines
# ``default_parameters()`` and ``reconstruct_sample(y, x0, verbose)``; the
# tutorial :doc:`d13_extending_lion` writes one.
