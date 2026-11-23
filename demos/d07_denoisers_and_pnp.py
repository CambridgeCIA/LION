# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Denoisers and Plug-and-Play
===========================

Plug-and-Play (PnP) methods reconstruct with an iterative algorithm in which a
denoiser takes the place of a regulariser. The denoiser is trained once, on
clean images with added Gaussian noise, and works for any operator: the
operator only appears in the algorithm.

This tutorial trains a denoiser with ``GaussianDenoiserSolver`` and
reconstructs with it (``LION.reconstructors.PnP``). Short runs, to show how.

Previous: :doc:`d06_self_supervised_training`. Next:
:doc:`d08_learned_regularisers`.

Needs a GPU and the processed LIDC-IDRI data in ``LION_DATA_PATH``; downloads
the authors' trained DRUNet (130 MB).

.. note::

   The models trained in these tutorials are toys: a few epochs on a few
   slices, to show how LION is used, so their results are poor (often worse
   than FDK). Trained as their papers train them (many epochs, all the data),
   they do what the papers report.
"""

# %%
# The data and the denoiser
# -------------------------
# The denoiser only uses the clean images (the targets of the experiment's
# samples). DRUNet (Zhang et al., 2021) takes the noise level as an input, so
# one network denoises every level.

import tempfile

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.metrics.psnr import PSNR
from LION.models.CNNs.drunet import DRUNet
from LION.optimizers.GaussianDenoiserSolver import GaussianDenoiserSolver

experiment = ct_experiments.LowDoseCTRecon()
training = Subset(experiment.get_dataset("train"), range(40))
validation = Subset(experiment.get_dataset("validation"), range(8))

training_loader = DataLoader(training, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation, batch_size=4)

parameters = DRUNet.default_parameters()
parameters.use_noise_level = True
denoiser = DRUNet(parameters)

# %%
# Noise levels are in the units the denoiser sees. The literature's (k/255)
# are for images in [0, 1], so the images are normalised to [0, 1] by their
# range over the training data. The normalisation is saved with the model and
# applied whenever it is used.

denoiser.set_normalisation("dataset", training)

# %%
# Training the denoiser
# ---------------------
# ``GaussianDenoiserSolver`` adds the noise. Its parameters follow the
# literature: one noise level per image, drawn from a range; random flips and
# rotations; optionally random crops (patches) and random erasing; fixed noise
# levels to validate at.

solver_parameters = GaussianDenoiserSolver.default_parameters()
solver_parameters.noise_level = [0.0, 50 / 255]
solver_parameters.patches = GaussianDenoiserSolver.patches(8, 128)  # 8 crops a slice
solver_parameters.validation_noise_levels = [15 / 255, 25 / 255]
print(solver_parameters)

# %%
# The loss: the papers use either. DRUNet is trained with the L1 loss; DnCNN,
# FFDNet and the gradient-step denoiser with the L2 loss (MSE). None of them
# adds a ``model_regularization`` (see "Jacobian penalties" below for the
# papers that do). The L2 loss trains towards the MMSE denoiser
# E[x | noisy x], the one PnP theory assumes (through Tweedie's formula); the L1
# loss towards the median, and often trains better (Zhao et al.).
#
# * L1, DRUNet: Zhang et al., "Plug-and-Play Image Restoration With Deep
#   Denoiser Prior", IEEE TPAMI, 2022.
# * L2, DnCNN: Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of
#   Deep CNN for Image Denoising", IEEE Trans. Image Processing, 2017.
# * L2, FFDNet: Zhang, Zuo and Zhang, "FFDNet: Toward a Fast and Flexible
#   Solution for CNN-Based Image Denoising", IEEE Trans. Image Processing,
#   2018.
# * L2, gradient-step denoiser: Hurault, Leclaire and Papadakis, "Gradient
#   Step Denoiser for Convergent Plug-and-Play", ICLR 2022.
# * L1 vs L2: Zhao, Gallo, Frosio and Kautz, "Loss Functions for Image
#   Restoration with Neural Networks", IEEE Trans. Computational Imaging, 2017.

l1_loss = torch.nn.L1Loss()  # DRUNet
l2_loss = torch.nn.MSELoss()  # DnCNN, FFDNet, GS denoiser
loss_fn = l1_loss

# %%
# The solver doesn't use the operator, but takes its parameters, recorded as
# the problem the denoiser was trained for.

folder = tempfile.mkdtemp()
solver = GaussianDenoiserSolver(
    denoiser,
    torch.optim.Adam(denoiser.parameters(), lr=1e-4),
    loss_fn,
    operator_params=experiment.operator_params,
    solver_params=solver_parameters,
)
solver.set_training(training_loader)
solver.set_validation(validation_loader, 1, PSNR(), validation_fname="DRUNet_best")
solver.set_saving(folder, "DRUNet")
solver.train(3)
solver.save_final_results()

# %%
# Denoising a validation image at 25/255. LION's solvers and reconstructors
# normalise for the model; called by hand, the model takes normalised images:

image = validation[0][1]
x = denoiser.normalise(image[None])  # to [0, 1], as the denoiser sees it
noisy = x + 25 / 255 * torch.randn_like(x)
denoiser.eval()
with torch.no_grad():
    denoised = denoiser(noisy, noise_level=25 / 255)

# %%
# Jacobian penalties
# ------------------
# PnP methods are proven to converge when the denoiser has a property a network
# doesn't have on its own: being firmly nonexpansive or a proximal operator. A
# penalty on the denoiser's Jacobian, given to the solver as its
# ``model_regularization``, trains it towards that. Both papers use the L2
# loss, train without the penalty first, then add it:
#
# * ``FirmlyNonexpansivePenalty`` (weight 1e-5, epsilon 0.05), on a DnCNN with
#   leaky ReLUs: Pesquet, Repetti, Terris and Wiaux, "Learning Maximally
#   Monotone Operators for Image Recovery", SIAM J. Imaging Sciences, 2021.
# * ``ProximalDenoiserPenalty`` (weight 1e-3 or 1e-2, epsilon 0.1), fine-tuning
#   a gradient-step DRUNet: Hurault, Leclaire and Papadakis, "Proximal Denoiser
#   for Convergent Plug-and-Play Optimization with Nonconvex Regularization",
#   ICML 2022.
#
# For example, fine-tuning the trained denoiser as Pesquet et al.:
#
# .. code-block:: python
#
#    from LION.losses.jacobian_penalty import FirmlyNonexpansivePenalty
#
#    solver = GaussianDenoiserSolver(
#        denoiser,
#        torch.optim.Adam(denoiser.parameters(), lr=1e-5),
#        l2_loss,
#        operator_params=experiment.operator_params,
#        solver_params=solver_parameters,
#        model_regularization=FirmlyNonexpansivePenalty(),  # off by default
#    )
#
# The gradient-step denoiser (``LION.models.PnP.gradient_step_denoiser.GSD``)
# wraps any denoiser N as ``x - grad g(x)``, ``g(x) = ½||x - N(x)||²``: the
# denoiser ``ProximalDenoiserPenalty`` is for.

# %%
# Plug-and-Play reconstruction
# ----------------------------
# ``PnP`` takes the operator (or its parameters), the denoiser and the
# algorithm's parameters: ADMM, HQS or FBS (forward-backward splitting).

from LION.operators.ct import fdk
from LION.reconstructors.PnP import PnP

pnp_parameters = PnP.default_parameters("ADMM")
print(pnp_parameters)
pnp_parameters.max_iter = 20
pnp_parameters.noise_level = 10 / 255  # the denoiser's, in its [0, 1] units
# eta, ADMM's weight between the data and the prior, is by default relative to
# the operator: 1e-2 ||A||² (larger trusts the prior more)
pnp = PnP(experiment.operator_params, denoiser, pnp_parameters)

sinogram, image = validation[0]
reconstruction = pnp.reconstruct(sinogram)
# or one algorithm with its settings given:
# reconstruction = pnp.admm_algorithm(sinogram, max_iter=20, noise_level=10 / 255)

# %%
# A pretrained denoiser
# ---------------------
# The denoiser above is a toy, trained for 3 epochs on 40 slices: a poor
# prior. The DRUNet its authors trained on natural images (a 130 MB download,
# on images in [0, 1]) is a good one, and the same reconstructor with it shows
# what PnP does:

pretrained = DRUNet.pretrained()
pretrained.set_normalisation("dataset", training)
pnp_pretrained = PnP(experiment.operator_params, pretrained, pnp_parameters)
pretrained_reconstruction = pnp_pretrained.reconstruct(sinogram)

# %%
# HQS and FBS are used the same way:

hqs = PnP(experiment.operator_params, denoiser, PnP.default_parameters("HQS"))
print(PnP.default_parameters("FBS"))

# %%
psnr = PSNR()
fbp = fdk(sinogram, experiment.operator_params)
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
shown = [
    ("noisy (25/255)", noisy[0]),
    ("denoised (toy denoiser)", denoised[0]),
    (f"FDK, {psnr(fbp, image, batched=False):.1f} dB", fbp),
    (f"PnP, toy, {psnr(reconstruction, image, batched=False):.1f} dB", reconstruction),
    (
        f"PnP, pretrained, {psnr(pretrained_reconstruction, image, batched=False):.1f} dB",
        pretrained_reconstruction,
    ),
]
for ax, (title, x) in zip(axes, shown):
    ax.imshow(x[0].cpu(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")

# %%
# A reconstructor is saved with ``pnp.save(file)``: its settings, and its model
# by the file it was loaded from (:doc:`d09_saving_and_records`).
