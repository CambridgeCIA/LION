# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Plug-and-Play: train a Gaussian denoiser, then reconstruct with it.

1. A DRUNet (Zhang et al., 2021) learns to remove Gaussian noise from clean
   CT images (``GaussianDenoiserSolver``): each image gets its own noise
   level, from a range, and a random flip or rotation, as in the literature.
2. The denoiser is the prior of a PnP-ADMM reconstruction
   (``LION.reconstructors.PnP``): ADMM alternates a data step (conjugate
   gradient on the CT operator) and a denoising step.

The noise levels are in the units of the images the denoiser sees: they are
normalised to [0, 1] here (``set_normalisation``), so the literature's
k/255 apply. PnP methods that converge need more of the denoiser: LION has the
penalties of Pesquet et al. (2021) and Hurault et al. (2022)
(``LION.losses.jacobian_penalty``), given to the solver as its
``model_regularization``. HQS and FBS are the other PnP algorithms; the
learned regulariser reconstructor (``LearnedRegularizerReconstructor``) is
used the same way.

Needs: a GPU, the processed LIDC-IDRI data in ``LION_DATA_PATH``, and
``LION_RESULTS_PATH``.
"""

# %% Imports
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.metrics.psnr import PSNR
from LION.models.CNNs.drunet import DRUNet
from LION.operators.ct import fdk
from LION.optimizers.GaussianDenoiserSolver import GaussianDenoiserSolver
from LION.reconstructors.PnP import PnP
from LION.utils.paths import results_path

# %% Settings
EPOCHS = 5
TRAINING_SLICES = 200
# LION uses the GPU if there is one (torch.cuda.set_device(k) picks another);
# device is where this script moves samples to use them
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
verbose = 2  # 0: nothing; 1: one progress bar; 2: also each epoch's bars and messages

save_folder = results_path("examples/pnp")
save_folder.mkdir(parents=True, exist_ok=True)

# %% Data: the denoiser only uses the clean images (the targets)
experiment = ct_experiments.LowDoseCTRecon()
training = experiment.get_dataset("train")
if TRAINING_SLICES is not None:
    training = Subset(training, range(min(TRAINING_SLICES, len(training))))

# %% The denoiser
parameters = DRUNet.default_parameters()
parameters.use_noise_level = True  # the noise level as an input, as in the paper
model = DRUNet(parameters)
# The images' range over the training data, mapped to [0, 1] (saved with it)
model.set_normalisation("dataset", training)

# %% Train it
solver_parameters = GaussianDenoiserSolver.default_parameters()
solver_parameters.noise_level = [0.0, 50 / 255]  # one level per image, from this range
solver_parameters.patches = GaussianDenoiserSolver.patches(8, 128)  # 8 crops a slice
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
solver = GaussianDenoiserSolver(
    model,
    optimizer,
    torch.nn.L1Loss(),
    operator_params=experiment.operator_params,
    solver_params=solver_parameters,
    verbose=verbose,
)
solver.set_training(DataLoader(training, batch_size=4, shuffle=True))
solver.set_saving(save_folder, "DRUNet")
solver.train(EPOCHS)
solver.save_final_results()

# %% Reconstruct with it
pnp_parameters = PnP.default_parameters("ADMM")
print(pnp_parameters)  # the algorithm's settings
pnp_parameters.max_iter = 20
pnp_parameters.noise_level = 10 / 255  # the denoiser's, in [0, 1] units
# The reconstructor takes the operator (or its parameters) and the model. The
# model loaded from its file, so the reconstructor's record names that file.
model, _, _ = DRUNet.load(save_folder / "DRUNet")
model.eval()
pnp = PnP(experiment.operator_params, model, pnp_parameters)
pnp.save(save_folder / "PnP.json")  # its settings and its model's file

sinogram, image = experiment.get_dataset("test")[0]
reconstruction = pnp.reconstruct(sinogram.to(device), verbose=verbose).cpu()
# or one algorithm with its settings given:
# reconstruction = pnp.admm_algorithm(sinogram.to(device), eta=1e-4, max_iter=20)
# FDK, the baseline: of the experiment's geometry
fdk_reconstruction = fdk(sinogram.to(device), experiment.operator_params).cpu()
# or any operator's default reconstruction (FDK for CT):
# fdk_reconstruction = pnp.op.default_reconstruction(sinogram.to(device)).cpu()

# %% Compare
psnr = PSNR()
fdk_psnr = psnr(fdk_reconstruction, image, batched=False)
pnp_psnr = psnr(reconstruction, image, batched=False)

# To numpy for plotting (the data may be on the GPU)
image_np = image[0].cpu().numpy()
fdk_np = fdk_reconstruction[0].cpu().numpy()
pnp_np = reconstruction[0].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (title, x) in zip(
    axes,
    [
        ("Ground truth", image_np),
        (f"FDK, {fdk_psnr:.2f} dB", fdk_np),
        (f"PnP-ADMM, {pnp_psnr:.2f} dB", pnp_np),
    ],
):
    ax.imshow(x, cmap="gray", vmin=image_np.min(), vmax=image_np.max())
    ax.set_title(title)
    ax.axis("off")
fig.savefig(save_folder / "pnp.png")
