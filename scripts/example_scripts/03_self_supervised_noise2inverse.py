# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Self-supervised training: Noise2Inverse (Hendriksen et al., 2020).

No clean images are used: the noisy sinograms are split into interleaved sets
of angles, each set is reconstructed (FDK, the operator's default), and the
model learns to map the reconstruction of some sets to that of the others.
The noise of the sets is independent, so the model learns to remove it.

The model is a U-Net, which fits a 24 GB GPU. The paper's MSD-Net
(``LION.models.CNNs.MSDNet``) is a one-line change, but at 512x512 its densely
connected layers need far more memory. Any image-to-image model works. The
other self-supervised solvers are used the same way: ``Sparse2InverseSolver``,
``Noisier2InverseSolver``, ``Proj2ProjSolver``, ``EquivariantSolver``,
``Equivariance2InverseSolver`` (each in ``LION.optimizers``).

The validation data do have clean images: they are only used to score the
reconstructions (``solver.reconstruct``), never to train.

Needs: a GPU, the processed LIDC-IDRI data in ``LION_DATA_PATH``, and
``LION_RESULTS_PATH``.
"""

# %% Imports
import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.metrics.psnr import PSNR
from LION.models.CNNs.UNets.Unet import UNet
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver
from LION.utils.paths import results_path
from LION.utils.device import default_device

# %% Settings
EPOCHS = 5
TRAINING_SLICES = 200
VALIDATION_SLICES = 20
# LION uses the GPU if there is one; to pick another, torch.cuda.set_device(k)
# here, not a different device below: the models and data follow LION's
device = default_device()  # where this script moves samples to use them
verbose = 2  # 0: nothing; 1: one progress bar; 2: also each epoch's bars and messages

save_folder = results_path("examples/noise2inverse")
save_folder.mkdir(parents=True, exist_ok=True)

# %% Data: the training targets are ignored
experiment = ct_experiments.LowDoseCTRecon()
training = experiment.get_dataset("train")
validation = experiment.get_dataset("validation")
if TRAINING_SLICES is not None:
    training = Subset(training, range(min(TRAINING_SLICES, len(training))))
    validation = Subset(validation, range(min(VALIDATION_SLICES, len(validation))))

# %% The model: an image-to-image network, without an operator
model = UNet()

# %% The solver's parameters
parameters = Noise2InverseSolver.default_parameters()
parameters.sino_split_count = 4  # 4 sets of angles: every 4th angle
# Which sets are the targets: the default is the paper's X:1 (each set in turn
# is the target, the mean of the others the input); 1:X is the reverse.
parameters.cali_J = Noise2InverseSolver.X_one_strategy(4)

# %% Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# The solver needs the operator's parameters to split it: the model has none.
solver = Noise2InverseSolver(
    model,
    optimizer,
    torch.nn.MSELoss(),
    operator_params=experiment.operator_params,
    solver_params=parameters,
    verbose=verbose,
)
solver.set_training(DataLoader(training, batch_size=2, shuffle=True))
solver.set_validation(
    DataLoader(validation, batch_size=2),
    validation_freq=1,
    validation_fn=PSNR(),
    validation_fname="Noise2Inverse_best",
)
solver.set_saving(save_folder, "Noise2Inverse")
solver.train(EPOCHS)
solver.save_final_results()

# %% Reconstruct
# A Noise2Inverse reconstruction is the mean of the model's outputs over the
# splits: solver.reconstruct, not the model alone.
sinogram, image = validation[0]
with torch.no_grad():
    reconstruction = solver.reconstruct(sinogram.unsqueeze(0).to(device))
print(f"PSNR: {PSNR()(reconstruction[0].cpu(), image, batched=False):.2f} dB")
