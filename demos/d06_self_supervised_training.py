# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Self-supervised training
========================

Clean images are rarely available for real scans. Self-supervised methods
train from the noisy measurements alone, using what is known about the noise
and the operator. LION has several, each a solver used like the supervised
one: the model, the optimizer, the loss, the operator's parameters and the
solver's own.

This tutorial trains with Noise2Inverse, then shows the others. A short run, to
show how.

Previous: :doc:`d05_supervised_training`. Next: :doc:`d07_denoisers_and_pnp`.

Needs a GPU and the processed LIDC-IDRI data in ``LION_DATA_PATH``.
"""

# %%
# Noise2Inverse
# -------------
# Noise2Inverse (Hendriksen et al., 2020) splits the sinogram into sets of
# angles. Each set's noise is independent of the others', so a network trained
# to map the reconstruction of some sets to that of the others learns to
# remove the noise.

import tempfile

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.metrics.psnr import PSNR
from LION.models.CNNs.UNets.Unet import UNet
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver

experiment = ct_experiments.LowDoseCTRecon()
training = Subset(experiment.get_dataset("train"), range(40))  # targets unused
validation = Subset(experiment.get_dataset("validation"), range(8))

training_loader = DataLoader(training, batch_size=2, shuffle=True)
validation_loader = DataLoader(validation, batch_size=2)

# %%
# The solver's parameters say how to split: into ``sino_split_count`` sets of
# every k-th angle, and which sets are the targets (``cali_J``). The paper's
# strategies are X:1 (each set in turn the target, the others the input) and
# 1:X (the reverse).

parameters = Noise2InverseSolver.default_parameters()
print(parameters)
parameters.sino_split_count = 4
parameters.cali_J = Noise2InverseSolver.X_one_strategy(4)
print(parameters.cali_J)

# %%
# The model takes images (the reconstructions of the sets), so it has no
# operator: the solver needs the operator's parameters, to split it.

model = UNet()
solver = Noise2InverseSolver(
    model,
    torch.optim.Adam(model.parameters(), lr=1e-3),
    torch.nn.MSELoss(),
    operator_params=experiment.operator_params,
    solver_params=parameters,
)
solver.set_training(training_loader)

# %%
# Validation can still use clean images, if there are some, to measure how well
# it reconstructs (never to train): the solver validates its reconstruction.

solver.set_validation(validation_loader, 1, PSNR(), validation_fname="best")
solver.set_saving(tempfile.mkdtemp(), "Noise2Inverse")
solver.train(3)
print("validation PSNR:", solver.validation_loss)
solver.save_final_results()

# %%
# A Noise2Inverse reconstruction is the mean of the model's outputs over the
# splits: ``solver.reconstruct``, not the model alone.

sinogram, image = validation[0]
with torch.no_grad():
    reconstruction = solver.reconstruct(sinogram[None])[0]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
shown = [
    ("target", image),
    ("FDK", solver.op.default_reconstruction(sinogram)),
    ("Noise2Inverse, 3 epochs", reconstruction),
]
for ax, (title, x) in zip(axes, shown):
    ax.imshow(x[0].cpu(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")

# %%
# The other self-supervised solvers
# ---------------------------------
# Each is used the same way; only its parameters differ. Each prints what it
# can be set to (and its paper's values):

from LION.optimizers.Equivariance2InverseSolver import Equivariance2InverseSolver
from LION.optimizers.EquivariantSolver import EquivariantSolver
from LION.optimizers.Noisier2InverseSolver import Noisier2InverseSolver
from LION.optimizers.Proj2ProjSolver import Proj2ProjSolver
from LION.optimizers.Sparse2InverseSolver import Sparse2InverseSolver

for solver_class in [
    Sparse2InverseSolver,
    Noisier2InverseSolver,
    Proj2ProjSolver,
    EquivariantSolver,
    Equivariance2InverseSolver,
]:
    print(solver_class.__name__)
    print(solver_class.default_parameters())

# %%
# E.g. equivariant imaging (Chen et al., 2021) learns from data consistency
# and from the reconstruction being equivariant to rotations:
#
# .. code-block:: python
#
#    solver = EquivariantSolver(
#        model,
#        torch.optim.Adam(model.parameters(), lr=1e-3),
#        torch.nn.MSELoss(),
#        operator_params=experiment.operator_params,
#    )
#
# Some of these methods are only defined for the mean squared error: their
# solvers train with it, and use ``loss_fn`` only to validate and test (they
# warn if it isn't the MSE).
