# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Supervised training
===================

A solver trains a model: it is a torch training loop with what research needs
around it (validation, checkpoints, saving with records, testing). This
tutorial trains FBPConvNet on pairs of noisy sinograms and images with
``SupervisedSolver``: first the least it needs, then each option.

Solvers are optional: a LION model is a torch module, which you can train with
your own loop (the end of this tutorial). What LION's solvers add is that
everything about the training is recorded with the model, so it can be
repeated, checked and published (the tutorials on records and publishing).

Short runs (a few epochs on a few slices) to show how; a real training needs
many more.

Previous: :doc:`d04_models`. Next: :doc:`d06_self_supervised_training`.

Needs a GPU and the processed LIDC-IDRI data in ``LION_DATA_PATH``.
"""

# %%
# The pieces
# ----------
# The data, the model, a torch optimizer and a loss. Any torch optimizer and
# loss work.

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.metrics.psnr import PSNR
from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.optimizers.SupervisedSolver import SupervisedSolver

experiment = ct_experiments.LowDoseCTRecon()
training = Subset(experiment.get_dataset("train"), range(40))
validation = Subset(experiment.get_dataset("validation"), range(8))
test = Subset(experiment.get_dataset("test"), range(8))

# torch DataLoaders: they batch (and shuffle) the samples
training_loader = DataLoader(training, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation, batch_size=4)
test_loader = DataLoader(test, batch_size=4)

# %%
# The least a solver needs
# ------------------------
# The model, the optimizer and the loss; the training data; where to save the
# model. Then ``train(n_epochs)`` and ``save_final_results()``.

model = FBPConvNet(experiment.operator_params)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

solver = SupervisedSolver(model, optimizer, torch.nn.MSELoss())
solver.set_training(training_loader)
folder = Path(tempfile.mkdtemp())
solver.set_saving(folder, "FBPConvNet")

solver.train(2)
solver.save_final_results()  # FBPConvNet.pt (weights) and FBPConvNet.json
print("training loss per epoch:", solver.train_loss)

# %%
# That is a complete training: the saved model holds its weights, its
# parameters and the record of how it was trained. Everything below is
# optional.
#
# Options of the solver
# ---------------------
# Every solver takes the same arguments, in this order: the model, the
# optimizer, the loss, then, all optional: the operator's parameters (taken
# from the model when it has them), the solver's own parameters, a model
# regularisation (a penalty on the model added to the loss, e.g. for PnP
# denoisers: the denoisers tutorial, d07), a learning-rate scheduler, how much
# it prints, the device and the seed.

model = FBPConvNet(experiment.operator_params)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

solver = SupervisedSolver(
    model,
    optimizer,
    torch.nn.MSELoss(),
    scheduler=scheduler,  # optional; steps every epoch (scheduler_step="batch": every batch)
    verbose=2,  # 0: nothing; 1 (default): one progress bar; 2: also each epoch's, and messages
    seed=0,  # the default: the same seed, the same model
)
solver.set_training(training_loader)
folder = Path(tempfile.mkdtemp())
solver.set_saving(folder, "FBPConvNet")

# %%
# Validation (optional) runs every ``validation_freq`` epochs and keeps the
# best model: by the lowest value, or the highest for a score like the PSNR.

solver.set_validation(
    validation_loader,
    validation_freq=1,
    validation_fn=PSNR(),
    validation_fname="FBPConvNet_best",
)

# %%
# Testing (optional) is a score on test data, run by ``solver.test()``. The
# evaluation tutorial (d10) evaluates properly, against other methods.

solver.set_testing(test_loader, PSNR())

# %%
# Checkpoints
# -----------
# A checkpoint is everything needed to carry on training as if never stopped:
# the model's weights, the optimizer's state (e.g. Adam's moments), the
# scheduler's, the epoch, the losses so far and the random state. Training
# takes long, and jobs get stopped: with checkpoints, a stopped training loses
# at most ``checkpoint_freq`` epochs.
#
# - **When they are saved:** every ``checkpoint_freq`` epochs, as
#   ``checkpoint_fname`` with its ``*`` replaced by the epoch, in the save
#   folder.
# - **When they are loaded:** at the start of ``train()``, the latest
#   checkpoint matching the name is loaded, if there is one (and
#   ``load_checkpoint_if_exists``, True by default), and training goes on from
#   its epoch. ``train(n)`` trains until n epochs are done in all, so a run
#   started again with the same set-up finishes where it would have.
# - **By default**, with ``set_saving`` alone, the solver checkpoints every 10
#   epochs as ``<name>_checkpoint_*.pt``. ``set_checkpointing`` sets them:

solver.set_checkpointing("FBPConvNet_check_*.pt", checkpoint_freq=2)
# solver.set_checkpointing(..., load_checkpoint_if_exists=False)  # always start anew

# %%
# Training
# --------

solver.train(4)
solver.save_final_results()

print("training loss per epoch:", solver.train_loss)
print("validation PSNR:", solver.validation_loss)
print("learning rate:", scheduler.get_last_lr())
print(sorted(p.name for p in folder.iterdir()))

# %%
# Resuming: a new solver with the same set-up (as after a crash) finds the
# checkpoint of epoch 4 and only trains epochs 5 and 6, with the losses of the
# first four kept.

model = FBPConvNet(
    experiment.operator_params
)  # new weights: replaced by the checkpoint's
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
resumed = SupervisedSolver(
    model,
    optimizer,
    torch.nn.MSELoss(),
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5),
)
resumed.set_training(training_loader)
resumed.set_saving(folder, "FBPConvNet")
resumed.set_checkpointing("FBPConvNet_check_*.pt", checkpoint_freq=2)
resumed.train(6)
print("training loss per epoch:", resumed.train_loss)

# %%
# Once a run is finished, its checkpoints can go:

resumed.clean_checkpoints()

# %%
# Testing and what was saved
# --------------------------

scores = solver.test()
print(f"test PSNR: {scores.mean():.2f} dB")

# %%
# The final model and the best model by validation, each as a ``.pt``
# (weights) and a ``.json`` (parameters and records). ``load`` makes the model
# again; the record says how it was trained (the data, the solver and its
# settings, the optimizer, the scheduler, the seed, the losses), enough to
# train the same model again (the tutorial on records, d09):

trained, options, data = FBPConvNet.load(folder / "FBPConvNet")
best, _, _ = FBPConvNet.load(folder / "FBPConvNet_best")
print(options.training.solver["_target_"], options.training.optimizer["lr"])

fig, ax = plt.subplots()
ax.plot(resumed.train_loss, label="training loss (MSE)")
ax.set_xlabel("epoch")
ax.legend(loc="upper left")
ax2 = ax.twinx()
ax2.plot(solver.validation_loss, "C1", label="validation PSNR")
ax2.legend(loc="upper right")

# %%
# Reconstructing
# --------------
# A reconstructor (``LION.reconstructors``) runs a trained model the way its
# solver did (the initial reconstruction, the normalisation), on one sample or
# a batch, without gradients. It is optional too: the model can be called
# directly, as below.

from LION.reconstructors.ForwardReconstructor import ForwardReconstructor

best.eval()  # the layers' evaluation behaviour (dropout, batch normalisation)
reconstructor = ForwardReconstructor(experiment.operator_params, best)
sinogram, image = validation[0]
reconstruction = reconstructor.reconstruct(sinogram)
# the same, by hand: FBPConvNet takes the measurements
# with torch.no_grad():
#     reconstruction = best(sinogram[None])[0]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, (title, x) in zip(
    axes, [("target", image), ("FBPConvNet, 4 epochs", reconstruction)]
):
    ax.imshow(x[0].cpu(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")

# %%
# Other supervised models
# -----------------------
# Any model is trained the same way: only the model changes. E.g. LPD:
#
# .. code-block:: python
#
#    from LION.models.iterative_unrolled.LPD import LPD
#    model = LPD(experiment.operator_params)
#
# Without LION's solver
# ---------------------
# A LION model is a torch module: train it with your own loop if you prefer.
# What you lose is what the solver records (which data, how it was trained),
# and so what the later tutorials do with it: training again from the record,
# checking that comparisons are fair, publishing.

model = FBPConvNet(experiment.operator_params)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
for sinograms, images in training_loader:
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(sinograms), images)
    loss.backward()
    optimizer.step()
print(f"one epoch by hand, last loss {loss.item():.4f}")
