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
``SupervisedSolver``, step by step.

A short run (a few epochs on a few slices) to show how; a real training needs
many more.

Previous: :doc:`d04_models`. Next: :doc:`d06_self_supervised_training`.

Needs a GPU and the processed LIDC-IDRI data in ``LION_DATA_PATH``.
"""

# %%
# The pieces
# ----------
# The data, the model, a torch optimizer and a loss. Any torch optimizer and
# loss work: they are recorded with the model.

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

model = FBPConvNet(experiment.operator_params)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# %%
# The solver
# ----------
# Every solver takes the same arguments, in this order: the model, the
# optimizer, the loss, then, as needed, the operator's parameters (taken from
# the model when it has them), the solver's own parameters, a model
# regularisation, a learning-rate scheduler, verbosity, device and seed.

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
solver = SupervisedSolver(
    model,
    optimizer,
    loss_fn,
    scheduler=scheduler,  # steps every epoch (scheduler_step="batch": every batch)
    verbose=1,  # 0: nothing; 1: one progress bar; 2: also each epoch's, and messages
    seed=0,  # the training's randomness: the same seed, the same model
)

# %%
# Then it is told about the data and what to do around training. Validation
# runs every ``validation_freq`` epochs and keeps the best model: by the lowest
# value, or the highest for a score like the PSNR.

folder = Path(tempfile.mkdtemp())

solver.set_training(DataLoader(training, batch_size=4, shuffle=True))
solver.set_validation(
    DataLoader(validation, batch_size=4),
    validation_freq=1,
    validation_fn=PSNR(),
    validation_fname="FBPConvNet_best",
)
solver.set_testing(DataLoader(test, batch_size=4), PSNR())
solver.set_saving(folder, "FBPConvNet")
solver.set_checkpointing("FBPConvNet_check_*.pt", checkpoint_freq=2)

# %%
# Training
# --------

solver.train(4)
solver.save_final_results()

print("training loss per epoch:", solver.train_loss)
print("validation PSNR:", solver.validation_loss)
print("learning rate:", scheduler.get_last_lr())

# %%
# A checkpoint is kept every 2 epochs. Called again with the same set-up,
# ``train`` resumes from the latest one (the model, the optimizer, the
# scheduler, the random state), so a run stopped half way ends with the same
# model as one that wasn't. Once done, they can go:

print(sorted(p.name for p in folder.iterdir()))
solver.clean_checkpoints()

# %%
# Testing
# -------
# ``test()`` scores the model on the test data (one PSNR per sample here). The
# evaluation tutorial (d10) does this properly, against other methods.

scores = solver.test()
print(f"test PSNR: {scores.mean():.2f} dB")

# %%
# What was saved
# --------------
# The final model, the best model by validation, each as a ``.pt`` (weights)
# and a ``.json`` (parameters and records). ``load`` makes the model again:

trained, options, data = FBPConvNet.load(folder / "FBPConvNet")
best, _, _ = FBPConvNet.load(folder / "FBPConvNet_best")
print(type(trained).__name__, data["epoch"])

# %%
# The records say how it was trained: the data (which slices), the solver and
# its settings, the optimizer, the scheduler, the seed, the losses. Enough to
# train the same model again (the tutorial on records, d09):

print(options.training.solver["_target_"], options.training.optimizer)

# %%
fig, ax = plt.subplots()
ax.plot(solver.train_loss, label="training loss (MSE)")
ax.set_xlabel("epoch")
ax.legend(loc="upper left")
ax2 = ax.twinx()
ax2.plot(solver.validation_loss, "C1", label="validation PSNR")
ax2.legend(loc="upper right")

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
# The reconstructions with the trained model, in a batch or one at a time:

from LION.reconstructors.ForwardReconstructor import ForwardReconstructor

reconstructor = ForwardReconstructor(experiment.operator_params, best)
sinogram, image = validation[0]
reconstruction = reconstructor.reconstruct(sinogram)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, (title, x) in zip(
    axes, [("target", image), ("FBPConvNet, 4 epochs", reconstruction)]
):
    ax.imshow(x[0].cpu(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
