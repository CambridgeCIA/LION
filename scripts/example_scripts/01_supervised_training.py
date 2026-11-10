# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Supervised training: FBPConvNet on LION's low-dose CT experiment.

A LION training is four pieces, each of which you can replace with your own:

- the experiment: the geometry, the noise and the data (here simulated low-dose
  sinograms of LIDC-IDRI slices, with their images as targets);
- the model: here FBPConvNet, a U-Net that cleans an FDK reconstruction;
- a torch optimizer, loss and (optionally) learning-rate scheduler;
- the solver, which trains the model on the data and saves it with a record of
  everything above (``SupervisedSolver`` for pairs of measurements and images).

The script validates every epoch (keeping the best model), checkpoints every
epoch (so a stopped run resumes, see 06_resume_and_fine_tune.py) and saves the
final model. Any other supervised model is one line away: e.g.
``FBPConvNetImage``, ``cFBPConvNet``, ``DeepFBPNetwork`` or LPD (see
02_learned_primal_dual.py).

Needs: a GPU (the CT operators use tomosipo/ASTRA), the processed LIDC-IDRI
data in ``LION_DATA_PATH``, and ``LION_RESULTS_PATH`` set (where results go).
"""

# %% Imports
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.metrics.psnr import PSNR
from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.optimizers.SupervisedSolver import SupervisedSolver
from LION.utils.paths import results_path

# %% Settings
# A short run, to see it work. For a real training: more epochs (the paper:
# 100s) and all the data (None).
EPOCHS = 5
TRAINING_SLICES = 200
VALIDATION_SLICES = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
verbose = 2  # 0: nothing; 1: one progress bar; 2: also each epoch's bars and messages

save_folder = results_path("examples/supervised")
save_folder.mkdir(parents=True, exist_ok=True)

# %% The experiment: geometry, noise and data
experiment = ct_experiments.LowDoseCTRecon()
# Each sample is (noisy sinogram, image). The splits are by patient.
training = experiment.get_dataset("train")
validation = experiment.get_dataset("validation")
if TRAINING_SLICES is not None:
    # a torch Subset of a LION dataset is recorded as such: which slices
    training = Subset(training, range(min(TRAINING_SLICES, len(training))))
    validation = Subset(validation, range(min(VALIDATION_SLICES, len(validation))))

# %% The model
# Models that use an operator take its parameters first: for CT, the geometry.
# default_parameters() lists everything that can be set.
model = FBPConvNet(experiment.operator_params)
print(model.cite("MLA"))

# %% Optimizer, loss, scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn = torch.nn.MSELoss()

# %% The solver
# The scheduler steps every epoch (scheduler_step="batch" for every batch).
solver = SupervisedSolver(
    model, optimizer, loss_fn, scheduler=scheduler, device=device, verbose=verbose
)
solver.set_training(DataLoader(training, batch_size=4, shuffle=True))
# Validate with the PSNR every epoch: the model with the best value is saved
# (PSNR is a score, higher is better; a loss would keep the lowest).
solver.set_validation(
    DataLoader(validation, batch_size=4),
    validation_freq=1,
    validation_fn=PSNR(),
    validation_fname="FBPConvNet_best",
)
solver.set_saving(save_folder, "FBPConvNet")
# A checkpoint every epoch; train() resumes from the latest one if it exists.
solver.set_checkpointing("FBPConvNet_check_*.pt", checkpoint_freq=1)

# %% Train and save
solver.train(EPOCHS)
solver.save_final_results()  # FBPConvNet.json (parameters, records) and .pt
solver.clean_checkpoints()

# %% The losses
fig, ax = plt.subplots()
ax.semilogy(solver.train_loss, label="training loss (MSE)")
ax.set_xlabel("epoch")
ax.legend()
ax2 = ax.twinx()
ax2.plot(solver.validation_loss, "C1", label="validation PSNR")
ax2.legend(loc="upper center")
fig.savefig(save_folder / "losses.png")

# The saved model loads anywhere, with what it was trained on:
#   model, _, _ = FBPConvNet.load(save_folder / "FBPConvNet")
