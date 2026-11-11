# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Christina Runkel, Ander Biguri
# =============================================================================
"""Trains one model of the cLPD paper on one of its problems.

Runkel, Biguri and Schönlieb, "Continuous Learned Primal Dual", 2024: cLPD
and LPD, each trained on six CT problems (full, sparse and limited angle, at
clinical and extremely low dose), with Adam (learning rate 1e-4), batch size 2,
for 100 epochs. One run trains one model on one problem:

    python train.py --experiment full_clinical --model cLPD

``--experiment`` is one of ``settings.EXPERIMENTS``, ``--model`` cLPD or LPD;
the README has the loop over all twelve. The model is saved in
``$LION_RESULTS_PATH/paper_scripts/cLPD/<experiment>/<model>`` (``.pt`` and
``.json``, with its records), with checkpoints every 10 epochs: a stopped run
resumes from the latest when run again. The model with the best validation
PSNR is also kept (``<model>_best``). ``--slices`` and ``--epochs`` make a
short run, to check it works.

Needs a GPU, the processed LIDC-IDRI data in ``LION_DATA_PATH`` and
``LION_RESULTS_PATH``.
"""

# %% Imports
import argparse

import torch
from torch.utils.data import DataLoader, Subset

from LION.metrics.psnr import PSNR
from LION.optimizers.SupervisedSolver import SupervisedSolver

import settings

# %% Settings
arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
arguments.add_argument("--experiment", choices=settings.EXPERIMENTS, required=True)
arguments.add_argument("--model", choices=settings.MODELS, required=True)
arguments.add_argument("--epochs", type=int, default=settings.EPOCHS)
arguments.add_argument(
    "--slices", type=int, default=None, help="training slices (default: all)"
)
arguments.add_argument("--gpu", type=int, default=0)
arguments = arguments.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(arguments.gpu)
verbose = 2  # 0: nothing; 1: one progress bar; 2: also each epoch's bars and messages

save_folder = settings.folder(arguments.experiment)
save_folder.mkdir(parents=True, exist_ok=True)
name = arguments.model

# %% The problem: geometry, noise, data
experiment = settings.EXPERIMENTS[arguments.experiment]()
training = experiment.get_dataset("train")
validation = experiment.get_dataset("validation")
if arguments.slices is not None:
    training = Subset(training, range(min(arguments.slices, len(training))))
    validation = Subset(
        validation, range(min(arguments.slices // 10 + 1, len(validation)))
    )

training_loader = DataLoader(training, batch_size=settings.BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation, batch_size=settings.BATCH_SIZE)

# %% The model, with the paper's parameters
model_class, paper_parameters = settings.MODELS[name]
model = model_class(experiment.operator_params, paper_parameters())
print(model.cite("MLA"))

# %% Training
optimizer = torch.optim.Adam(
    model.parameters(), lr=settings.LEARNING_RATE, betas=settings.BETAS
)
solver = SupervisedSolver(model, optimizer, settings.LOSS(), verbose=verbose)
solver.set_training(training_loader)
solver.set_validation(validation_loader, 10, PSNR(), validation_fname=f"{name}_best")
solver.set_saving(save_folder, name)
solver.set_checkpointing(f"{name}_checkpoint_*.pt", checkpoint_freq=10)

solver.train(arguments.epochs)
solver.save_final_results()
solver.clean_checkpoints()
