# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Christina Runkel, Ander Biguri
# =============================================================================
"""Trains the cLPD paper's models: one, or all twelve.

Runkel, Biguri and Schönlieb, "Continuous Learned Primal Dual", 2024: cLPD
and LPD, each trained on six CT problems (full, sparse and limited angle, at
two doses), with the settings of the code that trained the paper's models
(``settings.py``; the README lists what differs from the paper's text and from
2024). One model on one problem, or all of them:

    python train.py --experiment full_low --model cLPD
    python train.py --experiment all --model all

``--experiment``: a key of ``settings.EXPERIMENTS``, or ``all``; ``--model``:
cLPD, LPD or ``all``. Each model is saved in
``$LION_RESULTS_PATH/paper_scripts/cLPD/<experiment>/<model>`` (``.pt`` and
``.json``, with its records), with checkpoints every 10 epochs: a stopped run
resumes from the latest when run again, and models already trained are
skipped. The model with the lowest validation loss is also kept
(``<model>_best``). ``--slices`` and ``--epochs`` make a short run, to check
it works.

Needs a GPU, the processed LIDC-IDRI data in ``LION_DATA_PATH`` and
``LION_RESULTS_PATH``.
"""

# %% Imports
import argparse

import torch
from torch.utils.data import DataLoader, Subset

from LION.optimizers.SupervisedSolver import SupervisedSolver

import settings

# %% Settings
arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
arguments.add_argument(
    "--experiment", choices=[*settings.EXPERIMENTS, "all"], required=True
)
arguments.add_argument("--model", choices=[*settings.MODELS, "all"], required=True)
arguments.add_argument("--epochs", type=int, default=settings.EPOCHS)
arguments.add_argument(
    "--slices", type=int, default=None, help="training slices (default: all)"
)
arguments.add_argument(
    "--data-folder", default=None, help="another processed LIDC-IDRI folder"
)
arguments.add_argument("--gpu", type=int, default=0)
arguments = arguments.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(arguments.gpu)
verbose = 2  # 0: nothing; 1: one progress bar; 2: also each epoch's bars and messages


# %% One model on one problem
def train(experiment_name, name):
    save_folder = settings.folder(experiment_name)
    save_folder.mkdir(parents=True, exist_ok=True)
    if (save_folder / f"{name}.json").exists():
        print(f"{experiment_name}/{name}: already trained, skipped")
        return
    print(f"Training {name} on {experiment_name}")

    # The problem: geometry, noise, data
    experiment = settings.experiment(experiment_name, arguments.data_folder)
    training = experiment.get_dataset("train")
    validation = experiment.get_dataset("validation")
    if arguments.slices is not None:
        training = Subset(training, range(min(arguments.slices, len(training))))
        validation = Subset(
            validation, range(min(arguments.slices // 10 + 1, len(validation)))
        )

    training_loader = DataLoader(training, batch_size=settings.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=settings.BATCH_SIZE)

    # The model, with the parameters it was trained with
    model_class, paper_parameters = settings.MODELS[name]
    model = model_class(experiment.operator_params, paper_parameters())

    # Training
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.LEARNING_RATE, betas=settings.BETAS
    )
    solver = SupervisedSolver(model, optimizer, settings.LOSS(), verbose=verbose)
    solver.set_training(training_loader)
    # validated every epoch with the training loss; the lowest is kept
    solver.set_validation(validation_loader, 1, validation_fname=f"{name}_best")
    solver.set_saving(save_folder, name)
    solver.set_checkpointing(f"{name}_checkpoint_*.pt", checkpoint_freq=10)

    solver.train(arguments.epochs)
    solver.save_final_results()
    solver.clean_checkpoints()


# %% Train what was asked for
experiments = (
    list(settings.EXPERIMENTS)
    if arguments.experiment == "all"
    else [arguments.experiment]
)
models = list(settings.MODELS) if arguments.model == "all" else [arguments.model]
print(settings.MODELS["cLPD"][0].cite("MLA"))
for experiment_name in experiments:
    for name in models:
        train(experiment_name, name)
