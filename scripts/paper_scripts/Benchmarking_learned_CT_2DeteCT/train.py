# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Maximilian B. Kiss, Ander Biguri
# =============================================================================
"""Trains the 2DeteCT benchmark paper's learned methods: one, or all of them.

Kiss et al., "Benchmarking learned algorithms for computed tomography image
reconstruction tasks", 2025: learned methods trained on nine problems of the
measured 2DeteCT data (full data, limited and sparse angle, low dose, beam
hardening), with the settings of the code that produced the paper's results
(``settings.py``; the README lists what differs). One method on one problem,
or all of them:

    python train.py --experiment full --model LPD
    python train.py --experiment all --model all

``--experiment``: a key of ``settings.EXPERIMENTS``, or ``all``; ``--model``:
a key of ``settings.MODELS``, or ``all``. Each model is saved in
``$LION_RESULTS_PATH/paper_scripts/2detect_benchmark/<experiment>/<model>``
(``.pt`` and ``.json``, with its records), and the one with the lowest
validation loss as ``<model>_best``, the one tested. Checkpoints every 10
epochs: a stopped run resumes from the latest when run again, and models
already trained are skipped. ``--slices`` and ``--epochs`` make a short run,
to check it works.

Needs a GPU, the processed 2DeteCT data in ``LION_DATA_PATH`` and
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
arguments.add_argument("--gpu", type=int, default=0)
arguments = arguments.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(arguments.gpu)
verbose = 2  # 0: nothing; 1: one progress bar; 2: also each epoch's bars and messages


# %% One method on one problem
def train(experiment_name, name):
    save_folder = settings.folder(experiment_name)
    save_folder.mkdir(parents=True, exist_ok=True)
    if (save_folder / f"{name}.json").exists():
        print(f"{experiment_name}/{name}: already trained, skipped")
        return
    print(f"Training {name} on {experiment_name}")

    # The problem: the measured data, their geometry, the splits
    experiment = settings.EXPERIMENTS[experiment_name]()
    training = experiment.get_dataset("train")
    validation = experiment.get_dataset("validation")
    if arguments.slices is not None:
        training = Subset(training, range(min(arguments.slices, len(training))))
        validation = Subset(
            validation, range(min(arguments.slices // 10 + 1, len(validation)))
        )

    training_loader = DataLoader(training, batch_size=settings.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=settings.BATCH_SIZE)

    # The model, with the parameters it was trained with: networks on FDK
    # take only their parameters, unrolled methods also the operator's
    model_class, parameters, uses_operator, learning_rate = settings.MODELS[name]
    if uses_operator:
        model = model_class(experiment.operator_params, parameters())
    else:
        model = model_class(parameters())

    # Training: the solver gives networks on FDK the FDK of the measurements
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=settings.BETAS
    )
    solver = SupervisedSolver(
        model,
        optimizer,
        settings.LOSS(),
        operator_params=experiment.operator_params,
        verbose=verbose,
    )
    solver.set_training(training_loader)
    # validated with the training loss; the lowest is kept, and tested
    solver.set_validation(
        validation_loader, settings.VALIDATION_EVERY, validation_fname=f"{name}_best"
    )
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
print(settings.EXPERIMENTS["full"].cite("MLA"))
for experiment_name in experiments:
    for name in models:
        train(experiment_name, name)
