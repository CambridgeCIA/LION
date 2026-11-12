# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Zakhar Shumaylov, Ander Biguri
# =============================================================================
"""Trains the 2DeteCT benchmark paper's learned regularisers (AR, ACR) and
TDV: one, or all of them.

Kiss et al., "Benchmarking learned algorithms for computed tomography image
reconstruction tasks", 2025, with the settings of the code that produced the
paper's results (Zakhar Shumaylov's, 2024; ``settings.py``; the README lists
what differs). One method on one problem, or all of them:

    python train_regularisers.py --experiment full --model ACR
    python train_regularisers.py --experiment all --model all

``--experiment``: a key of ``settings.EXPERIMENTS``, or ``all``; ``--model``:
a key of ``settings.REGULARISERS``, or ``all``. Each is saved in
``$LION_RESULTS_PATH/paper_scripts/2detect_benchmark/<experiment>/`` three
times, as ``evaluate.py`` tests them: after the last epoch (``<model>``), at
the lowest validation loss (``<model>_best``) and at the lowest training loss
(``<model>_mintrain``). Checkpoints every epoch: a stopped run resumes from
the latest when run again, and models already trained are skipped.
``--slices`` and ``--epochs`` make a short run, to check it works.

Needs a GPU, the processed 2DeteCT data in ``LION_DATA_PATH`` and
``LION_RESULTS_PATH``.
"""

# %% Imports
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from LION.models.LIONmodel import LIONmodel
from LION.optimizers.AdversarialRegularizerSolver import ARSolver
from LION.optimizers.SupervisedSolver import SupervisedSolver
from LION.reconstructors.LearnedRegularizerReconstructor import estimate_alpha

import settings

# %% Settings
arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
arguments.add_argument(
    "--experiment", choices=[*settings.EXPERIMENTS, "all"], required=True
)
arguments.add_argument(
    "--model", choices=[*settings.REGULARISERS, "all"], required=True
)
arguments.add_argument("--epochs", type=int, default=settings.REGULARISER_EPOCHS)
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
    if (save_folder / f"{name}_mintrain.json").exists():
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

    model_class, parameters, batch_size = settings.REGULARISERS[name]
    model = model_class(experiment.operator_params, parameters())
    training_loader = DataLoader(training, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.REGULARISER_LEARNING_RATE,
        betas=settings.REGULARISER_BETAS,
    )

    if name == "TDV":
        # trained end to end on the measurements; validated on all the
        # validation set, by the training loss
        solver = SupervisedSolver(
            model,
            optimizer,
            torch.nn.MSELoss(),
            operator_params=experiment.operator_params,
            verbose=verbose,
        )
        validation_loader = DataLoader(validation, batch_size=batch_size)
        solver.set_validation(validation_loader, 1, validation_fname=f"{name}_best")
    else:
        # the images scaled by the first 100 training targets' range
        targets = [
            training[i][1]
            for i in range(min(settings.NORMALISATION_SLICES, len(training)))
        ]
        x_min = min(t.min().item() for t in targets)
        x_max = max(t.max().item() for t in targets)
        model.set_normalisation("custom", data_range=(x_min, x_max))
        # clean images against FDK reconstructions; validated by the error
        # of the reconstructions of the first 50 validation slices
        solver = ARSolver(
            model,
            optimizer,
            operator_params=experiment.operator_params,
            verbose=verbose,
        )
        validation = Subset(
            validation,
            range(min(settings.REGULARISER_VALIDATION_SLICES, len(validation))),
        )
        validation_loader = DataLoader(validation, batch_size=batch_size)
        op = experiment.operator_params.make_operator()
        alpha = estimate_alpha(op, validation_loader, device=solver.device)
        print(f"{name}: alpha {alpha:.4g}, estimated on the validation slices")
        solver.set_validation(
            validation_loader,
            1,
            validation_fname=f"{name}_best",
            reconstructor=settings.reconstruction(name, op, alpha),
        )
    solver.set_training(training_loader)
    solver.set_saving(save_folder, name)
    solver.set_checkpointing(f"{name}_checkpoint_*.pt", checkpoint_freq=1)

    solver.train(arguments.epochs)
    solver.save_final_results()

    # the model at the lowest training loss, from its checkpoint
    epoch = int(np.argmin(solver.train_loss)) + 1
    checkpoint = save_folder / f"{name}_checkpoint_{epoch:04d}.pt"
    best, _, _ = LIONmodel.load_checkpoint(checkpoint, allow=settings.ALLOW)
    best.save(
        save_folder / f"{name}_mintrain",
        epoch=epoch,
        training=solver.training_record("min_train", epoch),
        loss=solver.train_loss[epoch - 1],
        dataset=experiment.data_record("train"),
    )
    solver.clean_checkpoints()


# %% Train what was asked for
experiments = (
    list(settings.EXPERIMENTS)
    if arguments.experiment == "all"
    else [arguments.experiment]
)
names = list(settings.REGULARISERS) if arguments.model == "all" else [arguments.model]
print(settings.EXPERIMENTS["full"].cite("MLA"))
for experiment_name in experiments:
    for name in names:
        train(experiment_name, name)
