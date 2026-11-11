# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Maximilian B. Kiss, Ander Biguri
# =============================================================================
"""Trains the noise paper's models: one, or all eight.

Kiss et al., "Learned denoising with simulated and experimental low-dose CT
data", 2025: a U-Net and an MS-D network, each as a sinogram denoiser and end
to end (on the FDK of the sinogram), each trained on simulated and on
experimental low-dose 2DeteCT data, with the settings of the code that
produced the paper's results (``settings.py``; the README lists what
differs). One model on one kind of noise, or all of them:

    python train.py --model UNet --noise simulated
    python train.py --model all --noise all

``--model``: a key of ``settings.MODELS`` (UNet, MSDNet, FBP+UNet,
FBP+MSDNet), or ``all``; ``--noise``: simulated, experimental or ``all``.
Each model is saved in ``$LION_RESULTS_PATH/paper_scripts/noise_paper/``
as ``<model>_<noise>`` (``.pt`` and ``.json``, with its records), and the one
with the lowest validation loss as ``<model>_<noise>_best``, the one tested.
Checkpoints every 10 epochs: a stopped run resumes from the latest when run
again, and models already trained are skipped. ``--slices`` and ``--epochs``
make a short run, to check it works.

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
arguments.add_argument("--model", choices=[*settings.MODELS, "all"], required=True)
arguments.add_argument("--noise", choices=[*settings.NOISE, "all"], required=True)
arguments.add_argument("--epochs", type=int, default=settings.EPOCHS)
arguments.add_argument(
    "--slices", type=int, default=None, help="training slices (default: all)"
)
arguments.add_argument("--gpu", type=int, default=0)
arguments = arguments.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(arguments.gpu)
verbose = 2  # 0: nothing; 1: one progress bar; 2: also each epoch's bars and messages

save_folder = settings.folder()
save_folder.mkdir(parents=True, exist_ok=True)


# %% One model on one kind of noise
def train(model_name, noise):
    name = settings.name(model_name, noise)
    if (save_folder / f"{name}.json").exists():
        print(f"{name}: already trained, skipped")
        return
    print(f"Training {name}")

    # The data: noisy sinograms, and mode 2 sinograms or reconstructions
    task, make_model = settings.MODELS[model_name]
    training = settings.dataset(noise, task, "train")
    validation = settings.dataset(noise, task, "validation")
    if arguments.slices is not None:
        training = Subset(training, range(min(arguments.slices, len(training))))
        validation = Subset(
            validation, range(min(arguments.slices // 10 + 1, len(validation)))
        )

    training_loader = DataLoader(training, batch_size=settings.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=settings.BATCH_SIZE)

    # The model, untrained
    model = make_model()

    # Training: the solver gives the end-to-end networks the FDK of the sinogram
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.LEARNING_RATE, betas=settings.BETAS
    )
    solver = SupervisedSolver(
        model,
        optimizer,
        settings.LOSS(),
        operator_params=settings.operator_params(),
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
models = list(settings.MODELS) if arguments.model == "all" else [arguments.model]
noises = list(settings.NOISE) if arguments.noise == "all" else [arguments.noise]
for model_name in models:
    for noise in noises:
        train(model_name, noise)
