# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ferdia Sherry, Ander Biguri
# =============================================================================
"""Trains the 2DeteCT benchmark paper's denoisers, for its Plug-and-Play
methods: one, or all of them.

Kiss et al., "Benchmarking learned algorithms for computed tomography image
reconstruction tasks", 2025: DnCNN-PnP, DRUNet-PnP and GS-PnP use a Gaussian
denoiser trained on the targets of 2DeteCT (the same for every problem), at
each of the paper's noise levels; ``evaluate.py`` chooses, for each problem,
the noise level whose PnP reconstructions are best on the validation set.
With the settings of the code that produced the paper's results
(``settings.py``; the README lists what differs):

    python train_denoisers.py --denoiser DnCNN-PnP --noise_level 0.01
    python train_denoisers.py --denoiser all --noise_level all

``--denoiser``: a key of ``settings.DENOISERS``, or ``all``;
``--noise_level``: one of ``settings.NOISE_LEVELS``, or ``all``. Each is saved
in ``$LION_RESULTS_PATH/paper_scripts/2detect_benchmark/denoisers/`` as
``<denoiser>_sigma<noise level>`` (e.g. ``DnCNN-PnP_sigma0_005``; ``.pt``
and ``.json``, with its records), and the one with the best validation as
``..._best``, the one used. Checkpoints every 5 epochs: a stopped run resumes from the latest when
run again, and denoisers already trained are skipped. ``--slices`` and
``--epochs`` make a short run, to check it works.

Needs a GPU, the processed 2DeteCT data in ``LION_DATA_PATH`` and
``LION_RESULTS_PATH``.
"""

# %% Imports
import argparse

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from LION.experiments import deteCT_benchmark
from LION.optimizers.GaussianDenoiserSolver import GaussianDenoiserSolver

import settings

# %% Settings
arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
arguments.add_argument(
    "--denoiser", choices=[*settings.DENOISERS, "all"], required=True
)
arguments.add_argument(
    "--noise_level",
    choices=[*map(str, settings.NOISE_LEVELS), "all"],
    required=True,
)
arguments.add_argument("--epochs", type=int, default=settings.DENOISER_EPOCHS)
arguments.add_argument(
    "--slices", type=int, default=None, help="training slices (default: all)"
)
arguments.add_argument("--gpu", type=int, default=0)
arguments = arguments.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(arguments.gpu)
verbose = 2  # 0: nothing; 1: one progress bar; 2: also each epoch's bars and messages


class Targets(Dataset):
    """The targets alone, as the (input, target) pairs LION's solvers take: the
    solver makes the noisy inputs from the targets."""

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        return image, image


# %% One denoiser at one noise level
def train(name, noise_level):
    save_folder = settings.denoiser_folder()
    save_folder.mkdir(parents=True, exist_ok=True)
    fname = settings.denoiser_name(name, noise_level)
    if (save_folder / f"{fname}.json").exists():
        print(f"{fname}: already trained, skipped")
        return
    print(f"Training {name} at noise level {noise_level}")

    # The targets of 2DeteCT (mode 2 reconstructions), without measurements
    experiment = deteCT_benchmark.GroundTruthCT()
    training = experiment.get_dataset("train")
    validation = experiment.get_dataset("validation")
    if arguments.slices is not None:
        training = Subset(training, range(min(arguments.slices, len(training))))
        validation = Subset(
            validation, range(min(arguments.slices // 10 + 1, len(validation)))
        )
    training_loader = DataLoader(Targets(training), batch_size=1, shuffle=True)
    validation_loader = DataLoader(Targets(validation), batch_size=1)

    model_class, parameters = settings.DENOISERS[name]
    model = model_class(parameters())
    # the targets scaled to [0, 1]; the noise is added after
    model.set_normalisation("custom", data_range=settings.NORMALISATION)

    # 5 patches of 256 x 256 per image, half erased; noise at this level
    solver_params = GaussianDenoiserSolver.default_parameters()
    solver_params.noise_level = noise_level
    solver_params.patches = GaussianDenoiserSolver.patches(*settings.DENOISER_PATCHES)
    solver_params.augment = False
    solver_params.random_erasing = True
    solver_params.validation_noise_levels = [noise_level]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.DENOISER_LEARNING_RATE,
        betas=settings.DENOISER_BETAS,
    )
    # (the solver records an operator: the full-data problem's, the targets')
    solver = GaussianDenoiserSolver(
        model,
        optimizer,
        torch.nn.MSELoss(),
        operator_params=settings.EXPERIMENTS["full"]().operator_params,
        solver_params=solver_params,
        verbose=verbose,
    )
    solver.set_training(training_loader)
    # validated every epoch; the best is kept, and used
    solver.set_validation(validation_loader, 1, validation_fname=f"{fname}_best")
    solver.set_saving(save_folder, fname)
    solver.set_checkpointing(f"{fname}_checkpoint_*.pt", checkpoint_freq=5)

    solver.train(arguments.epochs)
    solver.save_final_results()
    solver.clean_checkpoints()


# %% Train what was asked for
names = (
    list(settings.DENOISERS) if arguments.denoiser == "all" else [arguments.denoiser]
)
noise_levels = (
    settings.NOISE_LEVELS
    if arguments.noise_level == "all"
    else [float(arguments.noise_level)]
)
print(deteCT_benchmark.GroundTruthCT.cite("MLA"))
for name in names:
    for noise_level in noise_levels:
        train(name, noise_level)
