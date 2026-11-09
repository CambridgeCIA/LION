# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Resuming a training, and fine-tuning a trained model.

- **Resuming:** a solver with checkpointing saves the model, the optimizer's
  and scheduler's state, the losses and the random state every few epochs.
  ``train()`` resumes from the latest checkpoint, as if never stopped: a run
  stopped and resumed gives the same model as one that wasn't. Here the "stop"
  is a first ``train(2)`` of a 4-epoch run.
- **Fine-tuning:** a model loaded from its file and trained again, e.g. on
  another problem. Its new training record names the file it started from and
  that file's weights, so the chain of trainings is kept (and published, see
  07_publish_and_reproduce.py).

Fine-tunes the FBPConvNet of 01_supervised_training.py (run it first).

Needs: a GPU, the processed LIDC-IDRI data in ``LION_DATA_PATH``, and
``LION_RESULTS_PATH``.
"""

# %% Imports
import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.optimizers.SupervisedSolver import SupervisedSolver
from LION.optimizers.training_record import read_training_record
from LION.utils import spec
from LION.utils.paths import results_path

# %% Settings
TRAINING_SLICES = 100

examples = results_path("examples")
save_folder = examples / "resume_and_fine_tune"
save_folder.mkdir(parents=True, exist_ok=True)


def training_data(experiment):
    data = experiment.get_dataset("train")
    return DataLoader(
        Subset(data, range(min(TRAINING_SLICES, len(data)))),
        batch_size=4,
        shuffle=True,
    )


def solver_for(model, experiment):
    """A solver that checkpoints every epoch in save_folder."""
    solver = SupervisedSolver(
        model, torch.optim.Adam(model.parameters(), lr=1e-3), torch.nn.MSELoss()
    )
    solver.set_training(training_data(experiment))
    solver.set_saving(save_folder, "resumed")
    solver.set_checkpointing("resumed_check_*.pt", checkpoint_freq=1)
    return solver


# %% Resuming
experiment = ct_experiments.LowDoseCTRecon()
solver_for(FBPConvNet(experiment.operator_params), experiment).train(2)  # "stopped"

# Later (another session): the same set-up. The new model's weights are replaced
# by the checkpoint's, and training goes on from epoch 2 to 4.
solver = solver_for(FBPConvNet(experiment.operator_params), experiment)
solver.train(4)
solver.save_final_results()
solver.clean_checkpoints()
print("losses per epoch, all 4:", solver.train_loss)

# %% Fine-tuning
# The FBPConvNet of 01, trained on low-dose data, tuned to extremely low dose
model, _, _ = FBPConvNet.load(examples / "supervised" / "FBPConvNet")
extreme = ct_experiments.ExtremeLowDoseCTRecon()
solver = SupervisedSolver(
    model, torch.optim.Adam(model.parameters(), lr=1e-4), torch.nn.MSELoss()
)
solver.set_training(training_data(extreme))
solver.set_saving(save_folder, "fine_tuned")
solver.train(2)
solver.save_final_results()

# Its record says where its training started: the file (paths inside
# LION_RESULTS_PATH are saved relative to it) and the weights it held
start = read_training_record(save_folder / "fine_tuned").initial_weights
print("started from", spec.build(start.origin.file), start.origin.weights_sha256)
