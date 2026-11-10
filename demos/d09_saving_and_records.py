# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Saving, loading and records
===========================

Everything LION saves says how it was made. A model's file holds, with its
weights, **records**: of the data it was trained on, of its training, and of
the software it ran on. From them, each piece can be made again: the model
(trained, or untrained with the same initial weights), the experiment, the
data, the solver. That is what lets a training be repeated, a comparison be
checked and a model be published.

Previous: :doc:`d08_learned_regularisers`. Next: evaluation and comparison
(d10).

Needs a GPU and the processed LIDC-IDRI data in ``LION_DATA_PATH``.
"""

# %%
# A model to look at
# ------------------
# A short training, as in :doc:`d05_supervised_training`:

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.optimizers.SupervisedSolver import SupervisedSolver

experiment = ct_experiments.LowDoseCTRecon()
training = Subset(experiment.get_dataset("train"), range(16))
training_loader = DataLoader(training, batch_size=4, shuffle=True)

model = FBPConvNet(experiment.operator_params)
solver = SupervisedSolver(
    model, torch.optim.Adam(model.parameters(), lr=1e-3), torch.nn.MSELoss()
)
solver.set_training(training_loader)
folder = Path(tempfile.mkdtemp())
solver.set_saving(folder, "FBPConvNet")
solver.train(2)
solver.save_final_results()

file = folder / "FBPConvNet"
print(sorted(p.name for p in folder.iterdir()))

# %%
# The files
# ---------
# ``FBPConvNet.pt`` holds the weights (and the random state they were made
# from); ``FBPConvNet.json`` everything else, as plain JSON:

import json

print(list(json.loads(file.with_suffix(".json").read_text())))

# %%
# Loading
# -------
# ``load`` makes the model again, with its trained weights, and returns its
# options (the JSON, as parameters) and the file's other data (losses, epoch):

model, options, data = FBPConvNet.load(file)
print(type(model).__name__, data["epoch"])
print(options.model_name, list(vars(options)))

# %%
# A model can be loaded through ``LIONmodel`` too, without knowing its class:

from LION.models.LIONmodel import LIONmodel

model, _, _ = LIONmodel.load(file)
print(type(model).__name__)

# %%
# The records
# -----------
# **The data record** says what the model was trained on: the dataset, the
# experiment and split that made it, the noise and its seed, the subset's
# indices, every sample (by patient and slice) and the patients:

from LION.data_loaders.data_record import read_record

data = read_record(file)
print(data.split, data.subset.indices, data.samples.ids[:2], data.samples.groups[:3])

# %%
# **The training record** says how: the solver and its settings, the
# optimizer and its hyperparameters, the learning-rate scheduler, the seed,
# how the initial weights are made again, how the data were fed, the epochs
# and the history of the losses.

from LION.optimizers.training_record import read_training_record

training_record = read_training_record(file)
print(training_record.solver["_target_"], training_record.seed)
print(training_record.optimizer)
print(training_record.train)  # batch size, shuffle
print(training_record.epochs, training_record.history.train_loss)
print(training_record.initial_weights)

# %%
# **The environment record** says on what: LION's version and commit (and any
# uncommitted change), the packages that change the numbers, CUDA and the GPU.
# Nothing personal.

print(options.environment)

# %%
# Making things again
# -------------------
# Each piece makes itself again from the file: ``instantiate`` (with the
# recorded arguments, any of which can be replaced), and
# ``parameters_from_file`` (the recorded parameters, to change before making).
#
# The model, untrained, with the initial weights it was trained from:

again = LIONmodel.instantiate(file, initial_weights=True)

# %%
# The experiment and data it was trained on:

from LION.experiments.ct_experiments import Experiment

same_experiment = Experiment.instantiate(file)
same_data = Subset(same_experiment.get_dataset(data.split), data.subset.indices)

# %%
# The solver, with its settings, seed and optimizer, for that model. Trained
# for the same epochs on the same data, it trains the same model (exactly on a
# CPU; on a GPU, to about 1e-3, unless made with ``deterministic=True``).

from LION.optimizers.LIONsolver import LIONsolver

repeat = LIONsolver.instantiate(file, again)
repeat.set_training(
    DataLoader(
        same_data,
        batch_size=training_record.train.batch_size,
        shuffle=training_record.train.shuffle,
    )
)
repeat.set_saving(folder, "repeated")
repeat.train(training_record.epochs.asked)

trained, _, _ = LIONmodel.load(file)
difference = max(
    (a - b).abs().max().item()
    for a, b in zip(repeat.model.state_dict().values(), trained.state_dict().values())
    if a.is_floating_point()
)
print(f"largest difference in the weights: {difference:.2e}")

# %%
# ``LION.package.train_again`` does all of this in one call (the tutorial on
# publishing, d11).
#
# Changing what was recorded
# --------------------------
# ``parameters_from_file`` gives the recorded parameters to change; the model
# made with them is new (new random weights):

parameters = FBPConvNet.parameters_from_file(file)
parameters.activation = "LeakyReLU"
changed = FBPConvNet.instantiate(file, model_parameters=parameters)

# %%
# Fine-tuning
# -----------
# A model loaded from its file and trained again records where it started:
# the original's file and weights. So a fine-tuned model's file leads to its
# original's records.

model, _, _ = FBPConvNet.load(file)
extreme = ct_experiments.ExtremeLowDoseCTRecon()
tuning = SupervisedSolver(
    model, torch.optim.Adam(model.parameters(), lr=1e-4), torch.nn.MSELoss()
)
tuning.set_training(
    DataLoader(Subset(extreme.get_dataset("train"), range(16)), batch_size=4)
)
tuning.set_saving(folder, "fine_tuned")
tuning.train(1)
tuning.save_final_results()

from LION.optimizers.training_record import weights_sha256

origin = read_training_record(folder / "fine_tuned").initial_weights.origin
print(origin)
print(origin.weights_sha256 == weights_sha256(LIONmodel.load(file)[0]))

# %%
# Reconstructors
# --------------
# A reconstructor saves its settings and its model by the file the model was
# loaded from (the weights stay there). ``load`` makes both again:

from LION.reconstructors.ForwardReconstructor import ForwardReconstructor
from LION.reconstructors.LIONreconstructor import LIONReconstructor

reconstructor = ForwardReconstructor(experiment.operator_params, trained)
reconstructor.save(folder / "reconstructor.json")
reconstructor = LIONReconstructor.load(folder / "reconstructor.json")
print(type(reconstructor).__name__, type(reconstructor.model).__name__)

# %%
# Your own code
# -------------
# Files only make LION's, torch's and numpy's classes again: a file can't run
# other code. Classes of your own are allowed by their module's name, e.g. a
# model defined in your package:
#
# .. code-block:: python
#
#    model, _, _ = LIONmodel.load(file, allow=["my_package."])
#
# Loading a file saved with other LION code (another commit, or uncommitted
# changes) warns: the record says which.
