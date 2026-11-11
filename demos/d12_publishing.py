# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Publishing and reproducing
==========================

``LION.package.publish`` writes a trained model as a **package**: one folder
with everything needed to use it, test it and train it again, from the
records LION saved with it (:doc:`d09_saving_and_records`). Someone with the
folder and the data then loads it, builds its pieces, or trains it again.

Previous: :doc:`d11_evaluation`. Next: :doc:`d13_extending_lion`.

Needs a GPU and the processed LIDC-IDRI data in ``LION_DATA_PATH``. Publishing
also needs LION's code committed and pushed (it records the commit).

.. note::

   The models trained in these tutorials are toys: a few epochs on a few
   slices, to show how LION is used, so their results are poor (often worse
   than FDK). Trained as their papers train them (many epochs, all the data),
   they do what the papers report.
"""

# %%
# A model to publish
# ------------------
# Trained by a LION solver on a LION experiment, so that everything about it is
# recorded: here with validation and a learning-rate scheduler too, so the
# package has every piece.

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.optimizers.SupervisedSolver import SupervisedSolver

experiment = ct_experiments.LowDoseCTRecon()
training = Subset(experiment.get_dataset("train"), range(40))
validation = Subset(experiment.get_dataset("validation"), range(4))

training_loader = DataLoader(training, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation, batch_size=1)

folder = Path(tempfile.mkdtemp())
model = FBPConvNet(experiment.operator_params)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
solver = SupervisedSolver(
    model,
    optimizer,
    torch.nn.MSELoss(),
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
)
solver.set_training(training_loader)
solver.set_validation(validation_loader, 1, validation_fname="FBPConvNet_best")
solver.set_saving(folder, "FBPConvNet")
solver.train(2)
solver.save_final_results()

# %%
# Publishing
# ----------
# ``publish(file, folder, evaluation=...)``: ``file`` is the model's, or a
# reconstructor's that uses it (``reconstructor.save``), published with it.
# The evaluation (optional) is run and its test record included. The folder
# must be new (or empty).

from LION import package
from LION.evaluation import Evaluation
from LION.models.LIONmodel import LIONmodel
from LION.reconstructors.ForwardReconstructor import ForwardReconstructor

trained, _, _ = LIONmodel.load(folder / "FBPConvNet")
ForwardReconstructor(experiment.operator_params, trained).save(
    folder / "reconstructor.json"
)

evaluation = Evaluation(experiment, samples=10)
published = package.publish(
    folder / "reconstructor", folder / "package", evaluation=evaluation
)
print(sorted(p.name for p in published.iterdir()))

# %%
# The package holds the model file, the experiment, which samples it was
# trained and validated on, the training record, the test record, the
# environment, every citation, and a model card (``README.md``) with the code
# to use and train it again:

print((published / "README.md").read_text())

# %%
# It refuses what couldn't be reproduced from it, saying why: LION code with
# uncommitted or unpushed changes, a test on the training data, code outside
# LION it can't name (e.g. a model class defined in a script), data that
# aren't a LION dataset.

# %%
# Loading a package
# -----------------
# ``load`` reads the recorded arguments of every piece (the model, the
# experiment, the data, the solver, the optimizer, the training), without
# making anything, so they can be read or changed first:

p = package.load(published)
print(p.model)
print(p.optimizer)
print(p.training_data)  # which split, and which samples of it

# %%
# Every piece, one at a time
# --------------------------
# Each piece has its file in the package, and is made again from it by its
# base class's ``instantiate``, with nothing else: the model (and its
# operator), the experiment, the data, the solver (with its optimizer,
# scheduler and loss), the reconstructor and the evaluation. ``package.json``
# names the files:

files = p.records.package.files
print(files)
model_file = published / files.model

# %%
# **The model**: trained (``load``), or as its training started (the same
# architecture and initial weights), to train it again by hand.

model, _, _ = LIONmodel.load(model_file)
untrained = LIONmodel.instantiate(model_file, initial_weights=True)
print(type(model).__name__, model.model_parameters.model_input_type)

# %%
# **The operator**: its parameters are the model's (for CT, the geometry), and
# make it:

print(model.operator_params)
operator = model.operator_params.make_operator()

# %%
# **The experiment** (``experiment.json``), which makes the data, with their
# noise:

from LION.experiments.ct_experiments import Experiment

experiment_again = Experiment.instantiate(published / files.experiment)
print(experiment_again.param.name)

# %%
# **The data** (``data/training.json``, ``data/validation.json``): the split
# and the samples the model was trained and validated on, made by the
# experiment:

training_again = experiment_again.get_dataset(p.training_data.split)
if p.training_data.subset is not None:
    training_again = Subset(training_again, p.training_data.subset)
print(len(training_again), "training samples")

# The dataset alone, from its record, gives the same samples without the
# experiment's noise (a warning says so):
#   from LION.data_loaders.LIONDataset import LIONDataset
#   images = LIONDataset.instantiate(published / files.training_data)

# %%
# **The solver** (``solver.json``, the training record), with its optimizer,
# scheduler, loss and settings, for the model it is given; its data are set as
# usual (``set_training``):

from LION.optimizers.LIONsolver import LIONsolver

solver_again = LIONsolver.instantiate(published / files.solver, untrained)
print(type(solver_again).__name__, type(solver_again.loss_fn).__name__)
print(type(solver_again.optimizer).__name__, solver_again.optimizer.defaults["lr"])
print(type(solver_again.scheduler).__name__, solver_again.scheduler.step_size)

# %%
# **The reconstructor** (``reconstructor.json``), given the trained model, and
# **the evaluation** that tested it (``test.json``: its test set and
# metrics):

from LION.reconstructors.LIONreconstructor import LIONReconstructor

reconstructor = LIONReconstructor.instantiate(published / files.reconstructor, model)
evaluation_again = Evaluation.instantiate(published / files.test)

sinogram, image = experiment_again.get_dataset("test")[0]
reconstruction = reconstructor.reconstruct(sinogram)

# %%
# All at once
# -----------
# ``build`` makes all of them together, as above, with the solver's data set
# (``load_and_build`` loads and builds):

built = package.build(p)
print(type(built.model).__name__, len(built.training_data))
print(type(built.solver).__name__, type(built.reconstructor).__name__)

# %%
# Training again
# --------------
# ``train_again`` trains the model again from the same start, on the same
# data, with the same solver, seed and epochs: the published model again
# (exactly on a CPU; on a GPU, to about 1e-3).
#
# Changing a recorded argument first trains a variant: anything not changed
# stays as recorded. E.g. another activation (a new model, new random
# weights):

p.model.model_parameters.activation = "LeakyReLU"
variant = package.train_again(p, folder / "variant")
print(type(variant).__name__)
