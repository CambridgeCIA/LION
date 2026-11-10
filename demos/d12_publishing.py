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
# recorded:

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.optimizers.SupervisedSolver import SupervisedSolver

experiment = ct_experiments.LowDoseCTRecon()
training = Subset(experiment.get_dataset("train"), range(40))
training_loader = DataLoader(training, batch_size=4, shuffle=True)

folder = Path(tempfile.mkdtemp())
model = FBPConvNet(experiment.operator_params)
solver = SupervisedSolver(
    model, torch.optim.Adam(model.parameters()), torch.nn.MSELoss()
)
solver.set_training(training_loader)
solver.set_saving(folder, "FBPConvNet")
solver.train(2)
solver.save_final_results()

# %%
# Publishing
# ----------
# ``publish(model_file, folder, evaluation=...)``: the evaluation (optional) is
# run on the model and its test record included. The folder must be new (or
# empty).

from LION import package
from LION.evaluation import Evaluation

evaluation = Evaluation(experiment, samples=10)
published = package.publish(
    folder / "FBPConvNet", folder / "package", evaluation=evaluation
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
# ``build`` makes the pieces: the trained model, the experiment, the training
# and validation data, the solver, the evaluation (``load_and_build`` does
# both):

built = package.build(p)
print(type(built.model).__name__, len(built.training_data))

# %%
# The published model reconstructs like any other:

from LION.reconstructors.ForwardReconstructor import ForwardReconstructor

sinogram, image = built.experiment.get_dataset("test")[0]
reconstruction = ForwardReconstructor(
    built.experiment.operator_params, built.model
).reconstruct(sinogram)

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
