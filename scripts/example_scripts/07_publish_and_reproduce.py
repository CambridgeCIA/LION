# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Publishing a trained model, and making it again from what was published.

``LION.package.publish`` writes one folder with everything needed to use the
model, train it again and test it, from the records LION saved with it: the
model file, the experiment (geometry, noise, data), which samples it was
trained and validated on, the training record (solver, optimizer, scheduler,
seed, history), the test record, the software environment, citations, and a
model card (README.md) with the code to train it again.

Someone with the folder (and the data) then:

- ``load(folder)``: reads the recorded arguments of every piece, to change if
  they want (nothing is made yet);
- ``build(package)``: makes the pieces: model, experiment, data, solver,
  evaluation;
- ``train_again(package, folder)``: trains the model again from the same
  start, for the same epochs.

Publishing refuses what couldn't be reproduced: LION code with uncommitted or
unpushed changes, a test on training data, code outside LION it can't name.

Publishes the FBPConvNet of 01_supervised_training.py (run it first).

Needs: a GPU, the processed LIDC-IDRI data in ``LION_DATA_PATH``, and
``LION_RESULTS_PATH``.
"""

# %% Imports
import torch

from LION import package
from LION.evaluation import Evaluation
from LION.experiments import ct_experiments
from LION.utils.paths import results_path

# %% Settings
# The GPU to use: LION's models, data and operators all follow it
GPU = 0
if torch.cuda.is_available():
    torch.cuda.set_device(GPU)
examples = results_path("examples")
model_file = examples / "supervised" / "FBPConvNet"
folder = examples / "FBPConvNet_package"  # must not exist, or be empty

# %% Publish, with a test on 20 test slices
evaluation = Evaluation(ct_experiments.LowDoseCTRecon(), samples=20, save_folder=None)
package.publish(model_file, folder, evaluation=evaluation)
print((folder / "README.md").read_text())  # the model card

# %% What was recorded
published = package.load(folder)
print(published.model)  # the model's arguments
print(published.solver)  # the solver's: loss, seed, solver_params, ...
print(published.optimizer)  # the optimizer's hyperparameters
print(published.training)  # batch size, epochs, ...

# %% Made again, as published
built = package.load_and_build(folder)
print(type(built.model).__name__, "with its trained weights")
print(len(built.training_data), "training slices")

# %% Trained again, with a change
# e.g. another activation: the model is made with new random weights; anything
# not changed is as recorded (data, optimizer, seed, epochs).
published.model.model_parameters.activation = "LeakyReLU"
retrained = package.train_again(published, examples / "FBPConvNet_retrained")
