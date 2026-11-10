# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Evaluation and comparison
=========================

A result in a paper is a method scored on test data, against others. LION
makes that one standard procedure per problem:

- ``Evaluation``: the test data, the operator and the metrics of one problem
  (an experiment). ``evaluation.run(method)`` scores any method on every test
  sample and returns a **test record**, saved and reused;
- ``Comparison``: several methods on one evaluation, against a baseline. It
  first checks, from the methods' records, that the comparison is fair, then
  writes the tables and figures a paper needs.

Previous: :doc:`d10_reconstructors`. Next: :doc:`d12_publishing`.

Needs a GPU and the processed LIDC-IDRI data in ``LION_DATA_PATH``.

.. note::

   The models trained in these tutorials are toys: a few epochs on a few
   slices, to show how LION is used, so their results are poor (often worse
   than FDK). Trained as their papers train them (many epochs, all the data),
   they do what the papers report.
"""

# %%
# Methods to compare
# ------------------
# A method is a model, a model's file, a solver, a reconstructor (or its file),
# or a function ``f(measurements, operator)``. Here two FBPConvNets trained
# briefly (one more than the other), and SIRT:

import functools
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.operators.ct import sirt
from LION.optimizers.SupervisedSolver import SupervisedSolver

experiment = ct_experiments.LowDoseCTRecon()
training = Subset(experiment.get_dataset("train"), range(40))
training_loader = DataLoader(training, batch_size=4, shuffle=True)
folder = Path(tempfile.mkdtemp())

for epochs in (1, 3):
    model = FBPConvNet(experiment.operator_params)
    solver = SupervisedSolver(
        model, torch.optim.Adam(model.parameters()), torch.nn.MSELoss(), verbose=0
    )
    solver.set_training(training_loader)
    solver.set_saving(folder, f"FBPConvNet_{epochs}")
    solver.train(epochs)
    solver.save_final_results()

# %%
# An evaluation
# -------------
# The experiment's test split, scored with PSNR, SSIM and HaarPSI. The metrics'
# range is the test images' (the same for every method). ``samples`` takes
# evenly spaced test slices (all of them by default); results are saved in
# ``save_folder``.

from LION.evaluation import Evaluation, summary

evaluation = Evaluation(experiment, samples=20, save_folder=folder / "evaluations")
record = evaluation.run(folder / "FBPConvNet_3", "FBPConvNet, 3 epochs")

for metric, values in summary(record).items():
    print(f"{metric}: {values['mean']:.3f} ± {values['std']:.3f}")

# %%
# The test record holds the score of each test sample (by patient and slice),
# what was evaluated (the model's spec and weights hash), the test data's
# record and the environment. Running the same method on the same evaluation
# again reads it instead of computing it:

print(record.scores.ids[:3], record.scores.PSNR[:3])
again = evaluation.run(folder / "FBPConvNet_3", "FBPConvNet, 3 epochs")

# %%
# A comparison
# ------------
# Methods by name, against the problem's baseline (FDK, for CT):

from LION.evaluation import Comparison

comparison = Comparison(
    evaluation,
    {
        "FBPConvNet, 1 epoch": folder / "FBPConvNet_1",
        "FBPConvNet, 3 epochs": folder / "FBPConvNet_3",
        # a partial, so it can be recorded
        "SIRT": functools.partial(sirt, num_iterations=100),
    },
)
records = comparison.run()
print(comparison.table())
print(comparison.table("latex"))

# %%
# Before running, the comparison checks each method's records and stops on
# what makes it unfair:
#
# - a model trained or validated on a test patient (by patient, not only by
#   slice);
#
# and warns about what makes it doubtful:
#
# - models trained for another problem (another geometry or noise);
# - results made with other LION code (another commit, uncommitted changes).

# %%
# Saving it
# ---------
# ``save`` writes everything: the table (markdown, LaTeX, CSV), the score of
# every sample, the test records, box plots of the scores, and images of chosen
# samples (by default the best, median and worst by the baseline's PSNR), with
# zoomed crops and the differences to the target.

saved = comparison.save(
    folder / "comparison", zoom=[(200, 200, 100, 100)], differences=True
)
print(sorted(p.name for p in saved.iterdir()))

# %%
# The figures are also there to show or change:

figures = comparison.figures()
images = comparison.images(zoom=[(200, 200, 100, 100)])

# %%
# The saved comparison makes itself again, e.g. to add a method later:
#
# .. code-block:: python
#
#    comparison = Comparison.instantiate(folder / "comparison")
#
# Data without an experiment
# --------------------------
# An evaluation can also take a dataset and an operator directly, e.g. for
# data you load yourself:
#
# .. code-block:: python
#
#    evaluation = Evaluation(dataset=my_test_data, operator=my_operator)
